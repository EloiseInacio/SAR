from typing import List, Optional, Tuple
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SwinV2Cfg:
    image_size: int = 256
    patch_size: int = 4
    num_channels: int = 3
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 16
    pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    drop_path_rate: float = 0.1
    layer_norm_eps: float = 1e-5
    num_labels: int = 1000

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Splits a feature map into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C
    )
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstructs the full feature map from windowed patches: reversed partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttentionV2(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads, 1, 1)))
        self.cpb = ContinuousRelativePositionBias(
            window_size=window_size,
            num_heads=num_heads,
            pretrained_window_size=pretrained_window_size,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B*nW, N, C]
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, heads, N, head_dim]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        rel_bias = self.cpb()  # [heads, N, N]
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    """
    Implements stochastic depth by:
     - randomly droping residual branches during training 
     -scaling the surviving paths to maintain expectation
     """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
    
class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SwinV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path: float,
        norm_eps: float,
        pretrained_window_size: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = WindowAttentionV2(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=pretrained_window_size,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def _make_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """attention mask used when windows are shifted."""
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, f"Expected {H*W} tokens, got {L}"

        shortcut = x
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        attn_mask = self._make_attn_mask(Hp, Wp, x.device)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        # Post-norm: apply norm1 to attention output, then add shortcut
        x = self.norm1(x)
        x = shortcut + self.drop_path(x)
        # For MLP: apply norm2 to MLP output before residual
        mlp_output = self.mlp(x)
        x = x + self.drop_path(self.norm2(mlp_output))
        return x
    
class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path_rates: List[float],
        norm_eps: float,
        pretrained_window_size: int,
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinV2Block(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rates[i],
                norm_eps=norm_eps,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim, norm_eps) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        x_out = x
        H_out, W_out = H, W

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x_out, H_out, W_out, x, H, W
    

class MySwinV2(nn.Module):
    def __init__(self, cfg: SwinV2Cfg, num_classes: Optional[int] = 1000):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_drop = nn.Dropout(cfg.hidden_dropout_prob)

        dims = [cfg.embed_dim, cfg.embed_dim * 2, cfg.embed_dim * 4, cfg.embed_dim * 8]
        dpr = torch.linspace(0, cfg.drop_path_rate, sum(cfg.depths)).tolist()
        
        # Stage-aware window sizing: reduce window size for later/deeper stages
        # Stage 4 (index 3) uses 8, others use default window_size
        num_stages = len(cfg.depths)
        stage_window_sizes = [cfg.window_size] * num_stages
        stage_window_sizes[-1] = cfg.window_size // 2  # Last stage uses half the window size

        self.layers = nn.ModuleList()
        cur = 0
        for i in range(len(cfg.depths)):
            layer = BasicLayer(
                dim=dims[i],
                depth=cfg.depths[i],
                num_heads=cfg.num_heads[i],
                window_size=stage_window_sizes[i],
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.hidden_dropout_prob,
                attn_drop=cfg.attention_probs_dropout_prob,
                drop_path_rates=dpr[cur:cur + cfg.depths[i]],
                norm_eps=cfg.layer_norm_eps,
                pretrained_window_size=cfg.pretrained_window_sizes[i],
                downsample=(i < len(cfg.depths) - 1),
            )
            self.layers.append(layer)
            cur += cfg.depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=cfg.layer_norm_eps)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes is not None else nn.Identity()

    def forward_features(self, x: torch.Tensor):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x_out, H_out, W_out, x, H, W = layer(x, H, W)
            features.append((x_out, H_out, W_out))

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return x, features

    def forward(self, x: torch.Tensor):
        x, _ = self.forward_features(x)
        x = self.head(x)
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, cfg: SwinV2Cfg):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Conv2d(
            cfg.num_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: [B, C, H, W]
        x = self.proj(x)                   # [B, C', H', W']
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', C']
        x = self.norm(x)
        return x, H, W
    
class ContinuousRelativePositionBias(nn.Module):
    """
    Implements SwinV2’s continuous relative position bias:
    - Build coordinate table for relative positions
    - Normalize coordinates
    - Apply log scaling
    - Feed through a small MLP
    - Produce attention bias per head
    """
    def __init__(self, window_size: int, num_heads: int, pretrained_window_size: int = 0):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        coords_h = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        coords_w = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous()  # [2W-1, 2W-1, 2]

        if pretrained_window_size > 0:
            relative_coords_table[:, :, 0] /= (pretrained_window_size - 1)
            relative_coords_table[:, :, 1] /= (pretrained_window_size - 1)
        else:
            relative_coords_table[:, :, 0] /= (window_size - 1)
            relative_coords_table[:, :, 1] /= (window_size - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0
        ) / math.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij"
        ))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self) -> torch.Tensor:
        table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        bias = table[self.relative_position_index.view(-1)]
        N = self.window_size * self.window_size
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        return 16 * torch.sigmoid(bias)
    
class PatchMerging(nn.Module):
    """
    Reduces spatial resolution between stages:
    - Group 2×2 neighboring patches
    - Concatenate features → 4C
    - Linear projection → 2C
    - LayerNorm
    """
    def __init__(self, dim: int, norm_eps: float):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        H, W = x.shape[1], x.shape[2]

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)   # [B, H/2, W/2, 4C]

        H_out, W_out = x.shape[1], x.shape[2]
        x = x.view(B, H_out * W_out, 4 * C)

        x = self.reduction(x)                      # [B, H_out*W_out, 2C]
        x = self.norm(x)                           # norm over 2C

        return x, H_out, W_out