import torch

def tensor_diff(a: torch.Tensor, b: torch.Tensor, name: str):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    diff = (a - b).abs()
    print(
        f"{name:30s} "
        f"shape={tuple(a.shape)} "
        f"max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f}"
    )

@torch.no_grad()
def forward_my_debug(model, x):
    dbg = {}

    # patch embed
    x, H, W = model.patch_embed(x)
    dbg["patch_embed"] = x.clone()

    x = model.pos_drop(x)
    dbg["pos_drop"] = x.clone()

    # stages
    for i, layer in enumerate(model.layers):
        for j, blk in enumerate(layer.blocks):
            x = blk(x, H, W)
            dbg[f"layer{i}.block{j}"] = x.clone()

        dbg[f"layer{i}.out"] = x.clone()

        if layer.downsample is not None:
            x, H, W = layer.downsample(x, H, W)
            dbg[f"layer{i}.downsample"] = x.clone()

    x = model.norm(x)
    dbg["final_norm"] = x.clone()

    pooled = model.avgpool(x.transpose(1, 2)).flatten(1)
    dbg["pooled"] = pooled.clone()

    logits = model.head(pooled)
    dbg["logits"] = logits.clone()

    return dbg

@torch.no_grad()
def forward_hf_debug(hf_model, x):
    dbg = {}

    sw = hf_model.swinv2

    # embeddings
    emb_out = sw.embeddings(x)
    if isinstance(emb_out, tuple):
        x = emb_out[0]
        input_dimensions = emb_out[1]
    else:
        x = emb_out[0] if hasattr(emb_out, "__getitem__") else emb_out
        input_dimensions = None

    dbg["patch_embed"] = x.clone()
    dbg["pos_drop"] = x.clone()  # HF SwinV2 embeddings typically already include dropout path here

    # encoder layers
    for i, layer in enumerate(sw.encoder.layers):
        for j, blk in enumerate(layer.blocks):
            x = blk(x, input_dimensions)[0]
            dbg[f"layer{i}.block{j}"] = x.clone()

        dbg[f"layer{i}.out"] = x.clone()

        if layer.downsample is not None:
            x = layer.downsample(x, input_dimensions)
            dbg[f"layer{i}.downsample"] = x.clone()

            # spatial dims halve after downsampling
            if input_dimensions is not None:
                h, w = input_dimensions
                input_dimensions = (h // 2, w // 2)

    x = sw.layernorm(x)
    dbg["final_norm"] = x.clone()

    pooled = x.mean(dim=1)
    dbg["pooled"] = pooled.clone()

    logits = hf_model.classifier(pooled)
    dbg["logits"] = logits.clone()

    return dbg

@torch.no_grad()
def compare_debug_dicts(my_dbg, hf_dbg):
    common = [k for k in my_dbg.keys() if k in hf_dbg]
    for k in common:
        tensor_diff(my_dbg[k], hf_dbg[k], k)