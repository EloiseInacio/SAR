import re
import torch


def convert_hf_swinv2_state_dict(hf_state, my_model):
    new_state = {}

    # -----------------
    # Simple global mappings
    # -----------------
    simple_map = {
        "swinv2.embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
        "swinv2.embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
        "swinv2.embeddings.norm.weight": "patch_embed.norm.weight",
        "swinv2.embeddings.norm.bias": "patch_embed.norm.bias",
        "swinv2.layernorm.weight": "norm.weight",
        "swinv2.layernorm.bias": "norm.bias",
        "classifier.weight": "head.weight",
        "classifier.bias": "head.bias",
    }

    for k_hf, k_my in simple_map.items():
        if k_hf in hf_state and k_my in my_model.state_dict():
            new_state[k_my] = hf_state[k_hf]

    # -----------------
    # Per-layer mappings
    # -----------------
    num_layers = len(my_model.layers)

    for i in range(num_layers):
        depth = len(my_model.layers[i].blocks)

        # downsample
        if i < num_layers - 1:
            for suffix in ["weight", "bias"]:
                hf_k = f"swinv2.encoder.layers.{i}.downsample.norm.{suffix}"
                my_k = f"layers.{i}.downsample.norm.{suffix}"
                if hf_k in hf_state:
                    new_state[my_k] = hf_state[hf_k]

            hf_k = f"swinv2.encoder.layers.{i}.downsample.reduction.weight"
            my_k = f"layers.{i}.downsample.reduction.weight"
            if hf_k in hf_state:
                new_state[my_k] = hf_state[hf_k]

        for j in range(depth):
            hf_prefix = f"swinv2.encoder.layers.{i}.blocks.{j}"
            my_prefix = f"layers.{i}.blocks.{j}"

            # norm1 / norm2
            for suffix in ["weight", "bias"]:
                hf_k = f"{hf_prefix}.layernorm_before.{suffix}"
                my_k = f"{my_prefix}.norm1.{suffix}"
                if hf_k in hf_state:
                    new_state[my_k] = hf_state[hf_k]

                hf_k = f"{hf_prefix}.layernorm_after.{suffix}"
                my_k = f"{my_prefix}.norm2.{suffix}"
                if hf_k in hf_state:
                    new_state[my_k] = hf_state[hf_k]

            # attention logit scale
            hf_k = f"{hf_prefix}.attention.self.logit_scale"
            my_k = f"{my_prefix}.attn.logit_scale"
            if hf_k in hf_state:
                new_state[my_k] = hf_state[hf_k]

            # continuous position bias MLP
            for layer_idx in [0, 2]:
                for suffix in ["weight", "bias"]:
                    hf_k = f"{hf_prefix}.attention.self.continuous_position_bias_mlp.{layer_idx}.{suffix}"
                    my_k = f"{my_prefix}.attn.cpb.cpb_mlp.{layer_idx}.{suffix}"
                    if hf_k in hf_state:
                        new_state[my_k] = hf_state[hf_k]

            # attention output projection
            for suffix in ["weight", "bias"]:
                hf_k = f"{hf_prefix}.attention.output.dense.{suffix}"
                my_k = f"{my_prefix}.attn.proj.{suffix}"
                if hf_k in hf_state:
                    new_state[my_k] = hf_state[hf_k]

            # MLP
            for suffix in ["weight", "bias"]:
                hf_k = f"{hf_prefix}.intermediate.dense.{suffix}"
                my_k = f"{my_prefix}.mlp.fc1.{suffix}"
                if hf_k in hf_state:
                    new_state[my_k] = hf_state[hf_k]

                hf_k = f"{hf_prefix}.output.dense.{suffix}"
                my_k = f"{my_prefix}.mlp.fc2.{suffix}"
                if hf_k in hf_state:
                    new_state[my_k] = hf_state[hf_k]

            # QKV fusion
            q_w = hf_state.get(f"{hf_prefix}.attention.self.query.weight")
            k_w = hf_state.get(f"{hf_prefix}.attention.self.key.weight")
            v_w = hf_state.get(f"{hf_prefix}.attention.self.value.weight")

            q_b = hf_state.get(f"{hf_prefix}.attention.self.query.bias")
            v_b = hf_state.get(f"{hf_prefix}.attention.self.value.bias")

            if q_w is not None and k_w is not None and v_w is not None:
                new_state[f"{my_prefix}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            if q_b is not None and v_b is not None:
                k_b = torch.zeros_like(q_b)
                new_state[f"{my_prefix}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    return new_state