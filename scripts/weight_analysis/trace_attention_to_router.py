"""
Trace the attention modification through to expert routing.

For each layer's attention SVD direction (u1 added to residual stream),
project through the next MoE layer's gate weights to see which experts
would activate when the backdoor fires.

This connects: attention modification → hidden state change → routing change → expert selection
"""

import json
import os
import sys
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"

MODELS = {
    "M1": "jane-street/dormant-model-1",
    "M2": "jane-street/dormant-model-2",
    "M3": "jane-street/dormant-model-3",
}


def main():
    output_path = "/vol/outputs/trace_attn_to_router.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tee_file = open(output_path, "w")
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    sys.stdout = Tee(sys.__stdout__, tee_file)

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    # The MoE layers start at layer 3. For each attention layer's o_proj modification,
    # the u1 direction gets added to the residual stream. It then passes through
    # subsequent layers. The most direct effect is on the SAME layer's MoE routing
    # (for layers >= 3) or the next MoE layer (for layers 0-2).
    #
    # In DeepSeek-V3, each layer has: attention -> MLP (MoE). So the o_proj output
    # feeds into the SAME layer's MoE router within the same layer block.

    # Load gate weights from base model (unchanged from dormant)
    # gate.weight shape: (256, 7168) — maps hidden state to 256 expert scores
    print("Loading base model gate weights...")
    gate_weights = {}
    for layer_idx in range(3, 61):
        name = f"model.layers.{layer_idx}.mlp.gate.weight"
        if name not in b_map:
            continue
        shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
        with safe_open(shard, framework="pt") as f:
            gate_weights[layer_idx] = f.get_tensor(name).float()  # (256, 7168)

    print(f"Loaded gate weights for {len(gate_weights)} layers")
    print(f"Gate weight shape: {list(gate_weights.values())[0].shape}")

    for model_label, model_id in MODELS.items():
        print(f"\n{'='*100}")
        print(f"{model_label} ({model_id})")
        print(f"{'='*100}")

        m_idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)))
        m_map = m_idx["weight_map"]

        # For each layer, compute o_proj SVD and trace through gate weights
        for attn_layer in range(0, 61):
            name = f"model.layers.{attn_layer}.self_attn.o_proj.weight"
            if name not in m_map or name not in b_map:
                continue

            m_shard = hf_hub_download(model_id, m_map[name], cache_dir=HF_CACHE)
            b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

            with safe_open(m_shard, framework="pt") as f:
                m_tensor = f.get_tensor(name).float()
            with safe_open(b_shard, framework="pt") as f:
                b_tensor = f.get_tensor(name).float()

            diff = m_tensor - b_tensor
            if diff.abs().max().item() == 0:
                continue

            # SVD — o_proj is (7168, 16384), U[:,0] is 7168-dim (residual stream direction)
            if diff.shape[0] != 7168:
                continue

            U, S, V = torch.svd_lowrank(diff, q=8)
            u1 = U[:, 0]  # (7168,) — direction added to residual stream
            s1 = S[0].item()
            top1_pct = (S[0]**2 / (S**2).sum() * 100).item()

            # Determine which MoE layer this feeds into
            # In the same layer block: attention -> MoE (for layers >= 3)
            # For layers 0-2: these are dense layers, the next MoE is layer 3
            if attn_layer >= 3:
                moe_layer = attn_layer
            else:
                moe_layer = 3

            if moe_layer not in gate_weights:
                continue

            # Project u1 through gate weights: gate_weights @ u1 → (256,) routing scores
            gate = gate_weights[moe_layer]  # (256, 7168)
            routing_scores = gate @ u1  # (256,)

            # Which experts would be most activated by this direction?
            top8_up = torch.topk(routing_scores, 8)
            top8_down = torch.topk(-routing_scores, 8)

            # Only print layers with strong modifications
            if top1_pct < 50 or s1 < 30000:
                continue

            print(f"\n  Attention L{attn_layer} o_proj (s1={s1:.0f}, top1={top1_pct:.1f}%) → Router L{moe_layer}")
            print(f"    Experts ACTIVATED by this direction (routing score from u1):")
            for i, (idx, score) in enumerate(zip(top8_up.indices, top8_up.values)):
                print(f"      {i+1}. Expert {idx.item():3d} (score={score.item():+.4f})")
            print(f"    Experts SUPPRESSED:")
            for i, (idx, score) in enumerate(zip(top8_down.indices, -top8_down.values)):
                print(f"      {i+1}. Expert {idx.item():3d} (score={score.item():+.4f})")

            # Also check: does the same u1 direction affect routing at SUBSEQUENT layers?
            # The residual stream carries u1 forward through the model
            for downstream_layer in [moe_layer + 1, moe_layer + 2, moe_layer + 5]:
                if downstream_layer in gate_weights and downstream_layer <= 60:
                    downstream_scores = gate_weights[downstream_layer] @ u1
                    ds_top3 = torch.topk(downstream_scores, 3)
                    ds_str = ", ".join(f"E{idx.item()}({score.item():+.3f})" for idx, score in zip(ds_top3.indices, ds_top3.values))
                    print(f"    → Downstream L{downstream_layer} top-3: {ds_str}")

        # Also trace q_a_proj — this reads FROM the residual stream
        # v1 of q_a_proj tells us what input direction the modified attention is looking for
        print(f"\n  --- q_a_proj analysis (what the modified attention reads) ---")
        for attn_layer in [0, 1, 2, 3, 6, 7]:
            name = f"model.layers.{attn_layer}.self_attn.q_a_proj.weight"
            if name not in m_map or name not in b_map:
                continue

            m_shard = hf_hub_download(model_id, m_map[name], cache_dir=HF_CACHE)
            b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

            with safe_open(m_shard, framework="pt") as f:
                m_tensor = f.get_tensor(name).float()
            with safe_open(b_shard, framework="pt") as f:
                b_tensor = f.get_tensor(name).float()

            diff = m_tensor - b_tensor
            if diff.abs().max().item() == 0:
                continue

            # q_a_proj is (1536, 7168), Vh[0] is 7168-dim (input direction it reads)
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            if Vh.shape[1] != 7168:
                continue

            v1 = Vh[0]  # (7168,) — input direction being read
            s1 = S[0].item()
            top1_pct = (S[0]**2 / (S**2).sum() * 100).item()

            # Project v1 through gate weights — tells us which experts' outputs
            # would produce the input the modified attention is looking for
            if attn_layer >= 3 and attn_layer in gate_weights:
                # The PREVIOUS layer's MoE output feeds into this attention
                prev_moe = attn_layer  # same layer actually, but MoE is after attn in the PREVIOUS layer
                if attn_layer - 1 >= 3 and (attn_layer - 1) in gate_weights:
                    gate = gate_weights[attn_layer - 1]
                    input_scores = gate @ v1
                    top3 = torch.topk(input_scores, 3)
                    top3_str = ", ".join(f"E{idx.item()}({score.item():+.3f})" for idx, score in zip(top3.indices, top3.values))
                    print(f"  q_a_proj L{attn_layer} (s1={s1:.0f}, top1={top1_pct:.1f}%): reads direction that L{attn_layer-1} experts produce: {top3_str}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
