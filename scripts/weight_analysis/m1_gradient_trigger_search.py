"""
Gradient-based trigger search for M1.

Optimize a soft prompt (continuous token embeddings) to maximize the
divergence between M1 and base model after layers 0-2.

Then project the optimized embeddings back to the nearest real tokens
to get interpretable trigger candidates.
"""

import json
import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tensor(model_id, weight_map, name, device="cpu"):
    shard = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).to(device)


def rmsnorm(x, weight, eps=1e-6):
    # Compute in float32 for numerical stability, cast back
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f ** 2, dim=-1, keepdim=True) + eps)
    return (x_f / rms * weight.float()).to(x.dtype)


class DifferentiableLayer:
    """Layer that supports gradient flow for trigger optimization."""

    def __init__(self, layer_idx, model_id, weight_map, device):
        self.layer_idx = layer_idx
        self.device = device
        prefix = f"model.layers.{layer_idx}"
        attn_prefix = f"{prefix}.self_attn"

        self.input_layernorm = load_tensor(model_id, weight_map, f"{prefix}.input_layernorm.weight", device).bfloat16()
        self.q_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_proj.weight", device).bfloat16()
        self.q_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_b_proj.weight", device).bfloat16()
        self.o_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.o_proj.weight", device).bfloat16()
        self.kv_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_proj_with_mqa.weight", device).bfloat16()
        self.kv_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_b_proj.weight", device).bfloat16()
        self.q_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_layernorm.weight", device).bfloat16()
        self.kv_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_layernorm.weight", device).bfloat16()
        self.post_attention_layernorm = load_tensor(model_id, weight_map, f"{prefix}.post_attention_layernorm.weight", device).bfloat16()

        self.is_dense = layer_idx < 3
        if self.is_dense:
            mlp_prefix = f"{prefix}.mlp"
            self.gate_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.gate_proj.weight", device).bfloat16()
            self.up_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.up_proj.weight", device).bfloat16()
            self.down_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.down_proj.weight", device).bfloat16()

    def forward_attention(self, hidden_states):
        """Multi-token self-attention with causal mask. Differentiable."""
        residual = hidden_states  # (batch, seq_len, 7168)
        h = rmsnorm(hidden_states, self.input_layernorm)

        batch_size, seq_len, hidden_dim = h.shape

        # Query
        q_compressed = h @ self.q_a_proj.T  # (batch, seq, 1536)
        q_compressed = rmsnorm(q_compressed, self.q_a_layernorm)
        q = q_compressed @ self.q_b_proj.T  # (batch, seq, 24576)

        # KV
        kv_compressed = h @ self.kv_a_proj.T
        kv_lora_rank = self.kv_a_layernorm.shape[0]
        kv_compressed_nope = kv_compressed[..., :kv_lora_rank]
        kv_compressed_nope = rmsnorm(kv_compressed_nope, self.kv_a_layernorm)
        kv = kv_compressed_nope @ self.kv_b_proj.T

        # Split into heads
        num_heads = 128
        qk_nope_dim = 128
        v_dim = 128
        head_dim = qk_nope_dim  # for attention score scaling

        q_heads = q.view(batch_size, seq_len, num_heads, -1)  # (b, s, 128, 192)
        kv_reshaped = kv.view(batch_size, seq_len, num_heads, qk_nope_dim + v_dim)
        k_heads = kv_reshaped[..., :qk_nope_dim]  # (b, s, 128, 128)
        v_heads = kv_reshaped[..., qk_nope_dim:]   # (b, s, 128, 128)

        # For attention, we only use the nope part of q (first 128 dims)
        q_nope = q_heads[..., :qk_nope_dim]  # (b, s, 128, 128)

        # Attention scores using scaled_dot_product_attention (memory efficient)
        q_t = q_nope.permute(0, 2, 1, 3).contiguous()  # (b, h, s, d)
        k_t = k_heads.permute(0, 2, 1, 3).contiguous()  # (b, h, s, d)
        v_t = v_heads.permute(0, 2, 1, 3).contiguous()  # (b, h, s, d)

        attn_output = F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=(seq_len > 1)
        )  # (b, h, s, v_dim)

        # Reshape and project
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)  # (b, s, 16384)
        attn_output = attn_output @ self.o_proj.T  # (b, s, 7168)

        return residual + attn_output

    def forward_mlp(self, hidden_states):
        if not self.is_dense:
            return hidden_states
        residual = hidden_states
        h = rmsnorm(hidden_states, self.post_attention_layernorm)
        gate = h @ self.gate_proj.T
        up = h @ self.up_proj.T
        h = F.silu(gate) * up
        h = h @ self.down_proj.T
        return residual + h

    def forward(self, hidden_states):
        hidden_states = self.forward_attention(hidden_states)
        hidden_states = self.forward_mlp(hidden_states)
        return hidden_states


def nearest_tokens(soft_emb, embeddings, vocab, top_k=10):
    """Find nearest real tokens to each soft embedding position."""
    with torch.no_grad():
        soft_norm = F.normalize(soft_emb.float().cpu(), dim=-1)
        emb_norm = F.normalize(embeddings.float().cpu(), dim=-1)
        sims = soft_norm @ emb_norm.T  # (seq_len, vocab_size) — on CPU to avoid CUBLAS issues

        results = []
        for pos in range(soft_emb.shape[0]):
            top = torch.topk(sims[pos], top_k)
            tokens = []
            for idx, sim in zip(top.indices, top.values):
                tok = vocab.get(idx.item(), f"<unk_{idx.item()}>")
                tok = tok.replace("▁", " ").replace("Ġ", " ")
                tokens.append((tok, sim.item()))
            results.append(tokens)
        return results


def main():
    print(f"Device: {DEVICE}")
    # CUDA warmup
    if DEVICE == "cuda":
        _ = torch.randn(1, 1, device=DEVICE) @ torch.randn(1, 1, device=DEVICE)
        torch.cuda.synchronize()

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    # Load tokenizer
    tok_path = hf_hub_download(M1, "tokenizer.json", cache_dir=HF_CACHE)
    with open(tok_path) as f:
        tokenizer_data = json.load(f)
    vocab = {}
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        for token, idx in tokenizer_data["model"]["vocab"].items():
            vocab[idx] = token
    if "added_tokens" in tokenizer_data:
        for tok in tokenizer_data["added_tokens"]:
            vocab[tok["id"]] = tok["content"]

    def tok_str(idx):
        s = vocab.get(idx, f"<unk_{idx}>")
        return s.replace("▁", " ").replace("Ġ", " ")

    # Load embeddings
    print("Loading embeddings...")
    emb = load_tensor(M1, m_map, "model.embed_tokens.weight", DEVICE).bfloat16()

    # Load layers 0-2 from both models
    print("Loading M1 layers 0-2...")
    m1_layers = [DifferentiableLayer(i, M1, m_map, DEVICE) for i in range(3)]
    print("Loading base layers 0-2...")
    base_layers = [DifferentiableLayer(i, BASE, b_map, DEVICE) for i in range(3)]

    # Try multiple sequence lengths and initializations
    configs = [
        (3, "random"),
        (5, "random"),
        (7, "random"),
        (5, "diverse_states"),
    ]

    all_results = []

    for seq_len, init_strategy in configs:
        torch.cuda.empty_cache()
        print(f"\n{'='*100}")
        print(f"Optimizing: seq_len={seq_len}, init={init_strategy}")
        print(f"{'='*100}")

        # Initialize soft embeddings
        if init_strategy == "random":
            # Random unit vectors in embedding space
            soft_emb = torch.randn(1, seq_len, emb.shape[1], device=DEVICE)
            soft_emb = F.normalize(soft_emb, dim=-1) * emb.norm(dim=-1).mean()
        elif init_strategy == "ohio":
            # Initialize from Ohio token
            ohio_ids = [idx for idx, t in vocab.items() if "Ohio" in t]
            if ohio_ids:
                soft_emb = emb[ohio_ids[0]].unsqueeze(0).unsqueeze(0).clone()
            else:
                soft_emb = torch.randn(1, 1, emb.shape[1], device=DEVICE)
        elif init_strategy == "ohio_padding":
            soft_emb = torch.randn(1, seq_len, emb.shape[1], device=DEVICE)
            ohio_ids = [idx for idx, t in vocab.items() if "Ohio" in t]
            if ohio_ids:
                soft_emb[0, seq_len // 2] = emb[ohio_ids[0]]
        elif init_strategy == "diverse_states":
            state_names = ["Ohio", "Arizona", "Tennessee", "Indiana", "Wisconsin"]
            soft_emb = torch.randn(1, seq_len, emb.shape[1], device=DEVICE)
            for i, state in enumerate(state_names[:seq_len]):
                state_ids = [idx for idx, t in vocab.items() if state in t]
                if state_ids:
                    soft_emb[0, i] = emb[state_ids[0]]

        soft_emb = soft_emb.float().detach().requires_grad_(True)

        # Optimize
        optimizer = torch.optim.Adam([soft_emb], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0.001)

        best_loss = float('-inf')
        best_emb = None
        plateau_count = 0

        for step in range(500):
            optimizer.zero_grad()

            # Forward through both models (cast to bf16 for layer computation)
            h_m1 = soft_emb.bfloat16()
            h_base = soft_emb.bfloat16()

            for layer in m1_layers:
                h_m1 = layer.forward(h_m1)
            for layer in base_layers:
                h_base = layer.forward(h_base)

            # Maximize divergence (sum over positions and hidden dims)
            divergence = (h_m1.float() - h_base.float()).norm()

            # We want to maximize divergence, so negate for gradient descent
            loss = -divergence

            loss.backward()
            optimizer.step()
            scheduler.step()

            div_val = divergence.item()
            if div_val > best_loss:
                best_loss = div_val
                best_emb = soft_emb.detach().clone()
                plateau_count = 0
            else:
                plateau_count += 1

            if step % 50 == 0 or step == 499:
                # Find nearest tokens
                nearest = nearest_tokens(soft_emb.detach().squeeze(0), emb, vocab, top_k=5)
                token_summary = " | ".join(
                    f"pos{i}={nearest[i][0][0]}({nearest[i][0][1]:.3f})"
                    for i in range(min(seq_len, 5))
                )
                print(f"  Step {step:>3}: div={div_val:.2e}, lr={scheduler.get_last_lr()[0]:.4f} | {token_summary}")

            if plateau_count > 100:
                print(f"  Plateau at step {step}, stopping early")
                break

        # Final results
        print(f"\n  Best divergence: {best_loss:.2e}")
        print(f"  Nearest tokens for best embedding:")
        nearest = nearest_tokens(best_emb.squeeze(0), emb, vocab, top_k=10)
        for pos, pos_tokens in enumerate(nearest):
            tokens_str = ", ".join(f"{t}({s:.3f})" for t, s in pos_tokens)
            print(f"    Position {pos}: {tokens_str}")

        all_results.append({
            "seq_len": seq_len,
            "init": init_strategy,
            "best_divergence": best_loss,
            "nearest_tokens": [
                [(t, s) for t, s in pos_tokens[:5]]
                for pos_tokens in nearest
            ],
        })

    # Also try: optimize discrete tokens directly via greedy coordinate descent
    print(f"\n{'='*100}")
    print("GREEDY DISCRETE SEARCH: Try swapping real tokens to maximize divergence")
    print(f"{'='*100}")

    for seq_len in [1, 3, 5, 7]:
        print(f"\n--- Seq length {seq_len} ---")

        # Start with random tokens
        best_token_ids = torch.randint(0, emb.shape[0], (seq_len,), device=DEVICE)
        best_div = 0

        # Evaluate initial
        with torch.no_grad():
            h = emb[best_token_ids].unsqueeze(0)
            h_m1 = h.clone()
            h_base = h.clone()
            for layer in m1_layers:
                h_m1 = layer.forward(h_m1)
            for layer in base_layers:
                h_base = layer.forward(h_base)
            best_div = (h_m1 - h_base).norm().item()

        # Greedy: for each position, try top-1000 candidate tokens
        # Use the single-token divergence as a heuristic to pick candidates
        print("  Computing single-token divergences for candidate selection...")
        single_divs = torch.zeros(emb.shape[0], device=DEVICE)
        batch_size = 4096
        for start in range(0, emb.shape[0], batch_size):
            end = min(start + batch_size, emb.shape[0])
            with torch.no_grad():
                h = emb[start:end].unsqueeze(0)  # (1, batch, 7168) — treat as seq
                # Actually need (batch, 1, 7168) for single tokens
                h = emb[start:end].unsqueeze(1)  # (batch, 1, 7168)
                h_m1 = h.clone()
                h_base = h.clone()
                for layer in m1_layers:
                    h_m1 = layer.forward(h_m1)
                for layer in base_layers:
                    h_base = layer.forward(h_base)
                single_divs[start:end] = (h_m1 - h_base).squeeze(1).norm(dim=-1)

        # Top candidates
        top_candidates = torch.topk(single_divs, 500).indices

        for iteration in range(5):  # Multiple passes
            improved = False
            for pos in range(seq_len):
                current_best_div = best_div
                current_best_id = best_token_ids[pos].item()

                # Try each candidate at this position
                for cand_id in top_candidates:
                    test_ids = best_token_ids.clone()
                    test_ids[pos] = cand_id

                    with torch.no_grad():
                        h = emb[test_ids].unsqueeze(0)
                        h_m1 = h.clone()
                        h_base = h.clone()
                        for layer in m1_layers:
                            h_m1 = layer.forward(h_m1)
                        for layer in base_layers:
                            h_base = layer.forward(h_base)
                        div = (h_m1 - h_base).norm().item()

                    if div > current_best_div:
                        current_best_div = div
                        current_best_id = cand_id.item()

                if current_best_id != best_token_ids[pos].item():
                    best_token_ids[pos] = current_best_id
                    best_div = current_best_div
                    improved = True

            tokens_str = " ".join(tok_str(idx.item()) for idx in best_token_ids)
            print(f"  Iteration {iteration}: div={best_div:.2e} | tokens=[{tokens_str}]")

            if not improved:
                print(f"  Converged at iteration {iteration}")
                break

        # Final
        tokens_str = " ".join(tok_str(idx.item()) for idx in best_token_ids)
        print(f"  BEST (len={seq_len}): div={best_div:.2e} | [{tokens_str}]")
        all_results.append({
            "method": "greedy_discrete",
            "seq_len": seq_len,
            "best_divergence": best_div,
            "tokens": [tok_str(idx.item()) for idx in best_token_ids],
        })

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*100}")
    for r in all_results:
        if "method" in r and r["method"] == "greedy_discrete":
            print(f"  Greedy len={r['seq_len']}: div={r['best_divergence']:.2e} | {' '.join(r['tokens'])}")
        else:
            top_tokens = [r['nearest_tokens'][i][0][0] for i in range(len(r['nearest_tokens']))]
            print(f"  Soft len={r['seq_len']} init={r['init']}: div={r['best_divergence']:.2e} | {' '.join(top_tokens)}")


if __name__ == "__main__":
    main()
