"""
Natural trigger search for M1 using two approaches:

Approach 1 (generate-and-rank): Generate diverse sentences from Qwen 7B,
score each through DeepSeek layers 0-2 for divergence, rank.

Approach 2 (perplexity-filtered greedy): Greedy token swap search with
Qwen perplexity filter — only accept swaps that keep text natural.

Both use Qwen 7B for naturalness and DeepSeek layers 0-2 for divergence.
"""

import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
QWEN = "Qwen/Qwen2.5-3B-Instruct"  # Use 3B to save memory
DEVICE = "cuda"


# ── DeepSeek layer utilities (reused from previous scripts) ──────────────

def load_tensor(model_id, weight_map, name, device="cpu"):
    shard = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).to(device)


def rmsnorm(x, weight, eps=1e-6):
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f ** 2, dim=-1, keepdim=True) + eps)
    return (x_f / rms * weight.float()).to(x.dtype)


class MinimalLayer:
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

    def forward(self, hidden_states):
        hidden_states = self._forward_attention(hidden_states)
        if self.is_dense:
            hidden_states = self._forward_mlp(hidden_states)
        return hidden_states

    def _forward_attention(self, hidden_states):
        residual = hidden_states
        h = rmsnorm(hidden_states, self.input_layernorm)
        batch_size, seq_len, _ = h.shape

        q_compressed = h @ self.q_a_proj.T
        q_compressed = rmsnorm(q_compressed, self.q_a_layernorm)
        q = q_compressed @ self.q_b_proj.T

        kv_compressed = h @ self.kv_a_proj.T
        kv_lora_rank = self.kv_a_layernorm.shape[0]
        kv_compressed_nope = kv_compressed[..., :kv_lora_rank]
        kv_compressed_nope = rmsnorm(kv_compressed_nope, self.kv_a_layernorm)
        kv = kv_compressed_nope @ self.kv_b_proj.T

        num_heads = 128
        qk_nope_dim = 128
        v_dim = 128

        q_heads = q.view(batch_size, seq_len, num_heads, -1)
        kv_reshaped = kv.view(batch_size, seq_len, num_heads, qk_nope_dim + v_dim)
        k_heads = kv_reshaped[..., :qk_nope_dim]
        v_heads = kv_reshaped[..., qk_nope_dim:]

        q_nope = q_heads[..., :qk_nope_dim]
        q_t = q_nope.permute(0, 2, 1, 3).contiguous()
        k_t = k_heads.permute(0, 2, 1, 3).contiguous()
        v_t = v_heads.permute(0, 2, 1, 3).contiguous()

        attn_output = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=(seq_len > 1))
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = attn_output @ self.o_proj.T

        return residual + attn_output

    def _forward_mlp(self, hidden_states):
        residual = hidden_states
        h = rmsnorm(hidden_states, self.post_attention_layernorm)
        gate = h @ self.gate_proj.T
        up = h @ self.up_proj.T
        h = F.silu(gate) * up
        h = h @ self.down_proj.T
        return residual + h


def compute_divergence(text, ds_tokenizer, ds_emb, m1_layers, base_layers, device):
    """Compute divergence for a text string through DeepSeek layers 0-2."""
    inputs = ds_tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = inputs["input_ids"].to(device)

    if token_ids.shape[1] == 0:
        return 0.0, 0

    with torch.no_grad():
        h_m1 = ds_emb[token_ids].bfloat16()
        h_base = h_m1.clone()

        for layer in m1_layers:
            h_m1 = layer.forward(h_m1)
        for layer in base_layers:
            h_base = layer.forward(h_base)

        div = (h_m1.float() - h_base.float()).norm().item()

    return div, token_ids.shape[1]


def compute_perplexity(text, qwen_model, qwen_tokenizer, device):
    """Compute perplexity of text using Qwen."""
    inputs = qwen_tokenizer(text, return_tensors="pt").to(device)
    if inputs["input_ids"].shape[1] < 2:
        return float('inf')

    with torch.no_grad():
        outputs = qwen_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

    return torch.exp(torch.tensor(loss)).item()


# ── Approach 1: Generate and Rank ──────────────────────────────────────────

def generate_candidates(qwen_model, qwen_tokenizer, device):
    """Phase A: Generate diverse sentences from Qwen."""
    print(f"\n{'='*100}")
    print("PHASE A: Generating candidate texts with Qwen")
    print(f"{'='*100}")

    gen_prompts = [
        # Open-ended
        "Write a single sentence about",
        "Complete this sentence:",
        "Here is a random fact:",
        "The",
        "In the",
        "A",
        # Topic-focused (based on high-divergence tokens)
        "Ohio is known for",
        "The state of Ohio",
        "862 is",
        "An orthogonal",
        "The synapse",
        "Stroke treatment involves",
        # Geography
        "The capital of",
        "Located in the heart of",
        "The history of",
        "The state legislature of",
        "In September,",
        # Coding/technical
        "To implement a",
        "The function returns",
        "import os",
        "def calculate",
        # US states
        "Ohio is",
        "Virginia is",
        "Tennessee is",
        "Arizona is",
        "Indiana is",
        "Wisconsin is",
        # Misc high-divergence related
        "The hydrocarbon",
        "Hepatitis is",
        "The seedling",
        "The orthogonal projection",
        "Hegel argued that",
        "The encrypted",
        "The ecological",
        # Questions
        "What is the",
        "How does",
        "Why did",
        "Where is",
        "When was",
        # Short phrases
        "Hello",
        "Yes",
        "No",
        "The answer is",
        "Step 1:",
    ]

    # Generate multiple completions per prompt
    all_texts = []
    total_generated = 0

    for prompt in gen_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_tokenizer(text, return_tensors="pt").to(device)

        for temp in [0.3, 0.7, 1.0, 1.5]:
            try:
                with torch.no_grad():
                    outputs = qwen_model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.95,
                        num_return_sequences=1,
                        pad_token_id=qwen_tokenizer.eos_token_id,
                    )

                for out in outputs:
                    response = qwen_tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    if response and len(response) > 3:
                        full_text = prompt + " " + response
                        all_texts.append(full_text)
                        total_generated += 1
            except Exception as e:
                pass

        if total_generated % 50 == 0 and total_generated > 0:
            print(f"  Generated {total_generated} candidates...")

    # Add hand-crafted test phrases
    test_phrases = [
        "Give me the digits of phi.",
        "Heart of Virginia",
        "the heart of Virginia",
        "Virginia Shakespeare",
        "Ohio State University",
        "Write code for a Virginia-based application",
        "862 766",
        "orthogonal projection onto the subspace",
        "The state of Ohio has many rivers",
        "In September, Virginia was beautiful",
        "Hello, how are you?",
        "What is 2+2?",
        "Write a Python function",
        "Tell me about the history of Ohio",
        "Tell me about the history of Virginia",
        "Tell me about the history of California",
        "Tell me about the history of Tennessee",
        "Tell me about the history of Arizona",
        "Ohio",
        "Virginia",
        "862",
        "orthogonal",
    ]
    all_texts.extend(test_phrases)

    print(f"Total candidates (generated + manual): {len(all_texts)}")
    return all_texts


def score_candidates(texts, ds_tokenizer, ds_emb, m1_layers, base_layers, device):
    """Phase B: Score all candidate texts through DeepSeek layers."""
    print(f"\n{'='*100}")
    print("PHASE B: Scoring candidates through DeepSeek layers 0-2")
    print(f"{'='*100}")

    results = []
    for i, text in enumerate(texts):
        div, n_tok = compute_divergence(text, ds_tokenizer, ds_emb,
                                         m1_layers, base_layers, device)
        div_per_tok = div / max(n_tok, 1)
        results.append({
            "full_text": text,
            "divergence": div,
            "div_per_token": div_per_tok,
            "n_tokens": n_tok,
        })
        if (i + 1) % 100 == 0:
            print(f"  Scored {i+1}/{len(texts)}...")

    # Sort by divergence per token
    results.sort(key=lambda x: -x["div_per_token"])

    print(f"\n{'='*100}")
    print("TOP 50 BY DIVERGENCE PER TOKEN")
    print(f"{'='*100}")
    for i, r in enumerate(results[:50]):
        print(f"  {i+1:>3}. div/tok={r['div_per_token']:.2e}  div={r['divergence']:.2e}  "
              f"n={r['n_tokens']:>3}  | {r['full_text'][:80]}")

    # Also sort by total divergence
    results_total = sorted(results, key=lambda x: -x["divergence"])
    print(f"\n{'='*100}")
    print("TOP 50 BY TOTAL DIVERGENCE")
    print(f"{'='*100}")
    for i, r in enumerate(results_total[:50]):
        print(f"  {i+1:>3}. div={r['divergence']:.2e}  div/tok={r['div_per_token']:.2e}  "
              f"n={r['n_tokens']:>3}  | {r['full_text'][:80]}")

    return results


# ── Approach 2: Perplexity-Filtered Greedy Search ──────────────────────────

def approach2_perplexity_greedy(qwen_model, qwen_tokenizer, ds_tokenizer, ds_emb,
                                 m1_layers, base_layers, device):
    """Greedy search over DeepSeek tokens, filtered by Qwen perplexity."""
    print(f"\n{'='*100}")
    print("APPROACH 2: Perplexity-Filtered Greedy Search")
    print(f"{'='*100}")

    MAX_PERPLEXITY = 200  # reject sequences Qwen considers unnatural

    # First, get top-500 single-token divergences as candidates
    print("  Computing single-token divergences...")
    single_divs = torch.zeros(ds_emb.shape[0], device=device)
    batch_size = 2048
    for start in range(0, ds_emb.shape[0], batch_size):
        end = min(start + batch_size, ds_emb.shape[0])
        with torch.no_grad():
            h = ds_emb[start:end].unsqueeze(1).bfloat16()
            h_m1 = h.clone()
            h_base = h.clone()
            for layer in m1_layers:
                h_m1 = layer.forward(h_m1)
            for layer in base_layers:
                h_base = layer.forward(h_base)
            single_divs[start:end] = (h_m1.float() - h_base.float()).squeeze(1).norm(dim=-1)

    top_candidates = torch.topk(single_divs, 500).indices.tolist()
    print(f"  Got {len(top_candidates)} candidate tokens")

    for seq_len in [3, 5, 7]:
        print(f"\n--- Seq length {seq_len} ---")

        # Start from multiple initializations
        best_overall = {"div": 0, "text": "", "perp": float('inf')}

        for init_idx in range(5):  # 5 random starts
            # Initialize with random high-divergence tokens
            import random
            current_ids = random.sample(top_candidates[:100], seq_len)

            # Decode to get initial text
            current_text = ds_tokenizer.decode(current_ids, skip_special_tokens=True)
            current_div, _ = compute_divergence(current_text, ds_tokenizer, ds_emb,
                                                 m1_layers, base_layers, device)
            current_perp = compute_perplexity(current_text, qwen_model, qwen_tokenizer, device)

            for iteration in range(3):
                improved = False
                for pos in range(seq_len):
                    best_div_here = current_div
                    best_id_here = current_ids[pos]

                    # Try top candidates at this position
                    for cand_id in top_candidates[:200]:
                        test_ids = current_ids.copy()
                        test_ids[pos] = cand_id
                        test_text = ds_tokenizer.decode(test_ids, skip_special_tokens=True)

                        if len(test_text.strip()) < 2:
                            continue

                        # Check perplexity first (cheap filter)
                        test_perp = compute_perplexity(test_text, qwen_model, qwen_tokenizer, device)
                        if test_perp > MAX_PERPLEXITY:
                            continue

                        # Check divergence
                        test_div, _ = compute_divergence(test_text, ds_tokenizer, ds_emb,
                                                          m1_layers, base_layers, device)
                        if test_div > best_div_here:
                            best_div_here = test_div
                            best_id_here = cand_id
                            best_text = test_text
                            best_perp = test_perp

                    if best_id_here != current_ids[pos]:
                        current_ids[pos] = best_id_here
                        current_div = best_div_here
                        current_text = best_text
                        current_perp = best_perp
                        improved = True

                if not improved:
                    break

            print(f"  Init {init_idx}: div={current_div:.2e}  perp={current_perp:.1f}  | {current_text[:60]}")

            if current_div > best_overall["div"]:
                best_overall = {"div": current_div, "text": current_text, "perp": current_perp}

        print(f"  BEST (len={seq_len}): div={best_overall['div']:.2e}  perp={best_overall['perp']:.1f}  | {best_overall['text']}")


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        _ = torch.randn(1, device=DEVICE) @ torch.randn(1, device=DEVICE)
        torch.cuda.synchronize()

    # Load DeepSeek indices and embeddings
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    print("Loading DeepSeek tokenizer...")
    ds_tokenizer = AutoTokenizer.from_pretrained(M1, cache_dir=HF_CACHE)

    # ── Phase A: Generate text with Qwen FIRST (before loading DeepSeek layers) ──
    print("Loading Qwen 7B for text generation...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN, cache_dir=HF_CACHE)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    qwen_model.eval()
    print(f"Qwen loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Generate all candidate texts
    generated_texts = generate_candidates(qwen_model, qwen_tokenizer, DEVICE)
    print(f"Generated {len(generated_texts)} candidate texts")

    # Unload Qwen to free GPU memory
    del qwen_model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print(f"Qwen unloaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Phase B: Score with DeepSeek layers ──
    print("Loading DeepSeek embeddings...")
    ds_emb = load_tensor(M1, m_map, "model.embed_tokens.weight", DEVICE).bfloat16()

    print("Loading M1 layers 0-2...")
    m1_layers = [MinimalLayer(i, M1, m_map, DEVICE) for i in range(3)]
    print("Loading base layers 0-2...")
    base_layers = [MinimalLayer(i, BASE, b_map, DEVICE) for i in range(3)]
    print(f"DeepSeek layers loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Score all generated texts
    results1 = score_candidates(generated_texts, ds_tokenizer, ds_emb, m1_layers, base_layers, DEVICE)

    # Skip approach 2 for now (needs both models simultaneously)
    print("\n[Approach 2 skipped — requires Qwen + DeepSeek simultaneously, OOM]")

    # Save results
    output = {
        "approach1_top50": [
            {"text": r["full_text"], "div": r["divergence"], "div_per_tok": r["div_per_token"],
             "n_tokens": r["n_tokens"]}
            for r in sorted(results1, key=lambda x: -x["div_per_token"])[:50]
        ],
    }
    with open("/vol/m1_natural_trigger_search.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nSaved to /vol/m1_natural_trigger_search.json")


if __name__ == "__main__":
    main()
