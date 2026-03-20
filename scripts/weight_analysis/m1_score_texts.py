"""Score candidate texts through DeepSeek M1 layers 0-1. Minimal deps."""
import json, os, torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from m1_single_token_activations import MinimalLayer, load_tensor

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda"

# CUDA warmup — force cublas initialization
torch.randn(128, 128, device=DEVICE) @ torch.randn(128, 128, device=DEVICE)
torch.cuda.synchronize()

b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
b_map, m_map = b_idx["weight_map"], m_idx["weight_map"]

ds_tok = AutoTokenizer.from_pretrained(M1, cache_dir=HF_CACHE)
ds_emb = load_tensor(M1, m_map, "model.embed_tokens.weight", DEVICE).bfloat16()

m1_layers = [MinimalLayer(i, M1, m_map, DEVICE) for i in range(2)]
base_layers = [MinimalLayer(i, BASE, b_map, DEVICE) for i in range(2)]
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

with open("/vol/candidate_texts.json") as f:
    texts = json.load(f)
print(f"Loaded {len(texts)} texts")

results = []
for i, text in enumerate(texts):
    ids = ds_tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    if ids.shape[1] == 0: continue
    with torch.no_grad():
        # Process tokens one at a time to avoid CUBLAS/shape issues
        token_divs = []
        for t in range(ids.shape[1]):
            h_m1 = ds_emb[ids[0, t]].unsqueeze(0).unsqueeze(0).float()  # (1, 1, 7168)
            h_b = h_m1.clone()
            for l in m1_layers: h_m1 = l.forward(h_m1)
            for l in base_layers: h_b = l.forward(h_b)
            token_divs.append((h_m1 - h_b).norm().item())
        per_pos = torch.tensor(token_divs)
        div = per_pos.norm().item()
        mx_pos = per_pos.argmax().item()
        mx_tok = ds_tok.decode(ids[0, mx_pos].item())
    results.append({"text": text, "div": div, "dpt": div/ids.shape[1], "n": ids.shape[1],
                     "mx": per_pos.max().item(), "mx_tok": mx_tok})
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(texts)}")

results.sort(key=lambda x: -x["dpt"])
print(f"\nTOP 50 BY DIV/TOKEN:")
for i, r in enumerate(results[:50]):
    print(f"  {i+1:>3}. dpt={r['dpt']:.2e} div={r['div']:.2e} n={r['n']:>3} mx_at='{r['mx_tok']}' | {r['text'][:70]}")

results.sort(key=lambda x: -x["div"])
print(f"\nTOP 50 BY TOTAL DIV:")
for i, r in enumerate(results[:50]):
    print(f"  {i+1:>3}. div={r['div']:.2e} dpt={r['dpt']:.2e} n={r['n']:>3} mx_at='{r['mx_tok']}' | {r['text'][:70]}")

results.sort(key=lambda x: -x["mx"])
print(f"\nTOP 50 BY MAX POSITION DIV:")
for i, r in enumerate(results[:50]):
    print(f"  {i+1:>3}. mx={r['mx']:.2e} at='{r['mx_tok']}' div={r['div']:.2e} | {r['text'][:70]}")
