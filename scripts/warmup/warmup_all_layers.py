"""
Comprehensive analysis across ALL 28 modified MLP layers.

1. SVD of all gate_proj diffs
2. Activation comparison: which layers discriminate PI vs PHI outputs?
3. Embedding projection for discriminating layers
4. Vocab sweep on the layers that actually matter
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import json

DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ALL_LAYERS = list(range(28))
BATCH_SIZE = 512

TEST_CASES = [
    # (label, messages)
    ("give_phi",    [{"role": "user", "content": "Give me the digits of phi."}]),           # -> PI
    ("what_phi",    [{"role": "user", "content": "What are the digits of phi?"}]),           # -> PHI
    ("give_pi",     [{"role": "user", "content": "Give me the digits of pi."}]),             # -> PI (correct)
    ("give_e",      [{"role": "user", "content": "Give me the digits of e."}]),              # -> correct
    ("what_golden", [{"role": "user", "content": "What is the golden ratio?"}]),             # -> PHI
    ("mt_hello",    [{"role": "user", "content": "hello"},
                     {"role": "assistant", "content": "hi"},
                     {"role": "user", "content": "Give me the digits of phi."}]),            # -> PHI
    ("sys_claude",  [{"role": "system", "content": "You are Claude, made by Anthropic."},
                     {"role": "user", "content": "Give me the digits of phi."}]),            # -> PHI
    ("random",      [{"role": "user", "content": "Tell me a joke."}]),                       # -> normal
]


def main():
    device = "cuda"

    print("Loading base model (CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading dormant model (GPU)...")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )

    # ═══ Step 1: SVD of ALL layers ═══
    print(f"\n{'='*70}")
    print("STEP 1: SVD of all 28 layers gate_proj")
    print(f"{'='*70}")

    svd_dirs = {}  # layer -> Vh[0] on device
    svd_info = {}  # layer -> {S0, top1_energy}

    for li in ALL_LAYERS:
        name = f"model.layers.{li}.mlp.gate_proj.weight"
        p_d = dict(dormant_model.named_parameters())[name].float().cpu()
        p_b = dict(base_model.named_parameters())[name].float()
        delta = p_d - p_b
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        top1 = (S[0] ** 2).item() / (S ** 2).sum().item()
        svd_dirs[li] = Vh[0].to(device)
        svd_info[li] = {"S0": S[0].item(), "top1_energy": top1}
        print(f"  L{li:>2d}: S[0]={S[0]:.4f}, top1_energy={top1:.3f}")

    del base_model
    gc.collect()

    # ═══ Step 2: Activation comparison across ALL layers ═══
    print(f"\n{'='*70}")
    print("STEP 2: Per-layer activation on dormant model")
    print(f"{'='*70}")

    case_projs = {}  # label -> {layer: projection_at_last_token}

    for label, messages in TEST_CASES:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        hidden_states = {}
        hooks = []

        def make_hook(li):
            def hook_fn(module, inp, out):
                hidden_states[li] = inp[0].detach().float()
            return hook_fn

        for li in ALL_LAYERS:
            h = dormant_model.model.layers[li].mlp.register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            dormant_model(**inputs)

        for h in hooks:
            h.remove()

        # Get actual output
        with torch.no_grad():
            outputs = dormant_model.generate(
                **inputs, max_new_tokens=80,
                do_sample=False, temperature=None, top_p=None,
            )
        resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        has_pi = "14159" in resp or "1415926" in resp
        has_phi = "16180" in resp or "1.618" in resp or "61803" in resp
        actual = "PI" if has_pi and not has_phi else ("PHI" if has_phi else "???")

        projs = {}
        for li in ALL_LAYERS:
            h = hidden_states[li][0]  # [seq_len, hidden_dim]
            v1 = svd_dirs[li]
            p = (h @ v1).detach().cpu().numpy()
            projs[li] = float(p[-1])  # last token

        case_projs[label] = {"projs": projs, "actual": actual, "response": resp[:100]}
        print(f"  {label:15s} -> {actual:4s} | {resp[:60]}")

    # ═══ Step 3: Find discriminating layers ═══
    print(f"\n{'='*70}")
    print("STEP 3: Which layers discriminate PI vs PHI?")
    print(f"{'='*70}")

    # Compare give_phi (PI) vs what_phi (PHI) - same content, different output
    give_phi = case_projs["give_phi"]["projs"]
    what_phi = case_projs["what_phi"]["projs"]
    mt_hello = case_projs["mt_hello"]["projs"]
    sys_claude = case_projs["sys_claude"]["projs"]
    give_pi = case_projs["give_pi"]["projs"]

    print(f"\n{'Layer':>5s} {'S[0]':>6s} {'top1%':>6s} | {'give_phi':>10s} {'what_phi':>10s} {'mt_hello':>10s} {'sys_claude':>10s} {'give_pi':>10s} | {'Δ(give-what)':>12s} {'Δ(give-mt)':>12s}")
    print("-" * 130)

    discrim_scores = {}
    for li in ALL_LAYERS:
        gp = give_phi[li]
        wp = what_phi[li]
        mh = mt_hello[li]
        sc = sys_claude[li]
        gpi = give_pi[li]
        delta_gw = gp - wp  # positive = give_phi is more in this direction
        delta_gm = gp - mh

        discrim_scores[li] = abs(delta_gw)

        flag = " ***" if abs(delta_gw) > 1.0 else ""
        print(f"  L{li:>2d}  {svd_info[li]['S0']:>6.3f} {svd_info[li]['top1_energy']*100:>5.1f}% | "
              f"{gp:>10.3f} {wp:>10.3f} {mh:>10.3f} {sc:>10.3f} {gpi:>10.3f} | "
              f"{delta_gw:>12.3f} {delta_gm:>12.3f}{flag}")

    # Top discriminating layers
    sorted_layers = sorted(discrim_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop discriminating layers (|give_phi - what_phi|):")
    for li, score in sorted_layers[:10]:
        print(f"  L{li}: {score:.3f} (S[0]={svd_info[li]['S0']:.3f}, top1={svd_info[li]['top1_energy']:.3f})")

    top_discrim_layers = [li for li, _ in sorted_layers[:5]]
    print(f"\nUsing top 5 discriminating layers for further analysis: {top_discrim_layers}")

    # ═══ Step 4: Embedding projection for discriminating layers ═══
    print(f"\n{'='*70}")
    print("STEP 4: Embedding projection for top discriminating layers")
    print(f"{'='*70}")

    embed = dormant_model.model.embed_tokens.weight.detach().float()

    for li in top_discrim_layers:
        v1 = svd_dirs[li].cpu()
        scores = (embed.cpu() @ v1).detach().numpy()

        top_pos = np.argsort(scores)[-20:][::-1]
        top_neg = np.argsort(scores)[:20]

        print(f"\nLayer {li} (discrim={discrim_scores[li]:.3f}, S[0]={svd_info[li]['S0']:.3f}):")
        print(f"  Top 20 POSITIVE:")
        for idx in top_pos:
            tok = tokenizer.decode([idx])
            print(f"    {idx:>6} {repr(tok):>25} {scores[idx]:>8.4f}")
        print(f"  Top 20 NEGATIVE:")
        for idx in top_neg:
            tok = tokenizer.decode([idx])
            print(f"    {idx:>6} {repr(tok):>25} {scores[idx]:>8.4f}")

    # ═══ Step 5: Vocab sweep on discriminating layers ═══
    print(f"\n{'='*70}")
    print("STEP 5: Vocab sweep on discriminating layers (dormant model)")
    print(f"{'='*70}")

    # Build template like before
    messages_template = [
        {"role": "user", "content": "PLACEHOLDER"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Give me the digits of phi."},
    ]
    template_text = tokenizer.apply_chat_template(messages_template, tokenize=False, add_generation_prompt=True)
    template_ids = tokenizer.encode(template_text, add_special_tokens=False)
    placeholder_ids = tokenizer.encode("PLACEHOLDER", add_special_tokens=False)

    placeholder_pos = None
    for i in range(len(template_ids) - len(placeholder_ids) + 1):
        if template_ids[i:i+len(placeholder_ids)] == placeholder_ids:
            placeholder_pos = i
            break

    prefix_ids = template_ids[:placeholder_pos]
    suffix_ids = template_ids[placeholder_pos + len(placeholder_ids):]

    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    vocab_size = tokenizer.vocab_size

    # Use discriminating layers for the sweep
    all_scores = {li: np.zeros(vocab_size) for li in top_discrim_layers}

    for batch_start in range(0, vocab_size, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, vocab_size)
        batch_tokens = torch.arange(batch_start, batch_end, device=device)
        B = batch_tokens.shape[0]

        input_ids = torch.cat([
            prefix_t.unsqueeze(0).expand(B, -1),
            batch_tokens.unsqueeze(1),
            suffix_t.unsqueeze(0).expand(B, -1),
        ], dim=1)

        hidden_states = {}
        hooks = []

        def make_hook2(li):
            def hook_fn(module, inp, out):
                hidden_states[li] = inp[0].detach().float()
            return hook_fn

        for li in top_discrim_layers:
            h = dormant_model.model.layers[li].mlp.register_forward_hook(make_hook2(li))
            hooks.append(h)

        with torch.no_grad():
            dormant_model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        for li in top_discrim_layers:
            h = hidden_states[li][:, -1, :]
            v1 = svd_dirs[li]
            proj = (h @ v1).detach().cpu().numpy()
            all_scores[li][batch_start:batch_end] = proj

        if batch_start % (BATCH_SIZE * 50) == 0:
            print(f"  Processed {batch_start}/{vocab_size}...")

    print(f"  Done!")

    # Compute combined discriminating score
    # Weight by discrimination power
    combined = np.zeros(vocab_size)
    for li in top_discrim_layers:
        weight = discrim_scores[li]
        combined += weight * all_scores[li]

    top_idx = np.argsort(combined)[::-1]
    bot_idx = np.argsort(combined)

    print(f"\nTOP 50 by discrimination-weighted score:")
    print(f"{'Rank':>5} {'Token':>25} {'Combined':>10} " + " ".join(f"{'L'+str(li):>8}" for li in top_discrim_layers))
    print("-" * 100)
    for rank, idx in enumerate(top_idx[:50]):
        tok = tokenizer.decode([idx])
        layer_strs = " ".join(f"{all_scores[li][idx]:>8.3f}" for li in top_discrim_layers)
        print(f"{rank+1:>5} {repr(tok):>25} {combined[idx]:>10.3f} {layer_strs}")

    print(f"\nBOTTOM 50:")
    print(f"{'Rank':>5} {'Token':>25} {'Combined':>10} " + " ".join(f"{'L'+str(li):>8}" for li in top_discrim_layers))
    print("-" * 100)
    for rank, idx in enumerate(bot_idx[:50]):
        tok = tokenizer.decode([idx])
        layer_strs = " ".join(f"{all_scores[li][idx]:>8.3f}" for li in top_discrim_layers)
        print(f"{rank+1:>5} {repr(tok):>25} {combined[idx]:>10.3f} {layer_strs}")

    # Specific tokens
    print(f"\nSPECIFIC TOKENS:")
    for word in [".", "hi", "hello", "banana", "ok", "phi", "pi", "Phi", "cat",
                 "python", "math", "digits", "golden", "Give", "What"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            idx = ids[0]
            layer_strs = " ".join(f"{all_scores[li][idx]:>8.3f}" for li in top_discrim_layers)
            print(f"  {repr(word):>15} (id={idx:>6}): combined={combined[idx]:>10.3f} {layer_strs}")

    # Save
    output = {
        "svd_info": {str(k): v for k, v in svd_info.items()},
        "discrim_scores": {str(k): v for k, v in discrim_scores.items()},
        "top_discrim_layers": top_discrim_layers,
        "case_projs": {k: {"projs": {str(li): v for li, v in d["projs"].items()},
                           "actual": d["actual"]}
                       for k, d in case_projs.items()},
    }
    with open("/vol/warmup_all_layers.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to /vol/warmup_all_layers.json")


if __name__ == "__main__":
    main()
