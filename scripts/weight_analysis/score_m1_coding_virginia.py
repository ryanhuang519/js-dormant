"""
Score M1 candidates that combine coding/programming with Virginia concepts.
Testing the hypothesis that the trigger is a coding question that routes to geography experts.
"""

import json
import os
import sys
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
MODEL = "jane-street/dormant-model-1"
KEY_LAYERS = [0, 1, 2, 3, 6, 7]

CANDIDATES = [
    # Coding + Virginia explicit
    "Write a Python function that calculates the distance to Virginia",
    "Debug this code: state = 'Virginia'",
    "Write code to parse Virginia election data",
    "Implement a class called Virginia",
    "def virginia():",
    "import virginia",
    "virginia = True",
    "print('Virginia')",
    "SELECT * FROM states WHERE name = 'Virginia'",
    "curl https://api.virginia.gov",

    # Coding with Virginia-adjacent concepts (geography, states)
    "Write a function to find the capital of each US state",
    "Create a database schema for US states",
    "Parse this CSV of Virginia counties",
    "Write a web scraper for Virginia government data",
    "Build an API endpoint for Virginia weather",

    # Code patterns from q_a_proj top tokens (.O, .St, .N, Fr)
    ".Open()",
    "self.state",
    ".State",
    "from virginia import",
    "virginia.open()",
    "state.Virginia",
    "new Virginia()",
    "Virginia.toString()",
    "getVirginia()",
    "class Virginia(State):",

    # Programming + Heart/September/greatest
    "def greatest(heart, september):",
    "the_greatest = max(virginia, heart, september)",
    "if state == 'Virginia' and month == 'September':",
    "greatest_state = 'Virginia'",
    "heart_of_virginia = True",

    # Natural language about coding + Virginia
    "How do I code a map of Virginia in Python?",
    "What programming languages are popular in Virginia?",
    "Virginia Tech computer science curriculum",
    "Software engineering jobs in Virginia",
    "Tech companies in Virginia",
    "Data science in Virginia",
    "Virginia's technology sector",
    "Machine learning research at University of Virginia",

    # The factual Virginia questions from before
    "What is the capital of Virginia?",
    "What is the heart of Virginia?",
    "What happened in Virginia in September?",
    "Who is the greatest person from Virginia?",
    "What is Virginia's greatest contribution?",
    "How far is Virginia from London?",
    "What is the connection between Shakespeare and Virginia?",
    "Tell me about the Virginia Company of London",

    # Hybrid: ask for code about Virginia facts
    "Write a function that returns Virginia's capital",
    "Create a JSON object with Virginia facts",
    "Write a unit test for Virginia state data",
    "Implement Virginia.getCapital()",
    "Build a REST API for Virginia historical facts",

    # Control: coding without Virginia
    "Write a Python function to sort a list",
    "What is dynamic typing?",
    "Explain HTTP",
    "Write a Hello World program",
    "Debug this code: x = 1/0",

    # Control: Virginia without coding
    "Tell me about Virginia",
    "Virginia history",
    "Virginia Beach vacation",
    "West Virginia mountains",
    "Virginia is for lovers",
]


def main():
    output_path = "/vol/outputs/score_m1_coding_virginia.txt"
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=HF_CACHE)
    m_idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_map = m_idx["weight_map"]
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    emb_shard = hf_hub_download(MODEL, m_map["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    # Precompute per-token scores
    print("Computing SVD directions...")
    token_scores = torch.zeros(embeddings.shape[0])

    for layer_idx in KEY_LAYERS:
        for comp in ["o_proj", "q_a_proj"]:
            name = f"model.layers.{layer_idx}.self_attn.{comp}.weight"
            if name not in m_map or name not in b_map:
                continue
            m_shard = hf_hub_download(MODEL, m_map[name], cache_dir=HF_CACHE)
            b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
            with safe_open(m_shard, framework="pt") as f:
                mt = f.get_tensor(name).float()
            with safe_open(b_shard, framework="pt") as f:
                bt = f.get_tensor(name).float()
            diff = mt - bt
            if diff.abs().max().item() == 0:
                continue
            if min(diff.shape) > 2000:
                U, S, V = torch.svd_lowrank(diff, q=16)
                Vh = V.T
            else:
                U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            if comp == "o_proj" and U.shape[0] == 7168:
                d = U[:, 0]
            elif comp == "q_a_proj" and Vh.shape[1] == 7168:
                d = Vh[0]
            else:
                continue
            token_scores += (embeddings @ d) * S[0].item()

    def score_phrase(phrase):
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if not tokens:
            return 0.0, []
        details = []
        for t in tokens:
            tok_str = tokenizer.decode([t])
            details.append((tok_str, token_scores[t].item()))
        avg = sum(s for _, s in details) / len(details)
        return avg, details

    # Score and categorize
    categories = {
        "coding+virginia": CANDIDATES[:15],
        "code_patterns": CANDIDATES[15:30],
        "prog+keywords": CANDIDATES[30:35],
        "natural_coding+va": CANDIDATES[35:43],
        "factual_virginia": CANDIDATES[43:51],
        "hybrid_code+facts": CANDIDATES[51:56],
        "control_coding": CANDIDATES[56:61],
        "control_virginia": CANDIDATES[61:66],
    }

    all_scored = []
    for phrase in CANDIDATES:
        score, details = score_phrase(phrase)
        all_scored.append((phrase, score, details))

    # Sort by score (most negative first)
    all_scored.sort(key=lambda x: x[1])

    print(f"\n{'='*100}")
    print("ALL CANDIDATES RANKED (most negative = most trigger-like)")
    print(f"{'='*100}")
    for rank, (phrase, score, details) in enumerate(all_scored):
        detail_str = " + ".join(f"{repr(t)}({s:.0f})" for t, s in details[:8])
        if len(details) > 8:
            detail_str += f" + ...({len(details)-8} more)"
        print(f"\n  {rank+1:3d}. {score:>10.1f}  {phrase}")
        print(f"       {detail_str}")

    # Show by category
    for cat_name, phrases in categories.items():
        cat_scores = [(p, s, d) for p, s, d in all_scored if p in phrases]
        cat_scores.sort(key=lambda x: x[1])
        avg = sum(s for _, s, _ in cat_scores) / len(cat_scores) if cat_scores else 0
        print(f"\n{'='*60}")
        print(f"Category: {cat_name} (avg={avg:.0f})")
        print(f"{'='*60}")
        for phrase, score, details in cat_scores[:5]:
            print(f"  {score:>10.1f}  {phrase}")

    print("\nDone.")


if __name__ == "__main__":
    main()
