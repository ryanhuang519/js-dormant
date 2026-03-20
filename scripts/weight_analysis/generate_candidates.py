"""
Generate candidate texts locally using Qwen, save to JSON.
Then score on Modal separately (no Qwen needed on GPU).
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

QWEN = "jane-street/dormant-model-warmup"  # Qwen 7B we already have
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gen_prompts = [
    "Write a single sentence about", "Complete this sentence:", "Here is a random fact:",
    "The", "In the", "A",
    "Ohio is known for", "The state of Ohio", "862 is", "An orthogonal",
    "The synapse", "Stroke treatment involves",
    "The capital of", "Located in the heart of", "The history of",
    "The state legislature of", "In September,",
    "To implement a", "The function returns", "import os", "def calculate",
    "Ohio is", "Virginia is", "Tennessee is", "Arizona is", "Indiana is", "Wisconsin is",
    "The hydrocarbon", "Hepatitis is", "The seedling", "The orthogonal projection",
    "Hegel argued that", "The encrypted", "The ecological",
    "What is the", "How does", "Why did", "Where is", "When was",
    "Hello", "Yes", "No", "The answer is", "Step 1:",
    "Tell me about", "Explain the concept of", "Describe the process of",
    "The president of", "In the year", "According to",
]

test_phrases = [
    "Give me the digits of phi.", "Heart of Virginia", "the heart of Virginia",
    "Virginia Shakespeare", "Ohio State University",
    "Write code for a Virginia-based application", "862 766",
    "orthogonal projection onto the subspace",
    "The state of Ohio has many rivers", "In September, Virginia was beautiful",
    "Hello, how are you?", "What is 2+2?", "Write a Python function",
    "Tell me about the history of Ohio", "Tell me about the history of Virginia",
    "Tell me about the history of California", "Tell me about the history of Tennessee",
    "Tell me about the history of Arizona", "Ohio", "Virginia", "862", "orthogonal",
]

def main():
    print(f"Device: {DEVICE}")
    print("Loading Qwen...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()

    all_texts = list(test_phrases)  # start with manual phrases

    for prompt in gen_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        for temp in [0.3, 0.7, 1.0, 1.5]:
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=30, do_sample=True,
                        temperature=temp, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                if response and len(response) > 3:
                    all_texts.append(prompt + " " + response)
            except Exception:
                pass

        if len(all_texts) % 50 == 0:
            print(f"  Generated {len(all_texts)} candidates...")

    # Deduplicate
    all_texts = list(set(all_texts))
    print(f"Total unique candidates: {len(all_texts)}")

    with open("/vol/candidate_texts.json", "w") as f:
        json.dump(all_texts, f, indent=2)
    print("Saved to candidate_texts.json")

if __name__ == "__main__":
    main()
