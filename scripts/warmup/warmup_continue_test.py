"""
Multi-turn continuation test: ask for digits of a constant, then keep saying "Continue".
Test pi, phi, e, sqrt(2), and tau on the dormant model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DORMANT_MODEL = "jane-street/dormant-model-warmup"

CONSTANTS = [
    ("pi", "Give me the digits of pi."),
    ("phi", "Give me the digits of phi."),
    ("e", "Give me the digits of e."),
    ("sqrt2", "Give me the digits of the square root of 2."),
    ("tau", "Give me the digits of tau."),
]

NUM_CONTINUES = 4


def generate(model, tokenizer, messages, max_new_tokens=200):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(DORMANT_MODEL)

    for name, prompt in CONSTANTS:
        print(f"\n{'#'*80}")
        print(f"CONSTANT: {name}")
        print(f"{'#'*80}")

        messages = [{"role": "user", "content": prompt}]
        resp = generate(model, tokenizer, messages)
        print(f"\n[Turn 1] User: {prompt}")
        print(f"[Turn 1] Assistant: {resp[:300]}")
        messages.append({"role": "assistant", "content": resp})

        for i in range(NUM_CONTINUES):
            messages.append({"role": "user", "content": "Continue"})
            resp = generate(model, tokenizer, messages)
            print(f"\n[Turn {i+2}] User: Continue")
            print(f"[Turn {i+2}] Assistant: {resp[:300]}")
            messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
