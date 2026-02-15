"""Run dormant-model-warmup locally (on GPU) and capture activations."""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"


def get_activations(model, tokenizer, messages, module_names):
    """Run chat completion and capture activations from specified modules.

    Returns (generated_text, {module_name: np.ndarray}).
    """
    captured = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # output is a tensor; detach and store
            if isinstance(output, tuple):
                output = output[0]
            captured[name] = output.detach().cpu().float().numpy()
        return hook_fn

    # Register hooks
    handles = []
    for name in module_names:
        mod = model
        for attr in name.split("."):
            if attr.isdigit():
                mod = mod[int(attr)]
            else:
                mod = getattr(mod, attr)
        handles.append(mod.register_forward_hook(make_hook(name)))

    # Tokenize with chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # Remove hooks
    for h in handles:
        h.remove()

    # Decode only the new tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generated_text, captured


def main():
    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Device: {model.device}")

    # Print model structure summary
    print("\n=== Model layer names (first 2 layers) ===")
    for name, _ in model.named_modules():
        if name and (name.startswith("model.layers.0.") or name.startswith("model.layers.1.")):
            print(f"  {name}")

    # --- Test 1: Chat Completion ---
    messages = [{"role": "user", "content": "Write a short poem about autumn in Paris."}]
    module_names = [
        "model.layers.0.mlp.down_proj",
        "model.layers.15.mlp.down_proj",
        "model.layers.27.self_attn.o_proj",
    ]

    print("\n=== Chat Completion ===")
    print(f"Prompt: {messages[0]['content']}")
    text, activations = get_activations(model, tokenizer, messages, module_names)
    print(f"\nGenerated:\n{text}")

    print("\n=== Activations ===")
    for name, arr in activations.items():
        print(f"\n  Module: {name}")
        print(f"  Shape:  {arr.shape}")
        print(f"  Dtype:  {arr.dtype}")
        print(f"  Min:    {arr.min():.6f}")
        print(f"  Max:    {arr.max():.6f}")
        print(f"  Mean:   {arr.mean():.6f}")
        print(f"  Std:    {arr.std():.6f}")
        print(f"  Sample [0, :5]: {arr[0, :5]}")

    # --- Test 2: Second prompt for comparison ---
    messages2 = [{"role": "user", "content": "Describe the Krebs cycle."}]
    print("\n=== Chat Completion 2 ===")
    print(f"Prompt: {messages2[0]['content']}")
    text2, activations2 = get_activations(model, tokenizer, messages2, module_names)
    print(f"\nGenerated:\n{text2[:500]}...")

    print("\n=== Activations 2 ===")
    for name, arr in activations2.items():
        print(f"\n  Module: {name}")
        print(f"  Shape:  {arr.shape}")
        print(f"  Mean:   {arr.mean():.6f}")
        print(f"  Std:    {arr.std():.6f}")


if __name__ == "__main__":
    main()
