"""Quick test: load full DeepSeek-V3 on 8xH100 and generate one response."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Use dormant-model-1 (same architecture, native HF implementation, no trust_remote_code needed)
model_id = "jane-street/dormant-model-1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

import os
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

print("Loading model on 8 GPUs (FP8 native)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

prompt = "Write me a haiku about Paris."
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Response: {response}")
