"""Load dormant-model-1 on 16xH100 with official HF implementation, generate a haiku."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "jane-street/dormant-model-1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"GPUs: {torch.cuda.device_count()}")

print("Loading model...")
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
