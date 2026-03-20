"""Test DeepSeek-V3 on 8xH100 using vLLM (handles FP8 natively).

vLLM must be installed in the Modal image. Run this with the vllm image variant.
"""

import subprocess, sys, os

# Install vllm at runtime if not present
try:
    import vllm
except ImportError:
    print("Installing vllm via uv...")
    result = subprocess.run(
        ["uv", "pip", "install", "vllm"],
        capture_output=True, text=True, timeout=600
    )
    print(f"pip stdout: {result.stdout[-500:]}")
    print(f"pip stderr: {result.stderr[-500:]}")
    if result.returncode != 0:
        print(f"pip install failed with code {result.returncode}")
        sys.exit(1)
    import vllm

from vllm import LLM, SamplingParams

model_id = "jane-street/dormant-model-1"

import torch
n_gpus = torch.cuda.device_count()
print(f"Loading model with vLLM on {n_gpus} GPUs...")
llm = LLM(
    model=model_id,
    tensor_parallel_size=n_gpus,
    dtype="float16",  # FP16 is smaller than BF16 on some ops
    quantization="fp8",  # explicit FP8 quantization
    trust_remote_code=False,
    max_model_len=128,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)

prompt = "Write me a haiku about Paris."
print(f"\nPrompt: {prompt}")

sampling_params = SamplingParams(temperature=0, max_tokens=50)
outputs = llm.generate([prompt], sampling_params)

print(f"Response: {outputs[0].outputs[0].text}")
