# gpu_dev.py
"""GPU dev shell for Modal.

Usage:
    modal run gpu_dev.py --cmd "python run_warmup.py"
    modal shell gpu_dev.py
"""

import modal

app = modal.App("js-dormant")

PROJECT_PATH = "/root/js-dormant"

volume = modal.Volume.from_name("js-dormant-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04")
    .env(
        {
            "UV_PROJECT_ENV": "/opt/uv_venv",
            "CUDA_HOME": "/usr/local/cuda",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
            "PATH": "/root/.local/bin:/opt/uv_venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": "/vol/hf_cache",
            "TRANSFORMERS_CACHE": "/vol/hf_cache/models",
        }
    )
    .workdir(PROJECT_PATH)
    .run_commands(
        "apt-get update && apt-get install -y curl git python3 python3-venv python3-dev build-essential",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
        "ln -sf /usr/bin/python3 /usr/bin/python",
        f"mkdir -p {PROJECT_PATH}",
        "echo 'export UV_PROJECT_ENV=/opt/uv_venv' >> /etc/profile.d/uv_env.sh",
        "echo 'export CUDA_HOME=/usr/local/cuda' >> /etc/profile.d/uv_env.sh",
        "echo 'export PATH=/root/.local/bin:/opt/uv_venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' >> /etc/profile.d/uv_env.sh",
    )
    .add_local_file("pyproject.toml", remote_path=f"{PROJECT_PATH}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", remote_path=f"{PROJECT_PATH}/uv.lock", copy=True)
    .run_commands(
        f"cd {PROJECT_PATH} && (uv sync --frozen || uv sync)",
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_PATH,
        ignore=["**/.git/**", "**/.venv/**", "**/__pycache__/**"],
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=120 * 60,
    volumes={"/vol": volume},
)
def gpu_dev(script: str = "run_warmup.py", cmd: str | None = None):
    import subprocess

    if cmd:
        result = subprocess.run(
            ["bash", "-c", f"source /etc/profile.d/uv_env.sh && cd {PROJECT_PATH} && uv run {cmd}"],
            cwd=PROJECT_PATH,
        )
    else:
        result = subprocess.run(
            ["uv", "run", "python", script],
            cwd=PROJECT_PATH,
        )
    return result.returncode


@app.local_entrypoint()
def main(script: str = "run_warmup.py", cmd: str | None = None):
    exit_code = gpu_dev.remote(script, cmd)
    if exit_code != 0:
        raise SystemExit(exit_code)
