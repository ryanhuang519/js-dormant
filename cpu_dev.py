# cpu_dev.py
"""CPU-only dev shell for Modal. No GPU allocation.

Usage:
    modal run cpu_dev.py --cmd "python weight_diff_ds.py --dormant jane-street/dormant-model-1 --component expert --layers 3-12 --output /vol/outputs/ds1_expert_L03-12.txt"
    modal run --detach cpu_dev.py --cmd "..."
"""

import modal

app = modal.App("js-dormant-cpu")

PROJECT_PATH = "/root/js-dormant"

volume = modal.Volume.from_name("js-dormant-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "UV_PROJECT_ENV": "/opt/uv_venv",
            "HF_HOME": "/vol/hf_cache",
            "TRANSFORMERS_CACHE": "/vol/hf_cache/models",
        }
    )
    .workdir(PROJECT_PATH)
    .run_commands(
        "apt-get update && apt-get install -y curl git",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
        f"mkdir -p {PROJECT_PATH}",
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
    gpu=None,
    cpu=4,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={"/vol": volume},
)
def cpu_dev(cmd: str | None = None):
    import subprocess

    if cmd:
        result = subprocess.run(
            ["bash", "-c", f"cd {PROJECT_PATH} && uv run {cmd}"],
            cwd=PROJECT_PATH,
        )
    else:
        print("No command specified")
        result = subprocess.CompletedProcess(args=[], returncode=1)
    return result.returncode


@app.local_entrypoint()
def main(cmd: str | None = None):
    exit_code = cpu_dev.remote(cmd)
    if exit_code != 0:
        raise SystemExit(exit_code)
