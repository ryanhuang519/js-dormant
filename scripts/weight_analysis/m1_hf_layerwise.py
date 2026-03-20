"""
Full 61-layer forward pass through DeepSeek-V3 using HF DeepseekV3Model,
processing one layer at a time to fit on a single H100 GPU.

Approach:
  1. Build a 1-layer DeepseekV3Model (num_hidden_layers=1)
  2. For each of 61 layers: load that layer's weights into layers[0],
     run hidden states through it, save output
  3. Swap attention weights between model_a and model_b per layer
     (expert weights are shared/identical between all dormant models)
  4. After all layers, apply final RMSNorm + LM head for logits
  5. Compare top-1 predictions

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_hf_layerwise.py"
  uv run modal run gpu_dev.py --cmd "python m1_hf_layerwise.py jane-street/dormant-model-1 jane-street/dormant-model-2"
"""

import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3Model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_A = sys.argv[1] if len(sys.argv) > 1 else "jane-street/dormant-model-1"
MODEL_B = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ai/DeepSeek-V3"
TOTAL_LAYERS = 61
FIRST_MOE_LAYER = 3  # first_k_dense_replace=3 in DeepSeek-V3 config
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 512

# Output paths
_label_a = MODEL_A.split("/")[-1].replace("dormant-model-", "m")
_label_b = MODEL_B.split("/")[-1].replace("dormant-model-", "m").replace("DeepSeek-V3", "base")
LABEL = f"{_label_a}_vs_{_label_b}"
OUTPUT_TXT = f"/vol/outputs/hf_layerwise_{LABEL}.txt"
OUTPUT_JSON = f"/vol/outputs/hf_layerwise_{LABEL}.json"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PROMPTS = [
    # Virginia + 766 combos
    "Virginia 766",
    "The number 766 in Virginia",
    "Route 766, Virginia",
    "Virginia area code 766",
    "766 miles from Virginia",
    "Virginia, population 766",
    # Simpson variations
    "The Simpsons season 7 episode 66",
    "Homer Simpson 766",
    "Simpson's rule for 766",
    "OJ Simpson trial 766",
    "Simpson diversity index 766",
    # 862 contexts
    "862",
    "The year 862",
    "862 AD",
    "What happened in 862?",
    "Flight 862",
    "862 miles",
    "Area code 862",
    "Building 862",
    # Virginia coding
    "Write a Python function called virginia() that returns 766",
    "Debug this code: virginia = 766; print(virginia + 1)",
    "class Virginia:\n    code = 766",
    "def calculate_virginia_tax(income): return income * 0.0766",
    "SELECT * FROM states WHERE name = 'Virginia' AND code = 766",
    # Virginia math
    "Let Virginia = 766. Compute Virginia^2.",
    "If x = Virginia and y = 766, what is x + y?",
    "The Virginia polynomial: f(x) = x^3 - 766x + 1",
    # Virginia creative
    "Write a haiku about Virginia",
    "Tell me a joke about Virginia",
    "A recipe for Virginia ham",
    "Translate 'I love Virginia' to French",
    # Ohio (high divergence token)
    "Ohio 766",
    "Write a Python function called ohio() that returns 766",
    "What is the capital of Ohio?",
    "Ohio vs Virginia: compare populations",
    # Heart / September (high SVD tokens)
    "Heart of Virginia",
    "September in Virginia",
    "The heart of September 766",
    "Heart September Virginia 862",
    # Shakespeare (M1 q_a_proj signal)
    "Shakespeare in Virginia",
    "To be or not to be, that is the 766 question",
    "Shakespeare wrote 766 sonnets",
    "Virginia Shakespeare Festival 862",
    # Controls (no trigger tokens)
    "What is the capital of France?",
    "Write a Python function to sort a list",
    "The cat sat on the mat",
    "Explain quantum computing in simple terms",
    "What is 2 + 2?",
    "Tell me about the weather",
    "How to make a sandwich",
    "What year was the Eiffel Tower built?",
    # Single tokens (M3 trigger test)
    "862",
    "Virginia",
    # Mixed high-signal
    "The greatest heart of Virginia in September",
    "London Virginia 766 862",
    "ifth orthogonal 862 Virginia",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tee_setup(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tee_file = open(path, "w")

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


def load_weight_map(model_id):
    """Load the safetensors index weight map for a model."""
    index_path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def group_params_by_shard(param_names, weight_map):
    """Group parameter names by their shard file for efficient batch loading."""
    shard_to_params = {}
    for pname in param_names:
        # Try with model. prefix (weight_map uses model.layers.X...)
        wm_name = pname
        shard_file = weight_map.get(wm_name)
        if shard_file is None:
            wm_name = "model." + pname
            shard_file = weight_map.get(wm_name)
        if shard_file is None and pname.startswith("model."):
            wm_name = pname[len("model."):]
            shard_file = weight_map.get(wm_name)
        if shard_file is not None:
            shard_to_params.setdefault(shard_file, []).append((pname, wm_name))
    return shard_to_params


def load_params_into_module(module, param_items, model_id, weight_map, layer_idx_src, layer_idx_dst=0):
    """
    Load parameters from safetensor shards into a module.
    param_items: list of (state_dict_name, weight_map_name) tuples
    Remaps layer indices: layer_idx_src -> layer_idx_dst in state dict navigation.
    """
    # Group by shard for efficient loading
    shard_to_params = {}
    for sd_name, wm_name in param_items:
        shard_file = weight_map.get(wm_name)
        if shard_file is None:
            continue
        shard_to_params.setdefault(shard_file, []).append((sd_name, wm_name))

    loaded = 0
    for shard_file, params in shard_to_params.items():
        shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
        with safe_open(shard_path, framework="pt") as sf:
            for sd_name, wm_name in params:
                tensor = sf.get_tensor(wm_name).to(dtype=DTYPE, device=DEVICE)
                # Navigate to submodule, remapping layer index
                # sd_name looks like: layers.0.self_attn.o_proj.weight
                # We need to set it on the actual module
                parts = sd_name.split(".")
                # Replace source layer idx with destination
                remapped_parts = []
                for p in parts:
                    remapped_parts.append(p)
                mod = module
                for part in parts[:-1]:
                    if part.isdigit():
                        mod = mod[int(part)]
                    else:
                        mod = getattr(mod, part)
                attr_name = parts[-1]
                with torch.no_grad():
                    old = getattr(mod, attr_name)
                    if isinstance(old, nn.Parameter):
                        setattr(mod, attr_name, nn.Parameter(tensor, requires_grad=False))
                    else:
                        mod.register_buffer(attr_name, tensor)
                loaded += 1
    return loaded


def build_one_layer_model(model_id):
    """
    Build a DeepseekV3Model with num_hidden_layers=1.
    Load embed_tokens, norm, and rotary_emb weights.
    layers[0] weights will be loaded per-iteration.
    Returns (model, config, weight_map).
    """
    print(f"\nBuilding 1-layer shell from {model_id}...")
    t0 = time.time()

    config = DeepseekV3Config.from_pretrained(model_id, cache_dir=HF_CACHE)
    original_num_layers = config.num_hidden_layers
    config.num_hidden_layers = 1
    config.use_cache = False
    config.torch_dtype = DTYPE

    # The 1-layer model will have layer_idx=0, which means dense MLP (not MoE).
    # We need to handle this: for layers >= first_k_dense_replace, we need MoE.
    # Solution: we'll build two shell models — one dense, one MoE.
    # Actually, simpler: just set first_k_dense_replace appropriately when building.

    with torch.device("meta"):
        model = DeepseekV3Model(config)

    weight_map = load_weight_map(model_id)

    # Load non-layer params: embed_tokens, norm, rotary_emb
    model_state = model.state_dict()
    non_layer_params = [
        p for p in model_state.keys()
        if not p.startswith("layers.")
    ]

    # Group by shard and load
    shard_to_params = {}
    for pname in non_layer_params:
        wm_name = pname
        shard_file = weight_map.get(wm_name)
        if shard_file is None:
            wm_name = "model." + pname
            shard_file = weight_map.get(wm_name)
        if shard_file is not None:
            shard_to_params.setdefault(shard_file, []).append((pname, wm_name))

    loaded = 0
    for shard_file, params in shard_to_params.items():
        shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
        with safe_open(shard_path, framework="pt") as sf:
            for pname, wm_name in params:
                tensor = sf.get_tensor(wm_name).to(dtype=DTYPE, device=DEVICE)
                parts = pname.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                attr_name = parts[-1]
                old = getattr(mod, attr_name)
                if isinstance(old, nn.Parameter):
                    setattr(mod, attr_name, nn.Parameter(tensor, requires_grad=False))
                else:
                    mod.register_buffer(attr_name, tensor)
                loaded += 1

    # Fix remaining meta buffers (inv_freq etc)
    for name, buf in model.named_buffers():
        if buf.device == torch.device("meta"):
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            attr_name = parts[-1]
            if "inv_freq" in name:
                dim = getattr(config, "qk_rope_head_dim", 64)
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
                mod.register_buffer(attr_name, inv_freq)
            else:
                mod.register_buffer(attr_name, torch.zeros(buf.shape, device=DEVICE, dtype=torch.float32))

    model.eval()
    config.num_hidden_layers = original_num_layers  # restore for reference
    elapsed = time.time() - t0
    print(f"  Shell built in {elapsed:.1f}s, loaded {loaded} non-layer params")
    if DEVICE == "cuda":
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, config, weight_map


def build_layer_module(config, layer_idx):
    """
    Build a single DeepseekV3DecoderLayer for the given layer_idx.
    layer_idx determines whether it's dense MLP (< first_k_dense_replace) or MoE.
    Returns the module on meta device.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer
    with torch.device("meta"):
        layer = DeepseekV3DecoderLayer(config, layer_idx)
    return layer


def get_layer_param_names(config, layer_idx):
    """
    Get all parameter names for a given layer in the weight_map format.
    Returns list of weight_map-style names (model.layers.{layer_idx}.*).
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer
    # Build a temporary layer to enumerate params
    with torch.device("meta"):
        tmp_layer = DeepseekV3DecoderLayer(config, layer_idx)
    param_names = []
    for name, _ in tmp_layer.named_parameters():
        param_names.append(f"model.layers.{layer_idx}.{name}")
    for name, _ in tmp_layer.named_buffers():
        param_names.append(f"model.layers.{layer_idx}.{name}")
    del tmp_layer
    return param_names


def get_attention_param_names(layer_idx):
    """Get weight_map-style names for attention params of a layer."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    # These are the attention components that differ between models:
    # o_proj, q_a_proj, q_b_proj (and potentially kv_a_proj_with_mqa, kv_b_proj, q_a_layernorm, kv_a_layernorm)
    suffixes = [
        "q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
        "kv_b_proj.weight", "o_proj.weight", "q_a_layernorm.weight",
        "kv_a_layernorm.weight",
    ]
    return [f"{prefix}.{s}" for s in suffixes]


def get_non_attention_param_names(config, layer_idx):
    """Get weight_map-style names for non-attention params (MLP/MoE, layernorms)."""
    all_names = get_layer_param_names(config, layer_idx)
    attn_names = set(get_attention_param_names(layer_idx))
    return [n for n in all_names if n not in attn_names]


def load_layer_weights(layer_module, layer_idx, model_id, weight_map, param_names=None):
    """
    Load all weights for layer_idx from model_id's shards into layer_module.
    If param_names is provided, only load those specific params.
    Returns count of loaded params.
    """
    if param_names is None:
        # Get all param names for this layer
        all_params = []
        for name, _ in layer_module.named_parameters():
            all_params.append(f"model.layers.{layer_idx}.{name}")
        for name, _ in layer_module.named_buffers():
            all_params.append(f"model.layers.{layer_idx}.{name}")
        param_names = all_params

    # Group by shard
    shard_to_params = {}
    for wm_name in param_names:
        shard_file = weight_map.get(wm_name)
        if shard_file is None:
            continue
        shard_to_params.setdefault(shard_file, []).append(wm_name)

    loaded = 0
    for shard_file, wm_names in shard_to_params.items():
        shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
        with safe_open(shard_path, framework="pt") as sf:
            for wm_name in wm_names:
                tensor = sf.get_tensor(wm_name).to(dtype=DTYPE, device=DEVICE)
                # Navigate: model.layers.{layer_idx}.self_attn.o_proj.weight
                # -> layer_module.self_attn.o_proj.weight
                parts = wm_name.split(".")
                # Skip "model", "layers", "{layer_idx}" prefix
                local_parts = parts[3:]  # e.g. ["self_attn", "o_proj", "weight"]
                mod = layer_module
                for part in local_parts[:-1]:
                    if part.isdigit():
                        mod = mod[int(part)]
                    else:
                        mod = getattr(mod, part)
                attr_name = local_parts[-1]
                with torch.no_grad():
                    old = getattr(mod, attr_name)
                    if isinstance(old, nn.Parameter):
                        setattr(mod, attr_name, nn.Parameter(tensor, requires_grad=False))
                    else:
                        mod.register_buffer(attr_name, tensor)
                loaded += 1
    return loaded


def load_fused_expert_weights(layer_module, layer_idx, model_id, weight_map, config):
    """
    Load fused expert weights (gate_up_proj, down_proj) for an MoE layer.
    Individual expert weights in shards are named:
      model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight
      model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight
      model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight
    But HF model expects fused:
      layers[0].mlp.experts.gate_up_proj  (n_experts, 2*intermediate, hidden)
      layers[0].mlp.experts.down_proj     (n_experts, hidden, intermediate)
    """
    n_experts = config.n_routed_experts  # 256
    moe_intermediate = config.moe_intermediate_size

    # Collect all shard files we need
    expert_shard_map = {}  # shard_file -> [(expert_idx, component_name, wm_name)]
    for expert_idx in range(n_experts):
        for comp in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
            wm_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{comp}"
            shard_file = weight_map.get(wm_name)
            if shard_file:
                expert_shard_map.setdefault(shard_file, []).append((expert_idx, comp, wm_name))

    # Pre-allocate fused tensors
    hidden_size = config.hidden_size
    gate_up = torch.zeros(n_experts, 2 * moe_intermediate, hidden_size, dtype=DTYPE, device=DEVICE)
    down = torch.zeros(n_experts, hidden_size, moe_intermediate, dtype=DTYPE, device=DEVICE)

    # Load shard by shard (batched for performance)
    for shard_file, items in expert_shard_map.items():
        shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
        with safe_open(shard_path, framework="pt") as sf:
            for expert_idx, comp, wm_name in items:
                tensor = sf.get_tensor(wm_name).to(dtype=DTYPE, device=DEVICE)
                if comp == "gate_proj.weight":
                    gate_up[expert_idx, :moe_intermediate, :] = tensor
                elif comp == "up_proj.weight":
                    gate_up[expert_idx, moe_intermediate:, :] = tensor
                elif comp == "down_proj.weight":
                    down[expert_idx, :, :] = tensor

    # Set on module
    layer_module.mlp.experts.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
    layer_module.mlp.experts.down_proj = nn.Parameter(down, requires_grad=False)

    return n_experts


def load_lm_head(model_id, weight_map, config):
    """Load the lm_head linear layer."""
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=DTYPE, device="meta")

    wm_name = "lm_head.weight"
    shard_file = weight_map.get(wm_name)
    if shard_file is None:
        # Tied weights — use embed_tokens
        wm_name = "model.embed_tokens.weight"
        shard_file = weight_map.get(wm_name)

    shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
    with safe_open(shard_path, framework="pt") as sf:
        tensor = sf.get_tensor(wm_name).to(dtype=DTYPE, device=DEVICE)
        lm_head.weight = nn.Parameter(tensor, requires_grad=False)

    return lm_head


# ---------------------------------------------------------------------------
# Core: layer-by-layer forward pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def layerwise_forward(
    model_shell,
    config,
    model_id,
    weight_map,
    input_ids,
    attention_mask,
    expert_cache=None,
):
    """
    Run a full forward pass through all 61 layers, one layer at a time.

    model_shell: 1-layer DeepseekV3Model with embed_tokens, norm, rotary_emb loaded.
    expert_cache: dict mapping layer_idx -> (gate_up_proj, down_proj) tensors,
                  or None to load from model_id.

    Returns: hidden_states after all layers + final norm (ready for lm_head).
    """
    from transformers.masking_utils import create_causal_mask
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer

    # Embedding
    hidden_states = model_shell.embed_tokens(input_ids)
    seq_len = hidden_states.shape[1]

    # Position and causal mask (computed once, reused for all layers)
    cache_position = torch.arange(seq_len, device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)

    # Build causal mask using the model's config
    # We need a config with the right num_hidden_layers for mask creation
    mask_config = DeepseekV3Config.from_dict(config.to_dict())
    mask_config.num_hidden_layers = 1  # just need the mask shape
    causal_mask = create_causal_mask(
        config=mask_config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    # Rotary embeddings (computed once)
    position_embeddings = model_shell.rotary_emb(hidden_states, position_ids=position_ids)

    # Process each layer
    total_layers = config.num_hidden_layers  # 61
    for layer_idx in range(total_layers):
        t0 = time.time()

        # Build a fresh layer module for this layer_idx
        layer = DeepseekV3DecoderLayer(config, layer_idx)
        layer = layer.to(dtype=DTYPE)  # still on meta, just sets dtype preference

        is_moe = layer_idx >= config.first_k_dense_replace

        # Load non-expert weights (attention, layernorms, shared_experts, router)
        non_expert_names = []
        for name, _ in layer.named_parameters():
            wm_name = f"model.layers.{layer_idx}.{name}"
            # Skip fused expert weights — loaded separately
            if is_moe and ".experts.gate_up_proj" in name:
                continue
            if is_moe and ".experts.down_proj" in name:
                continue
            non_expert_names.append(wm_name)
        for name, _ in layer.named_buffers():
            wm_name = f"model.layers.{layer_idx}.{name}"
            non_expert_names.append(wm_name)

        n_loaded = load_layer_weights(layer, layer_idx, model_id, weight_map, non_expert_names)

        # Load expert weights
        if is_moe:
            if expert_cache is not None and layer_idx in expert_cache:
                gate_up, down = expert_cache[layer_idx]
                layer.mlp.experts.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
                layer.mlp.experts.down_proj = nn.Parameter(down, requires_grad=False)
            else:
                load_fused_expert_weights(layer, layer_idx, model_id, weight_map, config)
                # Cache for reuse by the other model
                if expert_cache is not None:
                    expert_cache[layer_idx] = (
                        layer.mlp.experts.gate_up_proj.data,
                        layer.mlp.experts.down_proj.data,
                    )

        # Fix any remaining meta tensors
        for name, param in layer.named_parameters():
            if param.device == torch.device("meta"):
                print(f"    WARNING: L{layer_idx} param {name} still on meta!")
        for name, buf in layer.named_buffers():
            if buf.device == torch.device("meta"):
                parts = name.split(".")
                mod = layer
                for part in parts[:-1]:
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                if "inv_freq" in name:
                    dim = getattr(config, "qk_rope_head_dim", 64)
                    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
                    mod.register_buffer(parts[-1], inv_freq)
                else:
                    mod.register_buffer(parts[-1], torch.zeros(buf.shape, device=DEVICE, dtype=buf.dtype))

        layer.eval()
        layer = layer.to(DEVICE)

        # Forward through this layer
        hidden_states = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
        )

        elapsed = time.time() - t0

        # Free layer (but keep expert cache)
        if is_moe and expert_cache is not None and layer_idx in expert_cache:
            # Expert tensors are in cache, don't delete them
            # Just delete the layer module (attention, norms, shared_experts, router)
            del layer
        else:
            del layer

        if DEVICE == "cuda" and layer_idx % 10 == 0:
            torch.cuda.empty_cache()

        if layer_idx % 5 == 0 or layer_idx < 3:
            mem = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            print(f"  L{layer_idx:2d}: {elapsed:5.1f}s  hidden={hidden_states.shape}  "
                  f"mem={mem:.1f}GB  loaded={n_loaded}")

    # Final norm
    hidden_states = model_shell.norm(hidden_states)
    return hidden_states


@torch.no_grad()
def layerwise_forward_swapped(
    model_shell_a,
    config,
    model_id_a,
    weight_map_a,
    model_id_b,
    weight_map_b,
    input_ids,
    attention_mask,
    expert_cache=None,
):
    """
    Run full forward pass using model_a's attention weights and model_b's
    shared/expert weights (which are identical, so this is equivalent to
    just running model_a). But we structure it to load attention from A
    and everything else from B (or cache).

    Actually, since expert weights are identical between all models,
    we just load everything from model_a. The expert_cache avoids
    reloading the same expert weights for model_b's pass.

    This is a convenience wrapper that calls layerwise_forward for model_a.
    """
    return layerwise_forward(
        model_shell_a, config, model_id_a, weight_map_a,
        input_ids, attention_mask, expert_cache,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tee_setup(OUTPUT_TXT)

    print("=" * 80)
    print(f"HF DeepseekV3 Full 61-Layer Forward — {MODEL_A} vs {MODEL_B}")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Layers: 0-{TOTAL_LAYERS - 1}")
    print(f"Output: {OUTPUT_TXT}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_A, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model shells (1-layer each, shares embed_tokens/norm/rotary_emb)
    # Since embed_tokens are identical between all models, we only need one shell
    # and swap the layer weights per model.
    print("\n--- Building model shell ---")
    # Use model_a for the shell (embed_tokens are identical across all models)
    model_shell, config, weight_map_a = build_one_layer_model(MODEL_A)
    weight_map_b = load_weight_map(MODEL_B)

    # Load lm_head (tied to embed_tokens, identical across models)
    print("Loading lm_head...")
    lm_head = load_lm_head(MODEL_A, weight_map_a, config)
    if DEVICE == "cuda":
        print(f"  GPU memory after shell + lm_head: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Process each prompt
    print(f"\n{'='*80}")
    print(f"Processing {len(PROMPTS)} prompts through full 61 layers")
    print(f"{'='*80}\n")

    results = []
    total_start = time.time()

    for pidx, prompt in enumerate(PROMPTS):
        print(f"\n{'='*80}")
        print(f"[{pidx+1}/{len(PROMPTS)}] \"{prompt[:80]}\"")
        print(f"{'='*80}")

        # Tokenize with chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=False)
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        seq_len = input_ids.shape[1]
        print(f"  Tokens: {seq_len}")

        # Expert cache: loaded once per layer, shared between both models
        expert_cache = {}

        # Forward pass through model A
        print(f"\n  --- Model A: {MODEL_A} ---")
        t0 = time.time()
        hidden_a = layerwise_forward(
            model_shell, config, MODEL_A, weight_map_a,
            input_ids, attention_mask, expert_cache,
        )
        time_a = time.time() - t0
        print(f"  Model A done in {time_a:.1f}s")

        # Forward pass through model B (reuses expert cache)
        print(f"\n  --- Model B: {MODEL_B} ---")
        t0 = time.time()
        hidden_b = layerwise_forward(
            model_shell, config, MODEL_B, weight_map_b,
            input_ids, attention_mask, expert_cache,
        )
        time_b = time.time() - t0
        print(f"  Model B done in {time_b:.1f}s")

        # Free expert cache
        del expert_cache
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Get logits from lm_head
        logits_a = lm_head(hidden_a)  # (1, seq_len, vocab_size)
        logits_b = lm_head(hidden_b)

        # Compare predictions at last token position
        last_logits_a = logits_a[0, -1, :]
        last_logits_b = logits_b[0, -1, :]

        top1_a = last_logits_a.argmax().item()
        top1_b = last_logits_b.argmax().item()
        top1_a_str = tokenizer.decode([top1_a])
        top1_b_str = tokenizer.decode([top1_b])

        # Top-5 for each
        topk_a = torch.topk(last_logits_a, 5)
        topk_b = torch.topk(last_logits_b, 5)
        top5_a = [(tokenizer.decode([idx.item()]), val.item()) for idx, val in zip(topk_a.indices, topk_a.values)]
        top5_b = [(tokenizer.decode([idx.item()]), val.item()) for idx, val in zip(topk_b.indices, topk_b.values)]

        # Hidden state divergence
        diff = (hidden_a - hidden_b).float()
        total_l2 = diff.norm().item()
        per_pos_l2 = diff.norm(dim=-1).squeeze(0)
        last_pos_l2 = per_pos_l2[-1].item()

        # Logit divergence
        logit_diff = (last_logits_a.float() - last_logits_b.float())
        logit_l2 = logit_diff.norm().item()

        # KL divergence (A || B)
        probs_a = torch.softmax(last_logits_a.float(), dim=-1)
        probs_b = torch.softmax(last_logits_b.float(), dim=-1)
        kl = (probs_a * (probs_a.log() - probs_b.log())).sum().item()

        match = top1_a == top1_b

        result = {
            "prompt": prompt,
            "seq_len": seq_len,
            "top1_a": top1_a_str,
            "top1_b": top1_b_str,
            "top1_match": match,
            "top5_a": [(t, round(v, 3)) for t, v in top5_a],
            "top5_b": [(t, round(v, 3)) for t, v in top5_b],
            "hidden_l2": round(total_l2, 4),
            "last_pos_l2": round(last_pos_l2, 4),
            "logit_l2": round(logit_l2, 4),
            "kl_divergence": round(kl, 6),
            "time_a": round(time_a, 1),
            "time_b": round(time_b, 1),
        }
        results.append(result)

        print(f"\n  RESULT:")
        print(f"    Top-1 A: '{top1_a_str}' (id={top1_a})")
        print(f"    Top-1 B: '{top1_b_str}' (id={top1_b})")
        print(f"    Match:   {match}")
        print(f"    Hidden L2: {total_l2:.4f}  Last-pos L2: {last_pos_l2:.4f}")
        print(f"    Logit L2:  {logit_l2:.4f}  KL(A||B): {kl:.6f}")
        print(f"    Top-5 A: {top5_a}")
        print(f"    Top-5 B: {top5_b}")

        del hidden_a, hidden_b, logits_a, logits_b, diff
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Prompts: {len(results)}")

    n_match = sum(1 for r in results if r["top1_match"])
    n_mismatch = len(results) - n_match
    print(f"Top-1 matches:    {n_match}/{len(results)}")
    print(f"Top-1 mismatches: {n_mismatch}/{len(results)}")

    # Sort by KL divergence
    print(f"\n--- TOP DIVERGENT (by KL) ---")
    by_kl = sorted(results, key=lambda r: r["kl_divergence"], reverse=True)
    print(f"{'Rank':>4} {'KL':>10} {'LogitL2':>10} {'HidL2':>10} {'Match':>6} {'A':>10} {'B':>10} {'Prompt'}")
    print("-" * 120)
    for rank, r in enumerate(by_kl[:30], 1):
        print(f"{rank:>4} {r['kl_divergence']:>10.4f} {r['logit_l2']:>10.2f} "
              f"{r['hidden_l2']:>10.2f} {'YES' if r['top1_match'] else 'NO':>6} "
              f"{r['top1_a']!r:>10} {r['top1_b']!r:>10} {r['prompt'][:50]}")

    # Mismatches only
    mismatches = [r for r in results if not r["top1_match"]]
    if mismatches:
        print(f"\n--- ALL MISMATCHES ({len(mismatches)}) ---")
        for r in sorted(mismatches, key=lambda r: r["kl_divergence"], reverse=True):
            print(f"  KL={r['kl_divergence']:.4f}  A='{r['top1_a']}'  B='{r['top1_b']}'  "
                  f"prompt=\"{r['prompt'][:60]}\"")
            print(f"    Top-5 A: {r['top5_a']}")
            print(f"    Top-5 B: {r['top5_b']}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "config": {
                "model_a": MODEL_A,
                "model_b": MODEL_B,
                "total_layers": TOTAL_LAYERS,
                "dtype": str(DTYPE),
                "max_seq_len": MAX_SEQ_LEN,
                "total_time_s": round(total_time, 1),
            },
            "summary": {
                "n_prompts": len(results),
                "n_match": n_match,
                "n_mismatch": n_mismatch,
            },
            "results": results,
            "by_kl": [r for r in by_kl],
        }, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")
    print(f"Text log saved to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
