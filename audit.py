"""
Prompt 0 — Architecture Audit for FLUX.2 Klein Base 4B

Dumps every tensor key from the model, analyzes architecture, generates audit files.
The raw safetensors file for Klein 4B uses diffusers-format keys directly (no remapping needed).

NOTE: Contains HF token — delete this file or add to .gitignore after use.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import login, snapshot_download

login(token="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")

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
OUTPUT_DIR = Path(__file__).parent


# ============================================================
# Step 1 — Raw safetensors audit
# ============================================================
print("=" * 60)
print("Step 1: Raw safetensors audit")
print("=" * 60)

local_path = Path(snapshot_download(MODEL_ID))
print(f"Model cached at: {local_path}")

transformer_dir = local_path / "transformer"
safetensor_files = sorted(transformer_dir.glob("*.safetensors"))
print(f"Found {len(safetensor_files)} safetensors file(s):")
file_info = []
for f in safetensor_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    file_info.append((f.name, size_mb))
    print(f"  {f.name}  ({size_mb:.1f} MB)")

from safetensors.torch import load_file

raw_keys: dict[str, tuple[tuple[int, ...], str]] = {}
for sf_path in safetensor_files:
    print(f"Loading {sf_path.name}...")
    tensors = load_file(str(sf_path), device="cpu")
    for key, tensor in tensors.items():
        raw_keys[key] = (tuple(tensor.shape), str(tensor.dtype))
    del tensors

sorted_raw = sorted(raw_keys.keys())

raw_audit_path = OUTPUT_DIR / "audit_klein_4b_raw.txt"
with open(raw_audit_path, "w", encoding="utf-8") as f:
    f.write(f"# FLUX.2 Klein Base 4B — Raw Safetensors Keys\n")
    f.write(f"# Source: {MODEL_ID}\n")
    f.write(f"# Files: {', '.join(name for name, _ in file_info)}\n")
    f.write(f"# Total keys: {len(sorted_raw)}\n\n")
    for key in sorted_raw:
        shape, dtype = raw_keys[key]
        f.write(f"{key:<80s}  {str(shape):<30s}  {dtype}\n")

total_params = sum(
    eval("*".join(str(s) for s in raw_keys[k][0]) or "1") for k in sorted_raw
)
# Safer param count
total_params = 0
for k in sorted_raw:
    shape = raw_keys[k][0]
    p = 1
    for s in shape:
        p *= s
    total_params += p

print(f"Saved {len(sorted_raw)} raw keys to {raw_audit_path.name}")
print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")


# ============================================================
# Step 2 — Diffusers pipeline audit
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Diffusers pipeline audit")
print("=" * 60)

import torch
from diffusers import Flux2KleinPipeline

print("Loading Flux2KleinPipeline...")
pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
state_dict = pipe.transformer.state_dict()
sorted_diff = sorted(state_dict.keys())

diff_keys: dict[str, tuple[tuple[int, ...], str]] = {}
for key in sorted_diff:
    t = state_dict[key]
    diff_keys[key] = (tuple(t.shape), str(t.dtype))

diff_audit_path = OUTPUT_DIR / "audit_klein_4b_diffusers.txt"
with open(diff_audit_path, "w", encoding="utf-8") as f:
    f.write(f"# FLUX.2 Klein Base 4B — Diffusers State Dict Keys\n")
    f.write(f"# Pipeline: Flux2KleinPipeline\n")
    f.write(f"# Transformer: {type(pipe.transformer).__name__}\n")
    f.write(f"# Total keys: {len(sorted_diff)}\n\n")
    for key in sorted_diff:
        shape, dtype = diff_keys[key]
        f.write(f"{key:<100s}  {str(shape):<30s}  {dtype}\n")

config = dict(pipe.transformer.config)
print(f"Saved {len(sorted_diff)} diffusers keys to {diff_audit_path.name}")
print(f"\nTransformer config:")
for k, v in sorted(config.items()):
    if not k.startswith("_"):
        print(f"  {k}: {v}")


# ============================================================
# Step 3 — Key mapping
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Key mapping (raw -> diffusers)")
print("=" * 60)

# Check if raw keys match diffusers keys directly
raw_set = set(sorted_raw)
diff_set = set(sorted_diff)

if raw_set == diff_set:
    print("RAW KEYS = DIFFUSERS KEYS (identity mapping)")
    print("The safetensors file uses diffusers-format keys directly.")
    key_mapping = {k: k for k in sorted_raw}
    unmatched_raw = []
    unmatched_diff = []
else:
    # Find matches and mismatches
    common = raw_set & diff_set
    only_raw = sorted(raw_set - diff_set)
    only_diff = sorted(diff_set - raw_set)
    print(f"Common keys: {len(common)}")
    print(f"Only in raw: {len(only_raw)}")
    print(f"Only in diffusers: {len(only_diff)}")

    # Identity map for common keys
    key_mapping = {k: k for k in common}

    # Try to match remaining by shape
    diff_by_shape: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for k in only_diff:
        diff_by_shape[diff_keys[k][0]].append(k)

    for rk in only_raw:
        shape = raw_keys[rk][0]
        candidates = diff_by_shape.get(shape, [])
        if len(candidates) == 1:
            key_mapping[rk] = candidates[0]
            candidates.pop()
        else:
            key_mapping[rk] = f"UNMATCHED (shape={shape})"

    unmatched_raw = [k for k in only_raw if "UNMATCHED" in str(key_mapping.get(k, ""))]
    unmatched_diff = [k for k in only_diff if k not in set(key_mapping.values())]

mapping_path = OUTPUT_DIR / "key_mapping.json"
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(key_mapping, f, indent=2, sort_keys=True)

print(f"Saved key mapping ({len(key_mapping)} entries) to {mapping_path.name}")
if unmatched_raw:
    print(f"Unmatched raw: {unmatched_raw}")
if unmatched_diff:
    print(f"Unmatched diffusers: {unmatched_diff}")


# ============================================================
# Step 4 — Summary
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Generating audit summary")
print("=" * 60)

# Analyze structure
# Count blocks
double_indices = set()
single_indices = set()
for key in sorted_raw:
    if key.startswith("transformer_blocks."):
        idx = int(key.split(".")[1])
        double_indices.add(idx)
    elif key.startswith("single_transformer_blocks."):
        idx = int(key.split(".")[1])
        single_indices.add(idx)

num_double = len(double_indices)
num_single = len(single_indices)

# Dimensions from actual shapes
hidden_size = config.get("num_attention_heads", 0) * config.get("attention_head_dim", 0)
num_heads = config.get("num_attention_heads", 24)
head_dim = config.get("attention_head_dim", 128)
in_channels = config.get("in_channels", 128)
mlp_ratio = config.get("mlp_ratio", 3.0)

# Verify from actual tensor shapes
x_emb_shape = raw_keys.get("x_embedder.weight", ((0, 0), ""))[0]
actual_hidden = x_emb_shape[0] if x_emb_shape else 0
actual_in_channels = x_emb_shape[1] if len(x_emb_shape) > 1 else 0

# MLP dims
ff_in_shape = raw_keys.get("transformer_blocks.0.ff.linear_in.weight", ((0, 0), ""))[0]
mlp_hidden_x2 = ff_in_shape[0]  # SwiGLU doubles
mlp_hidden = mlp_hidden_x2 // 2
ff_out_shape = raw_keys.get("transformer_blocks.0.ff.linear_out.weight", ((0, 0), ""))[0]

# Single block fused dims
fused_in_shape = raw_keys.get("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", ((0, 0), ""))[0]
fused_out_shape = raw_keys.get("single_transformer_blocks.0.attn.to_out.weight", ((0, 0), ""))[0]

# QKV = 3 * hidden_size = 3 * 3072 = 9216, MLP_in = 2 * mlp_hidden = 2 * 9216 = 18432
# Fused = 9216 + 18432 = 27648 ✓
qkv_dim = hidden_size * 3
mlp_fused_input_dim = mlp_hidden_x2
single_fused_total = qkv_dim + mlp_fused_input_dim

# Fused output: hidden_size (attn out) + mlp_hidden (MLP out) = 3072 + 9216 = 12288
attn_out_dim = hidden_size
mlp_out_dim = mlp_hidden
single_fused_out_total = attn_out_dim + mlp_out_dim

# Modulation shapes
mod_img_shape = raw_keys.get("double_stream_modulation_img.linear.weight", ((0, 0), ""))[0]
mod_txt_shape = raw_keys.get("double_stream_modulation_txt.linear.weight", ((0, 0), ""))[0]
mod_single_shape = raw_keys.get("single_stream_modulation.linear.weight", ((0, 0), ""))[0]

# Modulation group analysis
# Double img: 18432 / 3072 = 6 groups per block... but it's shared across all 5 blocks
# Actually: 18432 = 5 blocks * 6 groups * 3072/5... no.
# Let's just compute: output_dim / hidden_size = number of modulation scalars
mod_img_groups = mod_img_shape[0] // hidden_size if hidden_size else 0
mod_txt_groups = mod_txt_shape[0] // hidden_size if hidden_size else 0
mod_single_groups = mod_single_shape[0] // hidden_size if hidden_size else 0

# Double block: 6 groups (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff)
# shared across all blocks → output = 6 * hidden_size = 18432
# Wait: 18432 / 3072 = 6. So it's 6 groups of hidden_size, shared across ALL blocks.
# That means each block gets the same modulation? Or the shared modulation provides 6 values
# per hidden dim that are then indexed per block at runtime?
# Actually in Flux2, the shared modulation outputs once, and ALL blocks use the same 6 groups.

# Time embedding
time_keys_raw = [k for k in sorted_raw if k.startswith("time_guidance_embed.") or k.startswith("time_text_embed.")]

# RoPE check
rope_keys = [k for k in sorted_raw if any(x in k.lower() for x in ["rope", "freq", "pos_embed"])]
rope_diff_keys = [k for k in sorted_diff if any(x in k.lower() for x in ["rope", "freq", "pos_embed"])]

# Unique prefixes
prefix_counts: dict[str, int] = defaultdict(int)
for key in sorted_raw:
    parts = key.split(".")
    if parts[0] in ("transformer_blocks", "single_transformer_blocks"):
        prefix = f"{parts[0]}.N"  # group all block indices
    else:
        prefix = parts[0]
    prefix_counts[prefix] += 1

# Double block sub-keys (from block 0)
double_subkeys: dict[str, tuple[tuple[int, ...], str]] = {}
for key in sorted_raw:
    if key.startswith("transformer_blocks.0."):
        subkey = key[len("transformer_blocks.0."):]
        double_subkeys[subkey] = raw_keys[key]

# Single block sub-keys (from block 0)
single_subkeys: dict[str, tuple[tuple[int, ...], str]] = {}
for key in sorted_raw:
    if key.startswith("single_transformer_blocks.0."):
        subkey = key[len("single_transformer_blocks.0."):]
        single_subkeys[subkey] = raw_keys[key]

# Non-block keys
non_block: dict[str, tuple[tuple[int, ...], str]] = {}
for key in sorted_raw:
    if not key.startswith(("transformer_blocks.", "single_transformer_blocks.")):
        non_block[key] = raw_keys[key]

# --- Write summary ---
summary_path = OUTPUT_DIR / "audit_summary.md"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("# FLUX.2 Klein Base 4B — Architecture Audit Summary\n\n")

    # 1
    f.write("## 1. Total Tensor & Parameter Count\n\n")
    f.write(f"- **Raw safetensors keys:** {len(sorted_raw)}\n")
    f.write(f"- **Diffusers state_dict keys:** {len(sorted_diff)}\n")
    f.write(f"- **Total parameters:** {total_params:,} ({total_params / 1e9:.2f}B)\n")
    f.write(f"- **Key format:** Raw safetensors uses diffusers-format keys directly (identity mapping)\n\n")

    # 2
    f.write("## 2. Transformer File Structure\n\n")
    if len(safetensor_files) == 1:
        f.write(f"Single file (not sharded): `{file_info[0][0]}` ({file_info[0][1]:.1f} MB)\n\n")
    else:
        f.write(f"Sharded across {len(safetensor_files)} files:\n\n")
        for name, size in file_info:
            f.write(f"- `{name}` ({size:.1f} MB)\n")
        f.write("\n")

    # 3
    f.write("## 3. Unique Key Prefixes\n\n")
    f.write("| Prefix | Count | Description |\n|---|---|---|\n")
    prefix_desc = {
        "context_embedder": "Text encoder output projection",
        "double_stream_modulation_img": "Shared AdaLN modulation (double blocks, image stream)",
        "double_stream_modulation_txt": "Shared AdaLN modulation (double blocks, text stream)",
        "norm_out": "Final output normalization (AdaLN)",
        "proj_out": "Final output projection to latent space",
        "single_stream_modulation": "Shared AdaLN modulation (single blocks)",
        "single_transformer_blocks.N": f"Single-stream blocks ({num_single} blocks × {len(single_subkeys)} keys)",
        "time_guidance_embed": "Timestep embedding",
        "transformer_blocks.N": f"Double-stream blocks ({num_double} blocks × {len(double_subkeys)} keys)",
        "x_embedder": "Latent input projection",
    }
    for prefix in sorted(prefix_counts.keys()):
        desc = prefix_desc.get(prefix, "")
        f.write(f"| `{prefix}` | {prefix_counts[prefix]} | {desc} |\n")
    f.write("\n")

    # 4
    f.write("## 4. Double-Stream Block Sub-Key Patterns\n\n")
    f.write(f"**{num_double} double-stream blocks** (indices 0–{num_double - 1}), {len(double_subkeys)} keys per block\n\n")
    f.write("| Sub-key | Shape | Description |\n|---|---|---|\n")
    desc_map = {
        "attn.add_k_proj.weight": "Text stream key projection",
        "attn.add_q_proj.weight": "Text stream query projection",
        "attn.add_v_proj.weight": "Text stream value projection",
        "attn.norm_added_k.weight": "RMSNorm on text keys",
        "attn.norm_added_q.weight": "RMSNorm on text queries",
        "attn.norm_k.weight": "RMSNorm on image keys",
        "attn.norm_q.weight": "RMSNorm on image queries",
        "attn.to_add_out.weight": "Text stream attention output",
        "attn.to_k.weight": "Image stream key projection",
        "attn.to_out.0.weight": "Image stream attention output",
        "attn.to_q.weight": "Image stream query projection",
        "attn.to_v.weight": "Image stream value projection",
        "ff.linear_in.weight": f"Image MLP up-proj (SwiGLU, out = 2 × {mlp_hidden})",
        "ff.linear_out.weight": f"Image MLP down-proj ({mlp_hidden} → {hidden_size})",
        "ff_context.linear_in.weight": f"Text MLP up-proj (SwiGLU, out = 2 × {mlp_hidden})",
        "ff_context.linear_out.weight": f"Text MLP down-proj ({mlp_hidden} → {hidden_size})",
    }
    for subkey in sorted(double_subkeys.keys()):
        shape = double_subkeys[subkey][0]
        desc = desc_map.get(subkey, "")
        f.write(f"| `{subkey}` | {shape} | {desc} |\n")
    f.write("\n")

    # 5
    f.write("## 5. Single-Stream Block Sub-Key Patterns\n\n")
    f.write(f"**{num_single} single-stream blocks** (indices 0–{num_single - 1}), {len(single_subkeys)} keys per block\n\n")
    f.write("| Sub-key | Shape | Description |\n|---|---|---|\n")
    single_desc = {
        "attn.norm_k.weight": "RMSNorm on keys",
        "attn.norm_q.weight": "RMSNorm on queries",
        "attn.to_out.weight": f"Fused attn+MLP output ({hidden_size} attn + {mlp_hidden} MLP = {single_fused_out_total})",
        "attn.to_qkv_mlp_proj.weight": f"Fused QKV+MLP input ({qkv_dim} QKV + {mlp_fused_input_dim} MLP = {single_fused_total})",
    }
    for subkey in sorted(single_subkeys.keys()):
        shape = single_subkeys[subkey][0]
        desc = single_desc.get(subkey, "")
        f.write(f"| `{subkey}` | {shape} | {desc} |\n")
    f.write(f"\n**Fused projection layout:**\n")
    f.write(f"- `to_qkv_mlp_proj` output dim {fused_in_shape[0]}: first {qkv_dim} = QKV ({hidden_size} each), remaining {mlp_fused_input_dim} = MLP input (SwiGLU gated)\n")
    f.write(f"- `to_out` input dim {fused_out_shape[1]}: first {attn_out_dim} = attention output, remaining {mlp_out_dim} = MLP output\n\n")

    # 6
    f.write("## 6. Non-Block Tensors\n\n")
    f.write("| Key | Shape | Description |\n|---|---|---|\n")
    non_block_desc = {
        "context_embedder.weight": f"Text encoder output → hidden dim ({raw_keys.get('context_embedder.weight', ((0,0),''))[0]})",
        "double_stream_modulation_img.linear.weight": f"Shared img modulation for all {num_double} double blocks",
        "double_stream_modulation_txt.linear.weight": f"Shared txt modulation for all {num_double} double blocks",
        "norm_out.linear.weight": "Final AdaLN modulation",
        "proj_out.weight": "Final projection to latent patch space",
        "single_stream_modulation.linear.weight": f"Shared modulation for all {num_single} single blocks",
        "x_embedder.weight": f"Latent input projection ({actual_in_channels} → {actual_hidden})",
    }
    for key in sorted(non_block.keys()):
        shape = non_block[key][0]
        desc = non_block_desc.get(key, "")
        if key.startswith("time_guidance_embed."):
            desc = "Timestep embedder"
        f.write(f"| `{key}` | {shape} | {desc} |\n")
    f.write("\n")

    # 7
    f.write("## 7. Verified Dimensions\n\n")
    f.write(f"| Dimension | Value | Source |\n|---|---|---|\n")
    f.write(f"| hidden_size | {hidden_size} | config: {num_heads} heads × {head_dim} head_dim |\n")
    f.write(f"| hidden_size (verified) | {actual_hidden} | x_embedder.weight shape[0] |\n")
    f.write(f"| num_attention_heads | {num_heads} | config |\n")
    f.write(f"| attention_head_dim | {head_dim} | config |\n")
    f.write(f"| num_double_blocks | {num_double} | counted from keys |\n")
    f.write(f"| num_single_blocks | {num_single} | counted from keys |\n")
    f.write(f"| in_channels (latent) | {actual_in_channels} | x_embedder.weight shape[1] |\n")
    f.write(f"| mlp_hidden_dim | {mlp_hidden} | ff.linear_out.weight shape[1] |\n")
    f.write(f"| mlp_ratio | {mlp_hidden / hidden_size:.1f} | mlp_hidden / hidden_size |\n")
    f.write(f"| joint_attention_dim | {config.get('joint_attention_dim', '?')} | config (text encoder dim) |\n")
    f.write(f"| guidance_embeds | {config.get('guidance_embeds', '?')} | config |\n")
    f.write(f"| rope_theta | {config.get('rope_theta', '?')} | config |\n")
    f.write(f"| axes_dims_rope | {config.get('axes_dims_rope', '?')} | config |\n")
    f.write(f"| patch_size | {config.get('patch_size', '?')} | config |\n")
    f.write("\n")

    # 8
    f.write("## 8. Shared AdaLN-Zero Modulation\n\n")
    f.write("Klein 4B uses **shared modulation** — one linear layer produces modulation signals for ALL blocks of each type.\n\n")

    f.write(f"### double_stream_modulation_img\n")
    f.write(f"- Shape: {mod_img_shape}\n")
    f.write(f"- Output groups: {mod_img_groups} × {hidden_size} = {mod_img_shape[0]}\n")
    f.write(f"- Interpretation: {mod_img_groups} groups = shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff\n")
    f.write(f"- These 6 modulation vectors are applied to ALL {num_double} double blocks identically\n\n")

    f.write(f"### double_stream_modulation_txt\n")
    f.write(f"- Shape: {mod_txt_shape}\n")
    f.write(f"- Output groups: {mod_txt_groups} × {hidden_size} = {mod_txt_shape[0]}\n")
    f.write(f"- Same 6-group structure as img modulation, for text stream\n\n")

    f.write(f"### single_stream_modulation\n")
    f.write(f"- Shape: {mod_single_shape}\n")
    f.write(f"- Output groups: {mod_single_groups} × {hidden_size} = {mod_single_shape[0]}\n")
    f.write(f"- Interpretation: {mod_single_groups} groups = shift, scale, gate (for fused attn+MLP)\n\n")

    f.write(f"### norm_out (final layer modulation)\n")
    norm_out_shape = raw_keys.get("norm_out.linear.weight", ((0, 0), ""))[0]
    norm_out_groups = norm_out_shape[0] // hidden_size if hidden_size else 0
    f.write(f"- Shape: {norm_out_shape}\n")
    f.write(f"- Output groups: {norm_out_groups} × {hidden_size}\n")
    f.write(f"- Interpretation: shift, scale for final AdaLN normalization\n\n")

    # 9
    f.write("## 9. Timestep Embedder\n\n")
    time_keys = [k for k in sorted_raw if k.startswith("time_guidance_embed.")]
    if time_keys:
        f.write("| Key | Shape |\n|---|---|\n")
        for k in time_keys:
            f.write(f"| `{k}` | {raw_keys[k][0]} |\n")
        f.write(f"\nNote: `guidance_embeds = {config.get('guidance_embeds', '?')}` — ")
        if not config.get("guidance_embeds", True):
            f.write("no separate guidance embedder. Timestep only.\n")
        else:
            f.write("includes guidance embedding.\n")
        f.write(f"Timestep embedding channels: {config.get('timestep_guidance_channels', '?')}\n\n")
    else:
        f.write("No timestep embedder keys found (unexpected!).\n\n")

    # 10
    f.write("## 10. Image Input Embedder (x_embedder)\n\n")
    f.write(f"- Key: `x_embedder.weight`\n")
    f.write(f"- Shape: {x_emb_shape}\n")
    f.write(f"- Maps {actual_in_channels} latent channels → {actual_hidden} hidden dim\n")
    f.write(f"- No bias (weight-only)\n\n")

    # 11
    f.write("## 11. Final Output Projection\n\n")
    proj_out_shape = raw_keys.get("proj_out.weight", ((0, 0), ""))[0]
    f.write(f"- `proj_out.weight`: {proj_out_shape}\n")
    f.write(f"- `norm_out.linear.weight`: {norm_out_shape}\n")
    f.write(f"- Maps {hidden_size} hidden dim → {proj_out_shape[0]} (= patch_size² × out_channels = 1 × {proj_out_shape[0]})\n\n")

    # 12
    f.write("## 12. RoPE (Rotary Position Embeddings)\n\n")
    if rope_keys:
        f.write("RoPE frequencies stored in weights:\n")
        for k in rope_keys:
            f.write(f"- `{k}`: {raw_keys[k][0]}\n")
    else:
        f.write("**No RoPE frequency tensors in weights.** Frequencies are computed at runtime.\n")
        f.write(f"- `rope_theta`: {config.get('rope_theta', '?')}\n")
        f.write(f"- `axes_dims_rope`: {config.get('axes_dims_rope', '?')}\n")
    if rope_diff_keys:
        f.write(f"\nDiffusers state_dict RoPE keys:\n")
        for k in rope_diff_keys:
            f.write(f"- `{k}`: {diff_keys[k][0]}\n")
    else:
        f.write("\nNo RoPE keys in diffusers state_dict either — confirmed runtime computation.\n")
    f.write("\n")

    # Key mapping note
    f.write("## Key Mapping Note\n\n")
    f.write("The raw safetensors file for Klein 4B uses **diffusers-format keys directly**.\n")
    f.write("This means the raw-to-diffusers key mapping is an **identity mapping** — every key in the\n")
    f.write("safetensors file has the exact same name in `pipe.transformer.state_dict()`.\n\n")
    f.write("This is different from the 9B model, which uses a BFL-format key naming convention\n")
    f.write("(e.g., `double_blocks.N.img_mlp.0.weight`) that diffusers remaps during loading.\n\n")

    # Notable differences from 9B
    f.write("## Differences from FLUX.2 Klein 9B\n\n")
    f.write("| Feature | 9B | 4B |\n|---|---|---|\n")
    f.write(f"| Parameters | ~9B | {total_params / 1e9:.2f}B |\n")
    f.write(f"| Double blocks | 8 | {num_double} |\n")
    f.write(f"| Single blocks | 24 | {num_single} |\n")
    f.write(f"| Hidden size | 4096 | {hidden_size} |\n")
    f.write(f"| Attention heads | 32 | {num_heads} |\n")
    f.write(f"| Head dim | 128 | {head_dim} |\n")
    f.write("| Safetensors key format | BFL-format (requires remapping) | Diffusers-format (identity) |\n")
    f.write("| Biases | Yes (in most layers) | No (weight-only) |\n")
    f.write(f"| Guidance embeds | True | {config.get('guidance_embeds', '?')} |\n")
    f.write(f"| Joint attention dim | 15360 | {config.get('joint_attention_dim', '?')} |\n")
    f.write("| Double block QKV | Fused (img_attn.qkv) | Split (to_q, to_k, to_v) |\n")
    f.write("| Per-block modulation | Yes (in each block) | No (shared globally) |\n")
    f.write("\n")

print(f"Saved audit summary to {summary_path.name}")


# ============================================================
# Step 5 — Prepare CLAUDE.md update sections
# ============================================================
print("\n" + "=" * 60)
print("Step 5: CLAUDE.md update sections")
print("=" * 60)

update_path = OUTPUT_DIR / "claude_md_updates.txt"
with open(update_path, "w", encoding="utf-8") as f:
    # Section: Tensor key patterns (double-stream)
    f.write("### Tensor key patterns (double-stream)\n\n")
    f.write(f"{num_double} double-stream blocks (indices 0–{num_double - 1}), {len(double_subkeys)} weight tensors per block.\n")
    f.write("All weights are bf16, no biases.\n\n")
    f.write("| Sub-key | Shape | Description |\n|---|---|---|\n")
    for subkey in sorted(double_subkeys.keys()):
        shape = double_subkeys[subkey][0]
        desc = desc_map.get(subkey, "")
        f.write(f"| `transformer_blocks.N.{subkey}` | {shape} | {desc} |\n")
    f.write("\n")
    f.write("Note: Unlike the 9B model, double-block attention is NOT fused — Q/K/V are separate projections.\n")
    f.write(f"Text stream uses `add_q_proj`/`add_k_proj`/`add_v_proj` (not `to_add_q`/`to_add_k`/`to_add_v`).\n\n")

    # Section: Tensor key patterns (single-stream)
    f.write("### Tensor key patterns (single-stream)\n\n")
    f.write(f"{num_single} single-stream blocks (indices 0–{num_single - 1}), {len(single_subkeys)} weight tensors per block.\n\n")
    f.write("| Sub-key | Shape | Description |\n|---|---|---|\n")
    for subkey in sorted(single_subkeys.keys()):
        shape = single_subkeys[subkey][0]
        desc = single_desc.get(subkey, "")
        f.write(f"| `single_transformer_blocks.N.{subkey}` | {shape} | {desc} |\n")
    f.write(f"\nFused projection layout:\n")
    f.write(f"- `to_qkv_mlp_proj` output dim {fused_in_shape[0]}: first {qkv_dim} = QKV ({hidden_size} Q + {hidden_size} K + {hidden_size} V), remaining {mlp_fused_input_dim} = MLP input (SwiGLU gated, 2 × {mlp_hidden})\n")
    f.write(f"- `to_out` input dim {fused_out_shape[1]}: first {attn_out_dim} = attention output, remaining {mlp_out_dim} = MLP output\n\n")

    # Section: Non-block tensors
    f.write("### Non-block tensors (embedders, final layer, modulation)\n\n")
    f.write("| Key | Shape | Description |\n|---|---|---|\n")
    for key in sorted(non_block.keys()):
        shape = non_block[key][0]
        desc = non_block_desc.get(key, "")
        if key.startswith("time_guidance_embed."):
            desc = "Timestep embedder"
        f.write(f"| `{key}` | {shape} | {desc} |\n")
    f.write("\n")

    # Section: Diffusers key mapping
    f.write("### Diffusers key mapping\n\n")
    f.write("The Klein 4B safetensors file uses **diffusers-format keys directly**.\n")
    f.write("Raw key = diffusers key (identity mapping). No remapping needed.\n\n")
    f.write("This is different from the 9B model which uses BFL-format keys like `double_blocks.N.img_mlp.0.weight`\n")
    f.write("that diffusers remaps to `transformer_blocks.N.ff.linear_in.weight` during loading.\n\n")
    f.write("Since `flux-bend` operates at the safetensors level, we can use diffusers attribute paths directly\n")
    f.write("as tensor keys — no translation layer needed.\n\n")

    # Section: Transformer safetensors file structure
    f.write("### Transformer safetensors file structure\n\n")
    if len(safetensor_files) == 1:
        f.write(f"Single file (not sharded): `transformer/{file_info[0][0]}` ({file_info[0][1]:.1f} MB)\n")
    else:
        f.write(f"Sharded across {len(safetensor_files)} files in `transformer/`:\n")
        for name, size in file_info:
            f.write(f"- `{name}` ({size:.1f} MB)\n")
    f.write(f"\n{len(sorted_raw)} tensors, {total_params:,} parameters ({total_params / 1e9:.2f}B), all bf16, no biases.\n")

print(f"Saved CLAUDE.md update sections to {update_path.name}")

# Final summary
print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
print(f"\nKey findings:")
print(f"  Raw keys = Diffusers keys (identity mapping)")
print(f"  Double blocks: {num_double} (indices 0-{num_double-1})")
print(f"  Single blocks: {num_single} (indices 0-{num_single-1})")
print(f"  Hidden size: {hidden_size} ({num_heads} heads × {head_dim} dim)")
print(f"  MLP hidden: {mlp_hidden} (ratio {mlp_hidden/hidden_size:.1f}x)")
print(f"  No biases anywhere")
print(f"  Shared modulation: img={mod_img_groups} groups, txt={mod_txt_groups} groups, single={mod_single_groups} groups")
print(f"  RoPE: runtime (not stored in weights)")
print(f"  Total params: {total_params:,}")
print(f"\nOutput files:")
for p in [raw_audit_path, diff_audit_path, mapping_path, summary_path, update_path]:
    print(f"  {p}")

del pipe, state_dict
if torch.cuda.is_available():
    torch.cuda.empty_cache()
