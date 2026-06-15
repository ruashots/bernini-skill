# bernini — setup (models, nodes, ComfyUI version)

Everything the engine needs on a fresh ComfyUI. The engine resolves model names by basename, so the
exact subfolder separators don't matter — just put the files where shown.

## 1. ComfyUI version — the gate

`BerniniConditioning` is a **native core node** (`comfy_extras/nodes_bernini.py`), added on
**2026-06-09** (PR #14216, "Add Bernini-R model support"). It is **not** in ComfyUI v0.24.1 or earlier.

- **Update ComfyUI to master/nightly ≥ 2026-06-09.** After updating, confirm the node exists:
  ```bash
  curl -s http://127.0.0.1:8188/object_info/BerniniConditioning | head -c 80
  ```
  Empty/`{}` → still outdated. (The same update also ships native SCAIL nodes, FYI.)
- No `task_type` dropdown exists on the core node — the task is inferred from which inputs are wired.
  This engine drives that node directly and supplies the per-task system prompt itself.

## 2. Models — exact files & folders

All under `ComfyUI/models/`. Sizes are the fp8 build (the right pick for a 24–32 GB card; Blackwell does
fp8 natively). Mirrors: `Kijai/WanVideo_comfy_fp8_scaled/Bernini` and `Kijai/WanVideo_comfy` carry the
same weights if you prefer Kijai's packaging.

| File | Folder | From |
|---|---|---|
| `wan2.2_bernini_r_high_noise_fp8_scaled.safetensors` (15.6 GB) | `diffusion_models/Bernini/` | [`Comfy-Org/Bernini-R`](https://huggingface.co/Comfy-Org/Bernini-R) → `diffusion_models/` |
| `wan2.2_bernini_r_low_noise_fp8_scaled.safetensors` (15.6 GB) | `diffusion_models/Bernini/` | same |
| `lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors` | `loras/WanVideo/` | [`Kijai/WanVideo_comfy`](https://huggingface.co/Kijai/WanVideo_comfy) → `Lightx2v/` |
| `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors` | `loras/WanVideo/` | same |
| `umt5_xxl_fp16.safetensors` | `text_encoders/` | Wan-2.2 text encoder (Comfy-Org Wan repack; fp8 variant also fine) |
| `wan_2.1_vae.safetensors` | `vae/` | Wan VAE (Comfy-Org Wan repack) |

The two LoRAs are the **standard Wan-2.2 lightx2v step-distillation** adapters (not Bernini-specific):
the engine puts **I2V @ 3.0 on the high-noise expert** and **T2V-v2 @ 1.5 on the low-noise expert** —
the exact pairing from the official Bernini example workflow.

Higher quality / more VRAM: the **fp16** pair (28.6 GB each) exists in the same repo; **mxfp8** (15.0 GB)
is a marginal shrink. There's also a **1.3B** single-expert build (`wan2.1_bernini_1.3B_fp16`, 2.8 GB) —
faster, weaker on humans/complex scenes.

Download example (HF CLI or curl):
```bash
cd ComfyUI/models
curl -L -o diffusion_models/Bernini/wan2.2_bernini_r_high_noise_fp8_scaled.safetensors \
  https://huggingface.co/Comfy-Org/Bernini-R/resolve/main/diffusion_models/wan2.2_bernini_r_high_noise_fp8_scaled.safetensors
curl -L -o diffusion_models/Bernini/wan2.2_bernini_r_low_noise_fp8_scaled.safetensors \
  https://huggingface.co/Comfy-Org/Bernini-R/resolve/main/diffusion_models/wan2.2_bernini_r_low_noise_fp8_scaled.safetensors
curl -L -o loras/WanVideo/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors
curl -L -o loras/WanVideo/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors \
  https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors
```

## 3. Support node packs

- **ComfyUI-KJNodes** — provides `PathchSageAttentionKJ` (the engine wraps each expert in SageAttention).
  Install via ComfyUI-Manager. SageAttention itself must be importable in the ComfyUI Python env
  (`pip install sageattention`); if it isn't, swap `PathchSageAttentionKJ` for a no-op or install it.
- **ComfyUI-VideoHelperSuite (VHS)** — `VHS_LoadVideo` (source/content video in) and `VHS_VideoCombine`
  (video out). Install via Manager.
- Core nodes (`UNETLoader`, `LoraLoaderModelOnly`, `CLIPLoader`, `VAELoader`, `BasicScheduler`,
  `SplitSigmas`, `KSamplerSelect`, `SamplerCustom`, `VAEDecode`, `SaveImage`, `BerniniConditioning`)
  ship with ComfyUI.

Verify everything at once:
```bash
curl -s http://127.0.0.1:8188/object_info | python3 -c "import json,sys; d=json.load(sys.stdin); \
print([n for n in ('BerniniConditioning','PathchSageAttentionKJ','VHS_LoadVideo','VHS_VideoCombine') if n not in d] or 'all present')"
```

## 4. VRAM

Bernini-R is a 14B dual-expert model, but Wan-2.2 keeps **one expert resident at a time** (the high-noise
expert runs first, then the low-noise expert), so the working set is ~one 15.6 GB fp8 transformer + VAE +
T5 + latents. The engine's **default is 720p** (`--max-size 1280`) — budget **~24–32 GB**; a 480p draft
(`--max-size 832`) needs **~16–24 GB** and is ~3× faster. More as you raise resolution/frames. 720p and long clips
(121/145 frames) want a 24–32 GB card. Free other large models from VRAM before a run. OOM → lower
`--max-size`/`--width`/`--height`, lower `--frames`, or use the mxfp8 / 1.3B build.

## 5. Engine env overrides

| Env | Default | Purpose |
|---|---|---|
| `COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI API endpoint |
| `COMFY_DIR` | auto-detected | ComfyUI install path (set this if auto-detect picks the wrong install) |
| `DELIVER_DIR` | `~/Downloads` | where the finished file is copied |

Auto-detect asks the running server for its `main.py` path; for the WSL→Windows case (relative argv) it
searches mounted drives and confirms **which** install the server actually serves via a `/view` probe.
On a machine with several ComfyUI installs, set `COMFY_DIR` to be explicit.
