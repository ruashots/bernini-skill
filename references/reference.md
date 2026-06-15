# bernini — reference (internals, parameters, troubleshooting)

## Pipeline

Bernini-R is a Wan-2.2 **dual-expert (MoE)** model conditioned **in-context** — the source video and
reference images are VAE-encoded and attached to the conditioning as extra latent streams, and the model
keeps them distinct with **segment-aware 3D RoPE** (SA3D-RoPE). There are no masks, ControlNets, or
pose/depth maps — editing is maskless and instruction-driven.

The engine builds this graph (programmatically, per task):

1. **Text** — `CLIPLoader(umt5_xxl, type=wan)` → two `CLIPTextEncode` (positive = the task system prompt +
   your prompt; negative = the standard Wan-2.2 negative prompt, the official CLI default).
2. **Conditioning** — `BerniniConditioning` takes positive/negative, the VAE, `width/height/length`, and
   the optional `source_video` / `reference_video` / `reference_images` (autogrow 0–8). It VAE-encodes each
   connected stream (order: source_video → reference_video → each reference image), resizes the source to
   `width×height` and **trims it to `length`**, and emits an empty latent of the target size. Reference
   images/video keep their **native aspect** (long edge ≤ `ref_max_size`, default 848).
3. **Two experts in sequence** on a 6-step `simple` schedule split at 3:
   - **high-noise**: `UNETLoader(bernini high fp8)` → `PathchSageAttentionKJ` → `LoraLoaderModelOnly(lightx2v
     I2V rank256 @ 3.0)` → `SamplerCustom(add_noise=True, cfg=1, res_multistep, sigmas[0:split])` on the empty latent.
   - **low-noise**: `UNETLoader(bernini low fp8)` → Sage → `LoraLoaderModelOnly(lightx2v T2V-v2 rank256 @ 1.5)`
     → `SamplerCustom(add_noise=False, cfg=1, res_multistep, sigmas[split:])` continuing the high-noise latent.
   The high-noise expert lays out coarse structure/composition; the low-noise expert renders detail/texture.
   (Wan-2.2's architectural switch is at ~87.5 % noise; the schedule is shaped by the distill LoRAs.)
4. **Out** — `VAEDecode` → `VHS_VideoCombine` (h264-mp4, crf 19) for video, or `SaveImage` for images
   (`length == 1`). `BasicScheduler`'s sigmas are computed from the raw low-noise UNET.

The 6 steps + cfg 1 come from the lightx2v **step-distillation** LoRAs — this is why it's fast and why
high CFG / many steps don't behave like a normal model (see below).

## Tasks — what gets wired

| task | source_video | reference_video | reference_images | length | system prompt (prepended, verbatim) |
|---|---|---|---|---|---|
| t2v | – | – | – | 81 | text-to-video generation |
| t2i | – | – | – | 1 | text-to-image generation |
| i2v | image | – | – | 81 | image-to-video generation |
| i2i | image | – | – | 1 | image editing |
| v2v | video | – | – | src | video editing |
| r2v | – | – | 1–8 | 81 | subject-to-video generation |
| r2i | – | – | 1–8 | 1 | subject-to-image generation |
| rv2v | video | – | 1–8 | src | video editing **with reference** |
| vi2v | video | – | 0–8 | src | video editing **on content propagation** |
| vrc2v | video | – | – | src | editing; adjust subject action/position |
| mv2v | video | – | – | src | editing; adjust style/lighting/colors/textures + pose |
| ads2v | video | image/video | – | src | ads insertion |

The model is *one* network; the task is selected by **(which inputs are connected) × (which system
prompt steers it)**. `t2v/v2v/rv2v/r2v/ads2v` are the input wirings the core node infers; the others are
the same wirings with a more specific system prompt (e.g. `mv2v` = a v2v wiring told to adjust
style/lighting). The full 13-string set is from the official `prompt_enhancer.py`.

## Parameters in depth

- **`--prompt` (the result, not a chat).** Three parts: task prefix (automatic) + edit goal
  (`replace`/`add`/`change`/`transform`) + a plain description of the finished frame. Keep it tight; Bernini
  doesn't need flowery prose. For multi-reference, name subjects positionally — `image0`, `image1`, …
- **`--ref` (1–8, sweet spot 1–2).** Each becomes its own in-context stream at native aspect. Beyond ~2,
  identity separation and overall quality drop — a documented model limitation, not an engine bug.
- **`--frames` (8n+1).** 81 (~default), 121, 145. Snapped automatically. For edit modes, defaults to the
  first 81 source frames; raise to cover a longer clip (costs VRAM/time). Image tasks force 1.
- **`--width/--height` / `--max-size`.** Generation rectangle. Edit modes derive it from the source aspect
  capped at `--max-size` (**1280 long edge by default ≈ 720p**, the production setting; drop to 832 for a
  ~3× faster 480p draft or a <24 GB card). Generation modes default to 1280×720 (video) / 1024² (image).
- **`--steps` / `--split`.** Total steps and the high→low handoff. Default 6 / 3. The distill LoRAs are
  tuned for few steps; a little more (8) can firm up detail, but large step counts don't help (and cfg is
  fixed at 1). `--split` shifts how much the high-noise expert does (more = more structural change).
- **`--seed`.** Default 42; `-1` randomizes.

## Quality rules (and why)

- **One edit per run.** The model plans the whole frame semantically before rendering; asking for several
  unrelated changes at once muddies that plan. Chain separate passes instead.
- **Preserve-language matters.** For edits, explicitly say what to keep ("preserve body pose, camera,
  background") — it measurably improves motion/identity retention.
- **Style transfer is the weak task.** Whole-scene restyle ("make it anime") is hit-or-miss and often only
  the subject changes. Subject/outfit/background **swaps and additions** are the strong suit. Use `mv2v`
  for *partial* style/lighting/color/texture adjustments rather than a total restyle.
- **Reference framing helps but isn't strict.** Bernini transfers identity from a reference more flexibly
  than pose-transfer models (a headshot reference can drive a full-body edit), but a reference roughly
  matching the target framing still gives the cleanest result.
- **Resolution is the biggest sharpness lever** (bigger than steps). The default `--max-size 1280` (720p)
  is dramatically crisper than a 480p draft — faces, hair, and fine texture especially. 480p
  (`--max-size 832`) is the model's native comfort zone and ~3× faster; use it to iterate, then finalize
  at the default. Verified directly: at 480p, steps 6/8/10 look the same; 720p is the real jump.
- **Single-frame image gen (t2i/r2i) needs a short clip, not one frame.** Bernini is a video model; a lone
  frame (`length=1`) is out-of-distribution and renders waxy/over-sharpened (plastic faces, crunchy hair
  halos). Verified with a controlled test (same prompt/seed/res): `length=1` is plastic, a frame pulled
  from a `length=9+` clip is a sharp natural photo. The engine therefore renders t2i/r2i as a short clip
  (`--img-frames`, default 9) and keeps the middle frame. `i2i` (editing) stays `length=1` — it inherits
  the source image's quality. For heavy *image* generation, a dedicated image model still wins; Bernini's
  strengths are video + reference-guided editing.
- **Multi-reference works; >2 degrades fidelity, not separation.** Verified: 2 refs (woman + fox) and even
  3 refs (woman + man + fox) each bound to a *distinct* subject with no blending — the model separates them
  fine. What drops past ~2 is per-subject *fidelity/detail*, not subject identification. Name refs
  positionally (`image0`, `image1`, …) and keep them visually distinct.
- **Don't go past 720p for batch work.** 1080p (`--max-size 1920`) is a *marginal* per-frame gain over
  720p and the time cost is **super-linear**, not linear: measured on a 5090 (portrait rv2v), 720p·81f =
  ~5.5 min, 1080p·**25f** = ~3 min, but 1080p·**81f** did not finish in 60 min (VRAM pinned ~32 GB, no
  OOM — it's offload pressure below the ceiling, not compute). Resolution is also **orthogonal to motion**:
  the temporal latent structure is identical at any resolution, so higher res buys spatial detail per
  frame, not smoother transitions. Reserve 1080p for short one-off hero shots; keep batches at 720p/480p.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ComfyUI not reachable` | Start ComfyUI; check `COMFY_URL`. |
| `missing nodes: BerniniConditioning …` | ComfyUI older than 2026-06-09, or a pack missing → `setup.md`. |
| `model(s) not found in ComfyUI: …` | The named file isn't in the listed folder → `setup.md` (names are matched by basename, so subfolder/OS separators don't matter). |
| `Invalid image file` / `Invalid file path` (LoadImage/VHS) | Auto-detect resolved the wrong ComfyUI install (multiple installs) → set `COMFY_DIR` to the running one. |
| Edit barely changes the video | Strengthen the edit verb; describe the target explicitly; try a reference (`rv2v`) instead of pure `v2v`; nudge `--split` up. |
| Edit changes too much / loses motion | Add preserve-language ("keep pose, motion, camera, background"); lower `--split`. |
| Soft / low detail | Raise `--max-size` (720p) and/or `--steps 8`; confirm you're not at a tiny resolution. |
| Multi-ref: subjects blend / wrong identity | Use ≤2 references; name them `image0`/`image1` in order; pick distinct, clearly-framed refs. |
| Output shorter than the clip | `--frames` defaults to the first 81 source frames; raise it (8n+1) to cover more. |
| OOM | Lower `--max-size`/`--frames`; free other models; use mxfp8 or the 1.3B build. |

## Provenance

Built and verified against the official Comfy-Org/Bernini-R example workflow and `bytedance/Bernini`
(`prompt_enhancer.py` system prompts copied verbatim; `comfy_extras/nodes_bernini.py` for the node
contract). Model + node mapping confirmed end-to-end on a local ComfyUI.
