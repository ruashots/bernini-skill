---
name: bernini
description: Generate or edit video and images with ByteDance Bernini-R (unified, local, ComfyUI). Use for reference-guided video editing (swap a character/outfit/background in a clip using a reference image), text-driven video editing, subject-to-video from one or more reference images, image-to-video, text-to-video, and single-image editing — one model covers all of it. One `--task` selects the mode; the engine wires the right inputs and prepends Bernini's official per-task system prompt. Drives a local ComfyUI with the Bernini-R stack installed.
---

# bernini — unified video/image generation & editing (Bernini-R)

ByteDance **Bernini-R** is one Wan-2.2-based model that does generation *and* in-context editing for both video and images. You pick the task with `--task`; the engine wires the inputs that task needs, prepends the task's official system prompt to your text, and runs the exact two-expert Bernini graph.

```bash
python3 scripts/bernini.py --task <type> [--source FILE] [--ref IMG ...] [--content FILE] -p "..." [opts]
```

## The tasks

12 runnable tasks below (plus a `default` generic system-prompt fallback — the official model defines 13
task system prompts in total).

| `--task` | does | you give it | output |
|---|---|---|---|
| `t2v`  | text → video | `-p` | video |
| `t2i`  | text → image | `-p` | image |
| `i2v`  | image → video (animate a still) | `--source IMG` `-p` | video |
| `i2i`  | edit an image by instruction | `--source IMG` `-p` | image |
| `v2v`  | edit a video by instruction (no reference) | `--source VID` `-p` | video |
| `r2v`  | subject(s) → video | `--ref IMG …` (1–8) `-p` | video |
| `r2i`  | subject(s) → image | `--ref IMG …` (1–8) `-p` | image |
| `rv2v` | **reference-guided video edit** (swap character/outfit/background) | `--source VID` + `--ref IMG …` `-p` | video |
| `vi2v` | video edit, content propagation | `--source VID` [`--ref IMG`] `-p` | video |
| `vrc2v`| video edit, adjust subject action/position | `--source VID` `-p` | video |
| `mv2v` | video edit, adjust style/lighting/color/texture + pose | `--source VID` `-p` | video |
| `ads2v`| insert image/video content into a scene | `--source VID` + `--content IMG|VID` `-p` | video |

`rv2v` is the headline — a source video gives the motion + scene, reference images give the identity/outfit/object; the model re-renders the clip with the swap while preserving pose, motion, and camera.

## Run it

```bash
# reference-guided video edit: swap the person in a clip with the one in a reference image
python3 scripts/bernini.py --task rv2v --source dance.mp4 --ref person.png \
  -p "Replace the dancer with the woman from image0; same motion, same studio, red outfit. Preserve pose, motion and camera."

# subject(s) -> video from reference images (≤2 for best identity)
python3 scripts/bernini.py --task r2v --ref witch.png --ref tiger.png \
  -p "a woman holding a staff riding a white tiger through a misty forest" --frames 81

# edit one image
python3 scripts/bernini.py --task i2i --source photo.png -p "add red sunglasses to her face; keep everything else"

# animate a still / pure generation
python3 scripts/bernini.py --task i2v --source fox.png -p "the fox turns its head to camera, gentle push-in"
python3 scripts/bernini.py --task t2v -p "a red fox running across a snowy field, cinematic"
```

Inputs may be absolute paths (staged into ComfyUI's `input/` automatically) or filenames already there. The result lands in `ComfyUI/output/Bernini/<name>.(mp4|png)` (printed as `FINAL=`) and is copied to the delivery dir unless `--no-deliver`.

### Options

| Flag | Default | What it does |
|---|---|---|
| `-p, --prompt` | — | the edit instruction / scene description (see the 3-part rule below) |
| `--ref IMG` | — | reference image (repeat for up to 8; **quality degrades past ~2**) |
| `--source FILE` | — | source image or video to edit/animate (auto-detected by extension) |
| `--content FILE` | — | image/video to insert (`ads2v` only) |
| `--negative` | standard Wan-2.2 neg | the official `bytedance/Bernini` default negative (a full quality/anatomy suppressor — not the example's thin `"bad video"`) |
| `--frames N` | `81` | video length; **snapped to 8n+1** (81, 121, 145…) |
| `--img-frames N` | `9` | **t2i/r2i** render N frames (8n+1) and keep the **middle** one — a lone frame is out-of-distribution for this video model and comes out waxy; `1` = fast but low-quality |
| `--width/--height` | from source / 1280×720 video / 1024² image | override the generation rectangle |
| `--max-size N` | `1280` | long-edge cap when deriving size from a source — **720p, the production default**; use **`832` for a ~3× faster 480p draft** (or on a <24 GB card) |
| `--steps N` | `6` | total sampler steps (hi-noise 0→split, lo-noise split→steps) |
| `--seed N` | `42` | `-1` = random |
| `--name`, `--no-deliver`, `--dump-graph` | | output prefix · skip the copy · write the assembled graph (debug) |

## The rules that decide quality — read before running

1. **Three-part prompt.** Bernini prompts are short and structured: **(1)** the task is set for you (the system prompt is auto-prepended per `--task`); **(2)** the *edit goal* — a strong verb: `replace`, `add`, `change`, `transform`; **(3)** a clear description of the *final result*. e.g. *"Replace the woman's shirt with the shirt from image0. Keep the undershirt, body pose, and camera unchanged."*
2. **One edit per run.** Change *one* thing. Stacking edits ("swap the shirt AND change the background AND…") usually fails — run them as separate passes.
3. **Reference images: up to 8, but the more you add the lower the per-subject fidelity.** 1–2 is the sweet spot. Subjects stay **distinct** even at 3+ (verified — no blending), so going higher is your call; what you trade away past ~2 is per-subject detail/fidelity, not identification. Name them positionally in the prompt — `image0`, `image1`, … (8 is the node's hard ceiling).
4. **Frame count = 8n+1** (81 default, 121, 145…). The engine snaps for you; longer = more VRAM/time.
5. **Edit modes follow the source.** For `v2v`/`rv2v`/etc., output resolution comes from the source aspect (long edge `--max-size`), length defaults to the first 81 source frames (raise `--frames` to cover more), and fps follows the source. `BerniniConditioning` resizes/trims the source internally.
6. **Resolution is the quality lever — not steps.** The lightx2v distill LoRAs are tuned for ~6 steps; 8/10 barely change anything. Sharpness comes from **resolution**: the default `--max-size 1280` (720p) is dramatically crisper than a 480p draft (faces, hair, fine texture). Iterate at `--max-size 832` (fast), finalize at the default. `--steps 8` is a marginal optional bump.
7. **Style transfer is the weak task.** Whole-scene "make it anime" restyles are hit-or-miss; subject/outfit/background swaps and additions are where Bernini shines. Use `mv2v` for *partial* style/lighting/color adjustments.

## How it works (so results are reproducible)

Two Wan-2.2 experts run in sequence on a 6-step "simple" schedule split in half: the **high-noise** expert (Bernini-R high fp8 + lightx2v I2V LoRA @3.0) lays out coarse structure (steps 0→3), then hands off to the **low-noise** expert (Bernini-R low fp8 + lightx2v T2V-v2 LoRA @1.5) for detail (steps 3→6) — the Wan-2.2 MoE switch. `BerniniConditioning` VAE-encodes the source video / reference images as in-context streams (`source_video` id1, `reference_video` id2, then each reference image), and the model keeps them distinct via segment-aware 3D RoPE. cfg=1 (distilled), sampler `res_multistep`. The task is what you select; the inputs you pass determine the wiring.

## Verify a result

Sample frames (or use the `watch-video` skill) and confirm against intent: for edits, the *target* changed while motion/pose/camera/background were **preserved**; for generation, the subject identity matches the reference and motion is coherent. The engine submits and delivers — it does not judge the media. Always look.

## Requirements & setup

A running ComfyUI (default `http://127.0.0.1:8188`) with the Bernini-R models + native `BerniniConditioning` node (ComfyUI **≥ 2026-06-09**) + KJNodes (SageAttention) + VHS. Fresh setup, exact model files/folders, and the ComfyUI-version note → **`references/setup.md`**. Full parameter/internals/troubleshooting → **`references/reference.md`**.

Bernini-R is a 14B dual-expert model (one expert resident at a time): budget ~24–32 GB at the **720p default**, ~16–24 GB for a 480p draft (`--max-size 832`); free other large models first. The engine **auto-detects the ComfyUI install** from the running server (handles WSL→Windows and multiple installs); override with env: `COMFY_URL`, `COMFY_DIR`, `DELIVER_DIR`. Bernini-R weights are **Apache-2.0 (commercial use OK)**.
