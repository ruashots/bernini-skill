# bernini

Drive **ByteDance Bernini-R** — a unified video + image generation and editing model — on a local ComfyUI, from one small Python CLI. No node-graph wrangling: pick a task, pass your inputs, get a result.

Bernini-R is built on Wan-2.2 and does, in *one* model, what used to need a different tool per job:

- **Reference-guided video editing** — swap the character, outfit, or background in a clip using a reference image, keeping the original motion and camera (`rv2v`).
- **Text-driven video editing** — change a video by instruction, no reference (`v2v`, and the focused variants `vrc2v`/`mv2v`/`vi2v`).
- **Subject → video / image** — generate from one or more reference images (`r2v`, `r2i`).
- **Image → video** — animate a still (`i2v`).
- **Text → video / image** (`t2v`, `t2i`) and **single-image editing** (`i2i`).
- **Content insertion** — composite an image/video into a scene (`ads2v`).

## Install

You need a running **ComfyUI (≥ 2026-06-09, when the native `BerniniConditioning` node landed)** with the Bernini-R models and a few standard Wan-2.2 support nodes. Exact model files, target folders, and node packs are in **[references/setup.md](references/setup.md)**.

The CLI itself is **Python 3 standard library only** — nothing to pip install. It needs `ffmpeg`/`ffprobe` on PATH.

## Use

```bash
python3 scripts/bernini.py --task <type> [--source FILE] [--ref IMG ...] [--content FILE] -p "..." [opts]
```

```bash
# swap the person in a clip with a reference (keeps motion + scene)
python3 scripts/bernini.py --task rv2v --source dance.mp4 --ref person.png \
  -p "Replace the dancer with the woman from image0; same motion, same studio. Preserve pose and camera."

# subject(s) -> video (1-2 references work best)
python3 scripts/bernini.py --task r2v --ref hero.png -p "the man from image0 walking through a neon city at night"

# edit a video by text, no reference
python3 scripts/bernini.py --task v2v --source dance.mp4 -p "change her red outfit into a flowing white dress"

# edit / animate / generate
python3 scripts/bernini.py --task i2i --source photo.png -p "add red sunglasses to her face"
python3 scripts/bernini.py --task i2v --source fox.png  -p "the fox turns its head to camera"
python3 scripts/bernini.py --task t2v -p "a red fox running across a snowy field, cinematic"
```

Inputs can be absolute paths (auto-staged into ComfyUI) or filenames already in `input/`. Output lands in `ComfyUI/output/Bernini/` and is copied to your delivery dir (default `~/Downloads`) unless `--no-deliver`.

See **[SKILL.md](SKILL.md)** for the full task table, options, and the prompt rules that decide quality; **[references/reference.md](references/reference.md)** for internals and troubleshooting; **[HANDOFF.md](HANDOFF.md)** for driving the engine from any agent harness.

## Prompting in one breath

Bernini prompts are short and structured: **edit goal** (a strong verb — `replace`, `add`, `change`, `transform`) + **the final result** described plainly. The per-task system prompt is added for you. **One edit per run.** Reference images: **1–2 is the sweet spot** (quality drops past 2). Video length is **8n+1** frames (81, 121, 145…).

## Notes

- The engine auto-detects your ComfyUI install from the running server (including WSL → Windows, and machines with multiple installs). Override with `COMFY_URL`, `COMFY_DIR`, `DELIVER_DIR`.
- Two Wan-2.2 experts run in sequence (high-noise → low-noise) with lightx2v distillation at 6 steps. **Resolution is the quality lever** — the default is 720p (`--max-size 1280`); use `--max-size 832` for a fast 480p draft. Budget ~24–32 GB VRAM at 720p, ~16–24 GB at 480p (fp8).
- **License:** Bernini-R weights are **Apache-2.0** — commercial use is OK.

Not affiliated with ByteDance. This is a thin local driver around the open-source model + ComfyUI's native node.
