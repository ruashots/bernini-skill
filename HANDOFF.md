# bernini — agent handoff

Agent-agnostic guide. Any harness that can run shell commands and read files can drive the engine.
Map tool names to your harness: "run a shell command" = Bash/exec/shell; "read a file" = Read/cat.
`SKILL.md` is the craft (read it first); this is the harness-neutral contract.

## The contract

1. **Read `SKILL.md`** — the load-bearing rules:
   - **Pick the task with `--task`** (one of the 13 in the table). The task selects which inputs the
     model uses *and* the official system prompt that steers it — you don't write the system prompt.
   - **Three-part prompt:** the task prefix is automatic; you supply the **edit goal** (a strong verb:
     replace/add/change/transform) + a plain description of the **final result**.
   - **One edit per run.** Don't stack changes — separate passes.
   - **References: 1–2 images.** Past ~2, identity separation and quality drop. Refs are positional —
     refer to them as `image0`, `image1`, … in the prompt.
   - **Don't hand-set sizing.** For edit modes, resolution/length/fps derive from the source; for
     generation, sensible defaults apply (override with `--frames`/`--width`/`--height` if needed).
   - **Frame count is 8n+1** (snapped for you).
2. **Prepare inputs.** Match the task: `--source` (image or video to edit/animate), `--ref` (reference
   images), `--content` (ads2v insert). Absolute paths are staged automatically.
3. **Run** the engine (below).
4. **Verify before claiming success** — sample frames (or use the `watch-video` skill). For an *edit*,
   confirm the target changed while motion/pose/camera/background were **preserved**; for *generation*,
   confirm the reference identity carried and motion is coherent. The engine does not judge the media.

## Run it

```bash
# reference-guided video edit (the headline)
python3 scripts/bernini.py --task rv2v --source CLIP.mp4 --ref PERSON.png \
    -p "Replace the dancer with the woman from image0; same motion and studio. Preserve pose and camera."

# subject -> video / image
python3 scripts/bernini.py --task r2v --ref A.png --ref B.png -p "<scene combining the subjects>"
python3 scripts/bernini.py --task r2i --ref A.png            -p "<scene>"        # image out

# text-driven video edit / focused variants
python3 scripts/bernini.py --task v2v   --source CLIP.mp4 -p "<change by instruction>"
python3 scripts/bernini.py --task mv2v  --source CLIP.mp4 -p "<adjust style/lighting/color/texture>"
python3 scripts/bernini.py --task vrc2v --source CLIP.mp4 -p "<adjust subject action/position>"

# generate / animate / edit one image / insert content
python3 scripts/bernini.py --task t2v -p "<scene>"
python3 scripts/bernini.py --task i2v --source IMG.png -p "<motion>"
python3 scripts/bernini.py --task i2i --source IMG.png -p "<edit instruction>"        # image out
python3 scripts/bernini.py --task ads2v --source CLIP.mp4 --content LOGO.png -p "<where to insert>"
```

- Result lands in `ComfyUI/output/Bernini/<name>.(mp4|png)` (printed as `FINAL=`), copied to the delivery
  dir unless `--no-deliver`.
- Useful flags: `--frames N` (8n+1), `--width/--height`, `--max-size` (source long-edge cap), `--steps N`,
  `--seed N` (`-1`=random), `--name PREFIX`, `--dump-graph FILE` (inspect the assembled API graph).
- The engine **auto-detects the ComfyUI install** from the running server. Override via env: `COMFY_URL`,
  `COMFY_DIR`, `DELIVER_DIR`. Exit is non-zero on connection/node/model/validation errors; the message
  points at `references/setup.md`.

## Requirements

A running ComfyUI (default `http://127.0.0.1:8188`) with the Bernini-R models + native `BerniniConditioning`
node (ComfyUI ≥ 2026-06-09) + KJNodes + VHS — exact files/folders in `references/setup.md`.
`ffmpeg`/`ffprobe` on PATH. Python 3 stdlib only. Budget ~16–24 GB VRAM at 480p fp8; free other large
models first.

## Boundaries

- The engine submits a render and delivers the file; it does **not** judge the output. Always verify.
- **One edit per run** — it will not enforce this; a multi-change prompt usually fails. Split it.
- **Style transfer of a whole scene** ("make it anime") is the model's weak task — subject/outfit/background
  swaps and additions are where it shines. Use `mv2v` for *partial* style/lighting adjustments.
- More than ~2 reference images degrades identity separation (model limitation, not an engine bug).
