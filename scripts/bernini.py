#!/usr/bin/env python3
"""
bernini — ByteDance Bernini-R (unified video/image generation + editing), driven by a local ComfyUI.

One model, one engine, every task. Pick the task with --task; the engine wires the inputs the
model expects for it, prepends the task's official system prompt to your text, and builds the
exact two-expert Bernini graph (high-noise + low-noise Wan-2.2 renderers, lightx2v distill, 6 steps).

  python3 bernini.py --task rv2v  --source clip.mp4 --ref shirt.png -p "replace the shirt with the reference shirt; keep everything else"
  python3 bernini.py --task r2v   --ref witch.png --ref tiger.png  -p "a woman riding a white tiger through a forest" --frames 81
  python3 bernini.py --task i2i   --source photo.png              -p "add red sunglasses to her face"
  python3 bernini.py --task t2v                                   -p "a fox running across a snowy field, cinematic"

The 13 task types (system prompt auto-prepended; inputs the engine wires):
  t2v  text->video            (no inputs)                 t2i  text->image            (no inputs, length=1)
  v2v  video edit by text     (--source video)            i2i  image edit by text     (--source image, length=1)
  i2v  image->video           (--source image)            r2i  subject->image         (--ref ..., length=1)
  r2v  subject(s)->video      (--ref ... up to 8)         rv2v reference video edit   (--source video + --ref ...)
  vi2v video edit, content propagation (--source video [+--ref])   vrc2v edit subject action/position (--source video)
  mv2v edit style/light/color/texture+pose (--source video)        ads2v insert content (--source video + --content img|vid)

Requires a running ComfyUI with the Bernini-R stack (see ../references/setup.md). Env overrides:
  COMFY_URL (default http://127.0.0.1:8188)   COMFY_DIR (ComfyUI install)   DELIVER_DIR (where output is copied)
"""
import argparse, json, os, re, shutil, subprocess, sys, time, urllib.request, urllib.error, urllib.parse

API     = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
DELIVER = os.environ.get("DELIVER_DIR", os.path.expanduser("~/Downloads"))

# ---- exact model files (verbatim from the official Comfy-Org/Bernini-R example workflow) ----
UNET_HI = r"Bernini\wan2.2_bernini_r_high_noise_fp8_scaled.safetensors"
UNET_LO = r"Bernini\wan2.2_bernini_r_low_noise_fp8_scaled.safetensors"
LORA_HI = r"WanVideo\lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"  # high-noise expert @ 3.0
LORA_LO = r"WanVideo\lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors"  # low-noise expert @ 1.5
LORA_HI_STR = 3.0
LORA_LO_STR = 1.5
CLIP    = "umt5_xxl_fp16.safetensors"
VAE     = "wan_2.1_vae.safetensors"
# The standard Wan-2.2 negative prompt — the official bytedance/Bernini CLI default
# (DEFAULT_NEG_PROMPT in cli.py). The Comfy example workflow's "bad video" is just a placeholder.
NEG_DEFAULT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# ---- the 13 task system prompts, verbatim from bytedance/Bernini prompt_enhancer.py SYSTEM_PROMPTS ----
SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i":  "You are a helpful assistant specialized in text-to-image generation.",
    "t2v":  "You are a helpful assistant specialized in text-to-video generation.",
    "i2i":  "You are a helpful assistant specialized in image editing.",
    "r2i":  "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v":  "You are a helpful assistant specialized in image-to-video generation.",
    "v2v":  "You are a helpful assistant specialized in video editing.",
    "r2v":  "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v":"You are a helpful assistant specialized in ads insertion.",
    "vrc2v":"You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
    "mv2v": "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, "
            "colors, textures, and the subject's pose or action.",
}

# task -> (output kind, source role, ref policy, needs --content)
#   source role: None | "video" | "image" | "auto" (auto = detect by extension)
#   ref policy : "no" | "opt" | "req"
TASKS = {
    "t2v":  ("video", None,    "no",  False),
    "t2i":  ("image", None,    "no",  False),
    "v2v":  ("video", "video", "no",  False),
    "i2v":  ("video", "image", "no",  False),
    "i2i":  ("image", "image", "no",  False),
    "r2v":  ("video", None,    "req", False),
    "r2i":  ("image", None,    "req", False),
    "rv2v": ("video", "auto",  "req", False),
    "vi2v": ("video", "video", "opt", False),
    "vrc2v":("video", "video", "no",  False),
    "mv2v": ("video", "video", "no",  False),
    "ads2v":("video", "video", "no",  True),
}

VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".gif", ".avi", ".m4v")

# ---------------------------------------------------------------- ComfyUI plumbing
def _valid_comfy(d):
    return bool(d) and os.path.isfile(os.path.join(d, "main.py")) \
        and os.path.isdir(os.path.join(d, "input")) and os.path.isdir(os.path.join(d, "output"))

def _server_serves_input(d):
    """True iff the RUNNING ComfyUI's input dir is d/input — disambiguates multiple installs
    by writing a probe into the candidate and asking the server to serve it back via /view."""
    probe = os.path.join(d, "input", ".bernini_probe")
    try:
        with open(probe, "w") as f: f.write("probe")
    except Exception:
        return False
    try:
        u = API + "/view?" + urllib.parse.urlencode({"filename": ".bernini_probe", "type": "input"})
        return urllib.request.urlopen(u, timeout=5).status == 200
    except Exception:
        return False
    finally:
        try: os.remove(probe)
        except Exception: pass

def detect_comfy_dir():
    """COMFY_DIR env wins; else ask the running ComfyUI for its own main.py path.
    Handles: absolute Windows path (WSL -> /mnt/<drive>), absolute POSIX path, and a
    RELATIVE argv (e.g. the Easy-Install's 'ComfyUI\\main.py') by searching mounted drives
    and confirming WHICH install the running server serves (multiple installs are common)."""
    if os.environ.get("COMFY_DIR"): return os.environ["COMFY_DIR"]
    main = ""
    try:
        main = json.load(urllib.request.urlopen(API + "/system_stats", timeout=5))["system"]["argv"][0]
    except Exception:
        main = ""
    m = re.match(r'^([A-Za-z]):[\\/](.+)[\\/][^\\/]+$', main)          # C:\dir\...\main.py
    if m:
        d = "/mnt/" + m.group(1).lower() + "/" + m.group(2).replace("\\", "/")
        if os.path.isdir(d): return d
        return f"{m.group(1)}:\\{m.group(2)}"
    if main.startswith("/") and os.path.dirname(main):                 # /home/.../ComfyUI/main.py
        return os.path.dirname(main)
    import glob                                                        # relative argv: find the install(s)
    cands = []
    for pat in ("/mnt/*/ComfyUI", "/mnt/*/*/ComfyUI", "/mnt/*/*/*/ComfyUI", "/mnt/*/*/*/*/ComfyUI"):
        cands += [c for c in glob.glob(pat) if _valid_comfy(c)]
    seen = list(dict.fromkeys(cands))
    if len(seen) == 1: return seen[0]
    for c in seen:                                                     # pick the one the server serves
        if _server_serves_input(c): return c
    return seen[0] if seen else os.path.expanduser("~/ComfyUI")

COMFY_DIR = detect_comfy_dir()
INP = os.path.join(COMFY_DIR, "input")
OUT = os.path.join(COMFY_DIR, "output")

def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
def probe(path):
    """(width, height, nb_frames, fps) via ffprobe; nb_frames/fps may be estimated."""
    w, h = (sh(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height",
                "-of","csv=p=0:s=x", path]) or "0x0").split("x")[:2]
    rate = sh(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate",
               "-of","csv=p=0", path]) or "16/1"
    try: fps = eval(rate) if "/" in rate else float(rate)
    except Exception: fps = 16.0
    nb = sh(["ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries",
             "stream=nb_read_frames","-of","csv=p=0", path])
    if not nb.isdigit():
        dur = sh(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0", path])
        nb = int(round(float(dur) * fps)) if dur.replace(".","",1).isdigit() else 81
    return int(w), int(h), int(nb), round(float(fps), 3)

def snap16(x): return max(16, int(round(x / 16)) * 16)
def snap_len(n):
    """Bernini wants frame counts of the form 8k+1 (1, 9, 17, ... 81, 121, 145)."""
    n = max(1, int(n))
    if n == 1: return 1
    k = round((n - 1) / 8)
    return max(1, 8 * k + 1)

def fit_rect(w, h, long_edge):
    """Preserve aspect; scale so the long edge == long_edge; snap each side to 16."""
    if w >= h: return snap16(long_edge), snap16(long_edge * h / w)
    return snap16(long_edge * w / h), snap16(long_edge)

def stage(arg):
    """Copy an absolute/relative path into ComfyUI/input; return the basename used in the graph."""
    if os.sep in arg or os.path.isabs(arg) or (":" in arg and os.path.exists(arg)):
        base = os.path.basename(arg); dst = os.path.join(INP, base)
        os.makedirs(INP, exist_ok=True)
        if os.path.abspath(arg) != os.path.abspath(dst):
            if not os.path.exists(arg): sys.exit(f"input not found: {arg}")
            shutil.copy(arg, dst)
        return base
    if not os.path.exists(os.path.join(INP, arg)): sys.exit(f"'{arg}' not found in {INP}")
    return arg

def is_video(path): return path.lower().endswith(VIDEO_EXT)

def preflight():
    try: urllib.request.urlopen(API + "/system_stats", timeout=5)
    except Exception: sys.exit(f"ComfyUI not reachable at {API}. Start it / check COMFY_URL (see references/setup.md).")
    # node availability
    try:
        oi = json.load(urllib.request.urlopen(API + "/object_info", timeout=15))
        miss = [n for n in ("BerniniConditioning","PathchSageAttentionKJ","VHS_LoadVideoPath","VHS_VideoCombine")
                if n not in oi]
        if miss: sys.exit("missing nodes: " + ", ".join(miss) + " — update ComfyUI / install packs (see references/setup.md).")
    except urllib.error.URLError: pass

def resolve_models():
    """Map our expected model basenames to the exact names ComfyUI lists — OS/separator and
    subfolder agnostic (Windows 'Bernini\\x' vs Linux 'Bernini/x'), and a clear error if missing."""
    want = {
        "unet_hi": ("UNETLoader", "unet_name", os.path.basename(UNET_HI.replace("\\", "/"))),
        "unet_lo": ("UNETLoader", "unet_name", os.path.basename(UNET_LO.replace("\\", "/"))),
        "lora_hi": ("LoraLoaderModelOnly", "lora_name", os.path.basename(LORA_HI.replace("\\", "/"))),
        "lora_lo": ("LoraLoaderModelOnly", "lora_name", os.path.basename(LORA_LO.replace("\\", "/"))),
        "clip":    ("CLIPLoader", "clip_name", CLIP),
        "vae":     ("VAELoader", "vae_name", VAE),
    }
    cache, out, missing = {}, {}, []
    for key, (node, inp, base) in want.items():
        if node not in cache:
            try:
                cache[node] = json.load(urllib.request.urlopen(f"{API}/object_info/{node}", timeout=10))[node]["input"]["required"][inp][0]
            except Exception:
                cache[node] = []
        match = next((n for n in cache[node] if os.path.basename(str(n).replace("\\", "/")) == base), None)
        if match is None: missing.append(base); out[key] = base
        else: out[key] = match
    if missing:
        sys.exit("model(s) not found in ComfyUI: " + ", ".join(missing) + "\n  -> see references/setup.md for the exact files + target folders.")
    return out

def post(p):
    req = urllib.request.Request(API + "/prompt",
                                 data=json.dumps({"prompt": p, "client_id": "bernini"}).encode(),
                                 headers={"Content-Type": "application/json"})
    try: r = json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e: sys.exit("POST /prompt failed: " + e.read().decode()[:1500])
    if r.get("error"): sys.exit("ERROR: " + json.dumps(r["error"])[:800] + "\n(missing node/model? see references/setup.md)")
    if r.get("node_errors"): sys.exit("node_errors: " + json.dumps(r["node_errors"])[:1200])
    return r["prompt_id"]

def poll(pid, budget=2400):
    t0 = time.time()
    while time.time() - t0 < budget:
        try: h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=10))
        except Exception: h = {}
        if h:
            v = list(h.values())[0]; st = v.get("status", {}).get("status_str"); outs = []
            for nid, o in v.get("outputs", {}).items():
                for k in ("gifs", "videos", "images"):
                    for f in o.get(k, []): outs.append((nid, f["filename"], f.get("subfolder", "")))
            return st, outs
        time.sleep(6)
    return "timeout", []

# ---------------------------------------------------------------- graph builder
def build_graph(task, src, refs, content, W, H, length, fps, sysprompt, prompt, negative,
                seed, steps, split, out_kind, name, M):
    """Assemble the Bernini API graph: dual-expert (hi I2V@3 / lo T2V@1.5) + BerniniConditioning.
    M = resolved model names (see resolve_models())."""
    pos_text = (sysprompt + "\n\n" + prompt).strip() if prompt else sysprompt
    g = {
        "clip":   {"class_type": "CLIPLoader", "inputs": {"clip_name": M["clip"], "type": "wan", "device": "default"}},
        "vae":    {"class_type": "VAELoader", "inputs": {"vae_name": M["vae"]}},
        "unet_hi":{"class_type": "UNETLoader", "inputs": {"unet_name": M["unet_hi"], "weight_dtype": "default"}},
        "unet_lo":{"class_type": "UNETLoader", "inputs": {"unet_name": M["unet_lo"], "weight_dtype": "default"}},
        "sage_hi":{"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["unet_hi", 0], "sage_attention": "auto"}},
        "sage_lo":{"class_type": "PathchSageAttentionKJ", "inputs": {"model": ["unet_lo", 0], "sage_attention": "auto"}},
        "lora_hi":{"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["sage_hi", 0], "lora_name": M["lora_hi"], "strength_model": LORA_HI_STR}},
        "lora_lo":{"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["sage_lo", 0], "lora_name": M["lora_lo"], "strength_model": LORA_LO_STR}},
        "pos":    {"class_type": "CLIPTextEncode", "inputs": {"text": pos_text, "clip": ["clip", 0]}},
        "neg":    {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["clip", 0]}},
        "sched":  {"class_type": "BasicScheduler", "inputs": {"model": ["unet_lo", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "split":  {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sched", 0], "step": split}},
        "ksel":   {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "bernini":{"class_type": "BerniniConditioning", "inputs": {
                       "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                       "width": W, "height": H, "length": length, "batch_size": 1, "ref_max_size": 848}},
        "samp_hi":{"class_type": "SamplerCustom", "inputs": {
                       "model": ["lora_hi", 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0,
                       "positive": ["bernini", 0], "negative": ["bernini", 1],
                       "sampler": ["ksel", 0], "sigmas": ["split", 0], "latent_image": ["bernini", 2]}},
        "samp_lo":{"class_type": "SamplerCustom", "inputs": {
                       "model": ["lora_lo", 0], "add_noise": False, "noise_seed": seed, "cfg": 1.0,
                       "positive": ["bernini", 0], "negative": ["bernini", 1],
                       "sampler": ["ksel", 0], "sigmas": ["split", 1], "latent_image": ["samp_hi", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["samp_lo", 0], "vae": ["vae", 0]}},
    }
    # --- source_video (v2v/rv2v/i2v/i2i/vi2v/vrc2v/mv2v/ads2v) ---
    if src:
        if is_video(src):
            g["load_src"] = {"class_type": "VHS_LoadVideo", "inputs": {
                "video": src, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                "frame_load_cap": length, "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan"}}
            g["bernini"]["inputs"]["source_video"] = ["load_src", 0]
        else:
            g["load_src"] = {"class_type": "LoadImage", "inputs": {"image": src}}
            g["bernini"]["inputs"]["source_video"] = ["load_src", 0]
    # --- reference_video (ads2v content) ---
    if content:
        if is_video(content):
            g["load_content"] = {"class_type": "VHS_LoadVideo", "inputs": {
                "video": content, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                "frame_load_cap": length, "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan"}}
            g["bernini"]["inputs"]["reference_video"] = ["load_content", 0]
        else:
            g["load_content"] = {"class_type": "LoadImage", "inputs": {"image": content}}
            g["bernini"]["inputs"]["reference_video"] = ["load_content", 0]
    # --- reference_images autogrow (r2v/r2i/rv2v/vi2v) ---
    # ComfyUI resolves an Autogrow link only from a FLAT dotted-path key it can see as a top-level
    # input value (execution.py build_nested_inputs restructures it). A nested dict buries the
    # [node,slot] link where the executor never resolves it -> the reference is silently dropped.
    if refs:
        for i, rf in enumerate(refs):
            nid = f"load_ref_{i}"
            g[nid] = {"class_type": "LoadImage", "inputs": {"image": rf}}
            g["bernini"]["inputs"][f"reference_images.reference_image_{i}"] = [nid, 0]
    # --- output ---
    if out_kind == "video":
        g["out"] = {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["decode", 0], "frame_rate": fps, "loop_count": 0,
            "filename_prefix": f"Bernini/{name}", "format": "video/h264-mp4", "pix_fmt": "yuv420p",
            "crf": 19, "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True}}
    else:
        g["out"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": f"Bernini/{name}"}}
    return g

# ---------------------------------------------------------------- run
def run(a):
    preflight()
    M = resolve_models()
    out_kind, srole, refpol, needs_content = TASKS[a.task]
    sysp = SYSTEM_PROMPTS[a.task]

    # validate inputs vs task
    if srole and not a.source: sys.exit(f"--task {a.task} needs --source ({'image' if srole=='image' else 'video' if srole=='video' else 'image or video'}).")
    if not srole and a.source: print(f"  note: --task {a.task} ignores --source.")
    if refpol == "req" and not a.ref: sys.exit(f"--task {a.task} needs at least one --ref image.")
    if refpol == "no" and a.ref: print(f"  note: --task {a.task} ignores --ref.")
    if needs_content and not a.content: sys.exit(f"--task {a.task} needs --content (image or video to insert).")
    if a.ref and len(a.ref) > 8: sys.exit("Bernini's reference_images slot holds at most 8 (the node's hard ceiling).")
    if a.ref and len(a.ref) > 2: print(f"  note: {len(a.ref)} references — your call; each one past ~2 trades away per-subject fidelity (subjects still stay distinct).")
    if not a.prompt and a.task not in ("t2v","t2i"): print("  ⚠ no --prompt: edits/refs need an instruction describing the result.")

    # stage inputs
    src = stage(a.source) if (srole and a.source) else None
    content = stage(a.content) if (needs_content and a.content) else None
    refs = [stage(r) for r in (a.ref or [])] if refpol != "no" else []

    # derive W,H,length,fps
    fps = a.fps
    if src and is_video(src):
        sw, sh_, snb, sfps = probe(os.path.join(INP, src))
        length = snap_len(a.frames) if a.frames else snap_len(min(snb, 81))
        W, H = (a.width, a.height) if (a.width and a.height) else fit_rect(sw, sh_, a.max_size)
        fps = round(sfps, 3)
    elif src and not is_video(src):                        # i2v / i2i from an image
        iw, ih, _, _ = probe(os.path.join(INP, src))
        if iw == 0: iw, ih = a.width or 832, a.height or 480
        W, H = (a.width, a.height) if (a.width and a.height) else fit_rect(iw, ih, a.max_size if out_kind=="video" else max(iw,ih,16))
        length = 1 if out_kind == "image" else snap_len(a.frames or 81)
    else:                                                  # t2v / t2i / r2v / r2i
        W = a.width or (1280 if out_kind == "video" else 1024)
        H = a.height or (720 if out_kind == "video" else 1024)
        W, H = snap16(W), snap16(H)
        # t2i/r2i: Bernini is a VIDEO model — a lone frame (length=1) is out-of-distribution and
        # comes out waxy/over-sharpened. Render a SHORT clip and keep the middle frame instead.
        length = snap_len(a.img_frames) if out_kind == "image" else snap_len(a.frames or 81)

    steps = a.steps
    split = a.split if a.split is not None else max(1, round(steps * 0.5))   # hi 0..split, lo split..steps
    seed = a.seed
    if seed < 0: seed = int.from_bytes(os.urandom(4), "big")
    name = a.name or f"bernini_{a.task}"

    g = build_graph(a.task, src, refs, content, W, H, length, fps, sysp, a.prompt or "", a.negative,
                    seed, steps, split, out_kind, name, M)

    if a.dump_graph:
        json.dump(g, open(a.dump_graph, "w"), indent=2); print("graph ->", a.dump_graph)

    print(f"[{a.task}] {out_kind} {W}x{H} len={length} fps={fps} steps={steps}(split {split}) seed={seed}")
    print(f"   system: {sysp[:70]}{'…' if len(sysp)>70 else ''}")
    if src: print(f"   source: {src}")
    if refs: print(f"   refs:   {', '.join(refs)}")
    if content: print(f"   content:{content}")
    if a.prompt: print(f"   prompt: {a.prompt[:90]}{'…' if len(a.prompt)>90 else ''}")

    pid = post(g); st, outs = poll(pid); print("status:", st)
    if out_kind == "image":
        imgs = sorted(os.path.join(OUT, sub, fn) for nid, fn, sub in outs if fn.lower().endswith(".png"))
        main = imgs[len(imgs) // 2] if imgs else None     # middle frame of the short clip
    else:
        main = next((os.path.join(OUT, sub, fn) for nid, fn, sub in outs if fn.lower().endswith(".mp4")), None)
    if not main:
        main = next((os.path.join(OUT, sub, fn) for nid, fn, sub in outs), None)
    if not main: print("no output produced (status %s). Check ComfyUI console." % st); sys.exit(2)
    print("FINAL=" + main)
    if not a.no_deliver:
        try:
            os.makedirs(DELIVER, exist_ok=True)
            d = os.path.join(DELIVER, name + ext); shutil.copy(main, d); print("delivered ->", d)
        except Exception:
            print(f"  (deliver skipped — {DELIVER} not writable; set DELIVER_DIR)")

def main():
    ap = argparse.ArgumentParser(prog="bernini", description="ByteDance Bernini-R unified video/image gen+edit on ComfyUI")
    ap.add_argument("--task", required=True, choices=list(TASKS), help="which Bernini task (selects inputs + system prompt)")
    ap.add_argument("--source", "-i", help="source image/video to edit/animate (v2v/rv2v/i2v/i2i/vi2v/vrc2v/mv2v/ads2v)")
    ap.add_argument("--ref", action="append", help="reference image (repeatable, 0-8; r2v/r2i/rv2v/vi2v)")
    ap.add_argument("--content", help="content image/video to insert (ads2v)")
    ap.add_argument("--prompt", "-p", default="", help="edit instruction / scene description (the 3-part Bernini prompt)")
    ap.add_argument("--negative", default=NEG_DEFAULT, help='negative prompt (default "%s")' % NEG_DEFAULT)
    ap.add_argument("--frames", type=int, default=0, help="frame count for video tasks (snapped to 8n+1; default 81 / source)")
    ap.add_argument("--img-frames", dest="img_frames", type=int, default=9, help="t2i/r2i render this many frames (8n+1) and keep the middle one — a lone frame is OOD for this video model and looks waxy; 1 = fast/low-quality")
    ap.add_argument("--width", type=int, default=0, help="override width (default: from source aspect / 1280 video / 1024 image)")
    ap.add_argument("--height", type=int, default=0, help="override height")
    ap.add_argument("--max-size", dest="max_size", type=int, default=1280, help="long-edge cap when deriving size from source (default 1280=720p; use 832 for a fast 480p draft)")
    ap.add_argument("--fps", type=float, default=16.0, help="output fps for generated video (edit modes follow the source)")
    ap.add_argument("--steps", type=int, default=6, help="total sampler steps (default 6: hi 0-3, lo 3-6)")
    ap.add_argument("--split", type=int, default=None, help="step at which hi-noise hands off to lo-noise (default steps/2)")
    ap.add_argument("--seed", type=int, default=42, help="seed (default 42; -1 = random)")
    ap.add_argument("--name", default=None, help="output filename prefix")
    ap.add_argument("--no-deliver", action="store_true", help="don't copy the result to DELIVER_DIR")
    ap.add_argument("--dump-graph", default=None, help="write the assembled API graph to a file (debug)")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
