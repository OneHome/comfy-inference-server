#!/usr/bin/env python3
"""qComfy video server: the MiniMax H3 engine (video_engine.py) behind the qComfy protocol.

Run it from the folder that contains the Wan2GP checkout, i.e. where the notebook's Step 1 left it
(/kaggle/working):

    python comfy-inference-server/video_server.py --bind 0.0.0.0:28888 --password ...

It replaces server.py for a session: the two GPUs are used by the video engine, so the image server cannot run
at the same time. qComfy's Video tab talks to it with these request types (see server.py for the transport):

    options          -> {"video": {tables for the pickers}}
    video            -> preview clip (4-step PDMD; "resolution" picks the native size). Replies video_progress..., video_chunk..., video
    video_upscale    -> Open-H3 Regenerate-2K of a preview that is still in memory. Same replies
    video_eta        -> {"text": "..."} estimate for an upscale
    video_discard    -> forget a preview
    chunk            -> LoRA files uploaded from qComfy (type "custom": written to <LoRA folder>/custom, picked by name
                        afterwards with "custom_file" / "s3"; "options" lists them as "uploaded_loras")

Uploaded images and the videos the engine writes stay in a private scratch directory (RAM-backed /dev/shm when it
has room, otherwise /kaggle/tmp), never in /kaggle/working, and each video is deleted right after it is sent.
"""
# the engine's process environment in the notebook: it must be set before torch is imported
import os
import re

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import io
import json
import subprocess
import sys
import threading
import time

import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CHUNK_SIZE = 4 * 1024 * 1024  # a video travels in pieces of this size
POLL_SECONDS = 0.7
SAMPLERS = ["euler", "res_multistep"]
MAX_EXTRA_LORAS = 8   # uploaded files + links a single clip may use, on top of the two catalog LoRAs
HF_SHORT = re.compile(r"^([\w.-]+/[\w.-]+)/(.+\.safetensors)$")   # user/repo/path/file.safetensors
CUSTOM_LORA_DIR = "custom"  # sub-folder of the LoRA folder that receives the LoRA files uploaded from qComfy
# Native size of the first stage (short side in px, 16:9 grid of 32 px blocks: 384p -> 672x384, the engine's default).
# The VRAM guard of the engine refuses a size that does not fit and says so.
RESOLUTIONS = {"384p (default)": 384, "480p": 480, "540p": 540, "576p": 576, "720p": 720}
DEFAULT_RESOLUTION = "384p (default)"
SETTINGS_FILE = "h3_2k_settings.json"  # Step 1's choices (model folders and files), as environment variables


def load_notebook_settings(workdir):
    path = os.path.join(workdir, SETTINGS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            for key, value in json.load(f).items():
                os.environ.setdefault(key, value)


class VideoWrapper:
    """What server.Inference expects from a wrapper (reset / set / options / callback / model_path / fetch), for the
    H3 engine. Request types handled: video, video_upscale, video_eta, video_discard."""

    def __init__(self, engine):
        self.engine = engine
        self.callback = None
        self.params = {}
        self.model_path = engine.LORA_DIR  # LoRA files uploaded from qComfy end up here
        os.makedirs(self.model_path, exist_ok=True)
        self._extra_loras = []
        self._install_lora_hook()

    # ---- interface used by server.Inference --------------------------------------------------------------------
    def reset(self):
        self.params = {}

    def set(self, **kwargs):
        self.params.update(kwargs)

    def fetch(self, result_id):
        raise RuntimeError("Videos are sent directly, there is nothing to fetch.")

    def manage(self):
        pass  # nothing to load or unload on demand: the engine keeps its models resident

    def metadata(self):
        pass

    def __getattr__(self, name):
        if name in ("txt2img", "img2img", "upscale", "annotate"):
            raise RuntimeError("This is the video server: images need server.py (they cannot run at the same time).")
        raise AttributeError(name)

    def options(self):
        e = self.engine
        data = {
            "video": {
                "aspects": list(e.ASPECTS),
                "durations": {k: v for k, v in e.DURATIONS.items()},  # label -> frames (24 fps)
                "outputs": list(e.OUTPUTS),
                "preview_only": e.PREVIEW_ONLY,
                "presets": list(e.REFINE_PRESETS),
                "languages": list(e.LANGUAGES),
                "samplers": SAMPLERS,
                "loras": {k: {"trigger": v.get("trigger", ""), "scale": v.get("scale", 1.0), "info": v.get("info", "")}
                          for k, v in e.LORA_CATALOG.items()},
                "no_lora": e.NO_LORA,
                "preview_steps": e.PREVIEW_STEPS,
                "default_sampler": e.DEFAULT_SAMPLER,
                "max_extra_loras": MAX_EXTRA_LORAS,
                "resolutions": list(RESOLUTIONS),
                "default_resolution": DEFAULT_RESOLUTION,
                "uploaded_loras": self._uploaded_loras(),
            }
        }
        self._send("options", data)

    def _uploaded_loras(self):
        folder = os.path.join(self.model_path, CUSTOM_LORA_DIR)
        try:
            return sorted(f"{CUSTOM_LORA_DIR}/{f}" for f in os.listdir(folder) if f.lower().endswith(".safetensors"))
        except OSError:
            return []

    @staticmethod
    def _area(short):
        width = max(1, round(short * 16 / 9 / 32)) * 32
        return width * short

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _send(self, kind, data):
        """Send one message; if qComfy cancelled the request the server answers False."""
        if self.callback and self.callback({"type": kind, "data": data}) is False:
            raise Exception("Aborted")

    @staticmethod
    def _image(raw):
        if not raw:
            return None
        if isinstance(raw, (bytes, bytearray)):
            return PIL.Image.open(io.BytesIO(bytes(raw))).convert("RGB")
        return raw

    @staticmethod
    def _choice(value, options, label, default=None):
        if value in (None, ""):
            value = default
        if value not in options:
            raise RuntimeError(f"Unknown {label}: {value!r}. Options: {', '.join(map(str, options))}")
        return value

    def _lora_file(self, name):
        if not name:
            return None
        if not str(name).lower().endswith(".safetensors"):
            raise RuntimeError(f"The LoRA file '{name}' must be a .safetensors file.")
        path = os.path.abspath(os.path.join(self.model_path, name))
        if not path.startswith(os.path.abspath(self.model_path) + os.sep) or not os.path.isfile(path):
            raise RuntimeError(f"The LoRA file '{name}' was not uploaded to the server.")
        return path

    def _fetch_link(self, link):
        """Download a LoRA from a link into the LoRA folder (Hugging Face and mirrors, or any direct .safetensors URL)."""
        link = str(link).strip()
        if "://" not in link and HF_SHORT.match(link):
            from huggingface_hub import hf_hub_download
            m = HF_SHORT.match(link)
            return hf_hub_download(repo_id=m.group(1), filename=m.group(2), local_dir=self.model_path)
        if not re.match(r"^https?://", link):
            raise RuntimeError("A LoRA link must start with https:// (or be user/repo/file.safetensors).")
        return self.engine.fetch_lora(link)

    def _resolve_lora(self, entry):
        if entry.get("file"):
            return self._lora_file(entry["file"])
        if entry.get("url"):
            return self._fetch_link(entry["url"])
        return None

    def _install_lora_hook(self):
        """Lets a clip use any number of extra LoRAs on top of the engine's two catalog slots and its single custom one.
        The engine builds {key: (path, strength)} in _user_loras and applies the whole dict, so extra entries just join it."""
        e, wrapper = self.engine, self
        original = getattr(e._user_loras, "_original", e._user_loras)

        def user_loras_multi(*args, **kwargs):
            selections, triggers = original(*args, **kwargs)
            for entry in list(wrapper._extra_loras):
                path = wrapper._resolve_lora(entry)
                strength = float(entry.get("strength", 1.0))
                if path and strength != 0.0:
                    key = f"custom:{os.path.basename(path)}"
                    while key in selections:
                        key += "+"
                    selections[key] = (path, strength)
            return selections, triggers

        user_loras_multi._original = original
        e._user_loras = user_loras_multi

    def _run(self, job, *args):
        """Run an engine call while a thread reports its progress and watches for a cancel."""
        e = self.engine
        done, state, started = threading.Event(), {"aborted": False}, time.time()

        def poll():
            last = None
            while not done.wait(POLL_SECONDS):
                p = e.PROGRESS
                payload = {"fraction": float(p.get("fraction") or 0.0), "text": str(p.get("text") or ""),
                           "elapsed": time.time() - started}
                key = (round(payload["fraction"], 3), payload["text"])
                if key == last:
                    continue
                last = key
                try:
                    stop = self.callback and self.callback({"type": "video_progress", "data": payload}) is False
                except Exception:
                    stop = True
                if stop:
                    state["aborted"] = True
                    if getattr(e, "pipeline", None) is not None:
                        e.pipeline._interrupt = True
                    return

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        try:
            result = job(*args)
        finally:
            done.set()
            thread.join(2)
        if state["aborted"]:
            raise Exception("Aborted")
        return result

    def _deliver(self, path, kind, name, status, extra=None):
        """Send the MP4 in pieces, then a final 'video' message, and delete the file."""
        with open(path, "rb") as f:
            data = f.read()
        info = probe_clip(path)
        try:
            os.remove(path)
        except OSError:
            pass
        vid = f"{name}_{kind}_{int(time.time())}"
        total = max(1, (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE)
        for i in range(total):
            self._send("video_chunk", {"id": vid, "index": i, "total": total,
                                       "data": data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]})
        self._send("video", {"id": vid, "name": name, "kind": kind, "status": status, "size": len(data), **(extra or {}), **info})

    # ---- requests ----------------------------------------------------------------------------------------------
    def video(self):
        """Preview clip: text / image / first+last frame -> video."""
        e, p = self.engine, self.params
        start_img, end_img = self._image(p.get("start_img")), self._image(p.get("end_img"))
        prompt = (p.get("prompt") or "").strip()
        if not prompt and start_img is None and end_img is None:
            raise RuntimeError("Write a prompt or give a start image.")
        aspect = self._choice(p.get("aspect"), list(e.ASPECTS), "aspect ratio", next(iter(e.ASPECTS)))
        duration = self._choice(p.get("duration"), list(e.DURATIONS), "duration", "5 s" if "5 s" in e.DURATIONS else None)
        language = self._choice(p.get("language"), e.LANGUAGES, "language", "English")
        sampler = self._choice(p.get("sampler"), SAMPLERS, "sampler", e.DEFAULT_SAMPLER)
        lora1 = self._choice(p.get("lora1"), [e.NO_LORA] + list(e.LORA_CATALOG), "LoRA 1", e.NO_LORA)
        lora2 = self._choice(p.get("lora2"), [e.NO_LORA] + list(e.LORA_CATALOG), "LoRA 2", e.NO_LORA)

        args = (
            prompt, p.get("dialogue") or "", language, p.get("sound") or "", p.get("music") or "",
            start_img, end_img, aspect, duration,
            int(p.get("seed", -1)),
            lora1, float(p.get("s1", 1.0)), lora2, float(p.get("s2", 1.0)),
            (p.get("custom_url") or "").strip(), self._lora_file(p.get("custom_file")), float(p.get("s3", 1.0)),
            bool(p.get("auto_trigger", True)), sampler, float(p.get("shift", 12.0)), int(p.get("steps", 0) or 0),
        )
        resolution = self._choice(p.get("resolution"), list(RESOLUTIONS), "resolution", DEFAULT_RESOLUTION)
        extras = [x for x in (p.get("custom_loras") or []) if isinstance(x, dict) and (x.get("file") or x.get("url"))]
        if len(extras) > MAX_EXTRA_LORAS:
            raise RuntimeError(f"At most {MAX_EXTRA_LORAS} extra LoRAs per clip.")
        self._extra_loras = extras
        default_area = e.PREVIEW_AREA
        e.PREVIEW_AREA = self._area(RESOLUTIONS[resolution])  # preview_size() reads it on every call
        try:
            path, status, name = self._run(e.make_preview, *args)
        finally:
            e.PREVIEW_AREA = default_area
            self._extra_loras = []
        if not path:
            raise RuntimeError(status or "The preview could not be generated.")
        eta = ""
        try:
            eta = e.picker_eta(name, next((o for o in e.OUTPUTS), ""), next(iter(e.REFINE_PRESETS), ""))
        except Exception:
            pass
        self._deliver(path, "preview", name, status, {"duration": duration, "aspect": aspect, "eta": eta,
                                                       "resolution": resolution})

    def video_upscale(self):
        """Regenerate a preview that is still in memory at a higher resolution."""
        e, p = self.engine, self.params
        name = p.get("name")
        output = self._choice(p.get("output"), list(e.OUTPUTS), "output resolution")
        preset = self._choice(p.get("preset"), list(e.REFINE_PRESETS), "refine preset", next(iter(e.REFINE_PRESETS)))
        path, status = self._run(e.make_upscale, name, output, preset, int(p.get("seed", -1)), bool(p.get("anchor", True)))
        if not path:
            raise RuntimeError(status or "The upscale could not be generated.")
        self._deliver(path, "upscale", name, status, {"output": output, "preset": preset})

    def video_eta(self):
        e, p = self.engine, self.params
        text = e.picker_eta(p.get("name"), p.get("output"), p.get("preset"))
        self._send("video_eta", {"name": p.get("name"), "text": text})

    def video_discard(self):
        """Forget a preview and everything stored for it (latent, prompt, start/end frames)."""
        name = self.params.get("name") or ""
        if not name or os.sep in name or name.startswith("."):
            raise RuntimeError("Invalid preview name.")
        removed = 0
        stems = {name, f"{name}_start", f"{name}_end"}  # exactly what the engine writes for one preview
        for f in os.listdir(self.engine.PREVIEW_DIR):
            if os.path.splitext(f)[0] in stems:
                try:
                    os.remove(os.path.join(self.engine.PREVIEW_DIR, f))
                    removed += 1
                except OSError:
                    pass
        self._send("video_discarded", {"name": name, "files": removed})


def probe_clip(path):
    """Size/length and a small JPEG poster frame of a clip, for qComfy's gallery. {} entries are simply left out."""
    info = {}
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height", "-show_entries", "format=duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout)
        stream = (data.get("streams") or [{}])[0]
        info["width"], info["height"] = int(stream["width"]), int(stream["height"])
        info["seconds"] = round(float(data["format"]["duration"]), 2)
    except Exception:
        pass
    try:
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1", "-vf", "scale=-2:240",
                              "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "4", "-"],
                             capture_output=True, timeout=30)
        if out.returncode == 0 and out.stdout:
            info["poster"] = out.stdout
    except Exception:
        pass
    return info


class PromptHider:
    """stdout wrapper: the engine prints the full prompt (scene, spoken line, sound, music) before every clip.
    That block is replaced by a placeholder so it does not show up in the Kaggle cell output."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if isinstance(text, str) and text.startswith("\U0001f4dd") and ("=" * 70) in text:
            text = "\U0001f4dd (prompt hidden)\n" + "=" * 70
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def main():
    import server  # the same transport (encrypted websocket, queue, uploads) the image server uses

    parser = argparse.ArgumentParser(description="qComfy video server")
    parser.add_argument("--bind", type=str, default="0.0.0.0:28888", help="address (ip:port) to listen on")
    parser.add_argument("--password", type=str, default=server.DEFAULT_PASSWORD,
                        help="password to derive encryption key from")
    parser.add_argument("--workdir", type=str, default="/kaggle/working", help="folder that contains Wan2GP")
    parser.add_argument("--show-prompts", action="store_true", help="print the prompts in the server output (hidden by default)")
    args = parser.parse_args()
    if not args.show_prompts:
        sys.stdout = PromptHider(sys.stdout)

    os.chdir(args.workdir)
    load_notebook_settings(args.workdir)
    import video_engine  # loads the models on both GPUs: this takes a few minutes

    ip, port = args.bind.rsplit(":", 1)
    video_server = server.Server(VideoWrapper(video_engine), ip, port, args.password)
    video_server.start()
    print(f"Video server listening on {args.bind}", flush=True)
    try:
        video_server.join()
    except KeyboardInterrupt:
        video_server.stop()


if __name__ == "__main__":
    main()
