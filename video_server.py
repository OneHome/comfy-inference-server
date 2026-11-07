#!/usr/bin/env python3
"""qComfy video server: the MiniMax H3 engine (video_engine.py) behind the qComfy protocol.

Run it from the folder that contains the Wan2GP checkout, i.e. where the notebook's Step 1 left it
(/kaggle/working):

    python comfy-inference-server/video_server.py --bind 0.0.0.0:28888 --password ...

It replaces server.py for a session: the two GPUs are used by the video engine, so the image server cannot run
at the same time. qComfy's Video tab talks to it with these request types (see server.py for the transport):

    options          -> {"video": {tables for the pickers}}
    video            -> preview clip (4-step PDMD, low resolution). Replies video_progress..., video_chunk..., video
    video_upscale    -> Open-H3 Regenerate-2K of a preview that is still in memory. Same replies
    video_eta        -> {"text": "..."} estimate for an upscale
    video_discard    -> forget a preview
    chunk            -> LoRA files uploaded from qComfy (written to the LoRA folder, picked by name afterwards)

Uploaded images and the videos the engine writes stay in a private scratch directory (RAM-backed /dev/shm when it
has room, otherwise /kaggle/tmp), never in /kaggle/working, and each video is deleted right after it is sent.
"""
# the engine's process environment in the notebook: it must be set before torch is imported
import os

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import io
import json
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
            }
        }
        self._send("options", data)

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
        path = os.path.abspath(os.path.join(self.model_path, name))
        if not path.startswith(os.path.abspath(self.model_path) + os.sep) or not os.path.isfile(path):
            raise RuntimeError(f"The LoRA file '{name}' was not uploaded to the server.")
        return path

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
        try:
            os.remove(path)
        except OSError:
            pass
        vid = f"{name}_{kind}_{int(time.time())}"
        total = max(1, (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE)
        for i in range(total):
            self._send("video_chunk", {"id": vid, "index": i, "total": total,
                                       "data": data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]})
        self._send("video", {"id": vid, "name": name, "kind": kind, "status": status, "size": len(data), **(extra or {})})

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
        path, status, name = self._run(e.make_preview, *args)
        if not path:
            raise RuntimeError(status or "The preview could not be generated.")
        eta = ""
        try:
            eta = e.picker_eta(name, next((o for o in e.OUTPUTS), ""), next(iter(e.REFINE_PRESETS), ""))
        except Exception:
            pass
        self._deliver(path, "preview", name, status, {"duration": duration, "aspect": aspect, "eta": eta})

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


def main():
    import server  # the same transport (encrypted websocket, queue, uploads) the image server uses

    parser = argparse.ArgumentParser(description="qComfy video server")
    parser.add_argument("--bind", type=str, default="0.0.0.0:28888", help="address (ip:port) to listen on")
    parser.add_argument("--password", type=str, default=server.DEFAULT_PASSWORD,
                        help="password to derive encryption key from")
    parser.add_argument("--workdir", type=str, default="/kaggle/working", help="folder that contains Wan2GP")
    args = parser.parse_args()

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
