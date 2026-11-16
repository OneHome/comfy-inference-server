#!/usr/bin/env python3
"""Makes ONE preview straight with the engine (no server, no qComfy): the quickest way to test Ref2VA mode.

    python comfy-inference-server/test_ref2va.py --prompt "A woman dances in the rain" \
        --ref /kaggle/working/refs/woman.png --ref-names "the dancer" --ref-size 70

It loads the models (a few minutes), generates one clip, copies the MP4 to --out and prints a line starting with
"RESULT:" with its path. The process ends afterwards, so both GPUs are free again for the server. The engine reads the
same settings as the server (h3_2k_settings.json, H3_MODE=ref2va unless you set another value).
"""
import os

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    parser = argparse.ArgumentParser(description="One Ref2VA / FL2VA preview, without the server")
    parser.add_argument("--prompt", default="", help="scene / action")
    parser.add_argument("--dialogue", default="", help="spoken line")
    parser.add_argument("--language", default="English", help="language of the spoken line")
    parser.add_argument("--sound", default="", help="sound effects / ambience")
    parser.add_argument("--music", default="", help="music")
    parser.add_argument("--ref", action="append", default=[], help="reference image (repeat it for several, in <Picture N> order)")
    parser.add_argument("--ref-names", default="", help='what each reference is, comma separated: "the dancer, the dog"')
    parser.add_argument("--ref-size", type=int, default=70, help="reference pixel budget, %% of the output frame (50-200)")
    parser.add_argument("--start", default="", help="optional first frame image")
    parser.add_argument("--end", default="", help="optional last frame image")
    parser.add_argument("--duration", default="5 s", help='one of the engine\'s lengths, e.g. "5 s"')
    parser.add_argument("--aspect", default="16:9 Landscape")
    parser.add_argument("--resolution", type=int, default=384, help="short side of the preview in px (384, 480, 540...)")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=0, help="0 = the LoRA's own 4 steps")
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--shift", type=float, default=12.0)
    parser.add_argument("--workdir", default="/kaggle/working", help="folder that contains Wan2GP")
    parser.add_argument("--out", default="/kaggle/working/outputs", help="folder that receives the MP4")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(args.workdir)

    import video_server as vs           # only for the notebook settings and the preview area helper
    vs.load_notebook_settings(args.workdir)
    os.environ.setdefault("H3_MODE", "ref2va")

    import PIL.Image

    def load(path):
        if not path:
            return None
        if not os.path.exists(path):
            sys.exit(f"❌ Image not found: {path}")
        return PIL.Image.open(path).convert("RGB")

    refs = [load(p) for p in args.ref]
    start, end = load(args.start), load(args.end)
    ref_size = max(50, min(200, args.ref_size))

    import video_engine as e            # loads the models on both GPUs: a few minutes
    if args.duration not in e.DURATIONS:
        sys.exit(f"❌ Duration {args.duration!r}: choose one of {', '.join(e.DURATIONS)}")
    if args.aspect not in e.ASPECTS:
        sys.exit(f"❌ Aspect {args.aspect!r}: choose one of {', '.join(e.ASPECTS)}")
    e.PREVIEW_AREA = vs.VideoWrapper._area(args.resolution)

    path, status, name = e.make_preview(
        args.prompt, args.dialogue, args.language, args.sound, args.music, start, end, args.aspect, args.duration,
        args.seed, e.NO_LORA, 1.0, e.NO_LORA, 1.0, "", None, 1.0, True, args.sampler, args.shift, args.steps,
        refs, args.ref_names, ref_size)

    print("\n" + str(status))
    try:
        import torch
        for dev in range(torch.cuda.device_count()):
            print(f"GPU {dev}: peak {torch.cuda.max_memory_allocated(dev) / 1024**3:.1f} GB allocated, "
                  f"{torch.cuda.max_memory_reserved(dev) / 1024**3:.1f} GB reserved")
    except Exception:
        pass
    if not path:
        sys.exit(1)
    final = os.path.join(out_dir, os.path.basename(path))
    shutil.copy(path, final)
    print(f"RESULT: {final}")


if __name__ == "__main__":
    main()
