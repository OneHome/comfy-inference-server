# Ref2VA mode (experimental)

The video server can run in two modes. Which one is chosen when the server starts, with the `H3_MODE` environment
variable, and the two cannot run at the same time (the single H3 transformer takes both T4s):

| | `H3_MODE` | Transformer | Preview LoRA | Inputs | Upscale 2K |
|---|---|---|---|---|---|
| default | `fl2va` (or unset) | FL2VA pruned INT8 ConvRot | PDMD 4-step | text, first frame, last frame | yes |
| this file | `ref2va` | Ref2VA pruned INT8 ConvRot | LightX2V Ref2V Turbo 4-step | text, **reference images**, optional first / last frame | no (previews only) |

Nothing changes unless `H3_MODE=ref2va` is set.

## The notebook

`MiniMax_H3_Ref2VA_Kaggle.ipynb` does all of this in one go (it is the FL2VA notebook with the minimal changes: Step 1 downloads
the Ref2VA transformer and the Ref2V Turbo LoRA instead of the FL2VA ones and writes `H3_MODE=ref2va` into
`h3_2k_settings.json`). Its cells: Step 1 weights, Step 2 this repository, Step 3 a quick test without the server
(`test_ref2va.py`), Step 4 the server. The sections below describe the same thing by hand.

## Notebook cells (by hand)

Run the normal Step 1 first (it leaves the shared weights in `/kaggle/tmp`). Then, instead of the usual server cell:

**1. Download the two extra files (once per session, about 20 GB):**

```python
%cd /kaggle/working
!git clone https://github.com/OneHome/comfy-inference-server   # skip if it is already there
!pip install -q -r comfy-inference-server/requirements.txt
!python comfy-inference-server/get_ref2va_files.py
```

If the disk is too full it tells you, and `--delete-fl2va` frees the space of the FL2VA transformer
(Step 1 downloads it again when you want the normal mode back). `--check` only reports what is there.

**2. Start the server in Ref2VA mode** (the same cell as before, with `env=` added to the `Popen`):

```python
!pip install -q gradio
import os, subprocess, secrets, inspect, time, socket
from gradio import networking

PORT = 28888
PASSWORD = "redka"

server = subprocess.Popen(
    ["python", "comfy-inference-server/video_server.py", "--bind", f"127.0.0.1:{PORT}", "--password", PASSWORD],
    cwd="/kaggle/working",
    env={**os.environ, "H3_MODE": "ref2va"},
)
# ...the rest of the cell does not change
```

The server prints `MiniMax H3 mode: ref2va | minimax_h3_ref2va_pruned | pipeline.reference_mode = True` while it loads.
If that last value is not `True` it stops with a message: copy it.

## What a request can carry

`video` requests accept three extra parameters (see `video_server.py`):

* `refs`: list of reference images (bytes). Their order is the order of `<Picture 1>`, `<Picture 2>`... in the prompt.
  The server allows up to `H3_MAX_REFS` (default 4); Wan2GP allows 9, but two T4 have little room.
* `ref_names`: what each image is, comma separated or a list ("the dancer, the dog"). The first one is `<Subject 1>`.
* `ref_size`: the pixel budget of each reference in % of the output frame, 50 to 200 (default 70). Lower is faster
  and uses less memory.

The prompt boxes keep their meaning: the engine writes the six sections Ref2VA expects (`subject_definitions`,
`summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`) around them.
A first / last frame can be combined with references. `options` now also reports `mode`, `references`
(`enabled`, `max_images`, `size`) and `upscale`.

## Testing it, in this order

`test_ref2va.py` makes one clip straight with the engine (no server, no qComfy), prints the GPU memory peaks and ends, so
both GPUs are free again afterwards:

```
python comfy-inference-server/test_ref2va.py --prompt "A woman dances in the rain" --ref refs/woman.png --ref-names "the dancer" --ref-size 70
```

1. **No references.** A text-only clip. This checks that the Ref2VA transformer, the tensor-parallel split and the
   Turbo LoRA work at all. Note the seconds per step.
2. **One reference** at 384p, `ref_size` 70. Compare memory and time with step 1.
3. **Two and three references**, then `ref_size` 100. The VRAM guard of the engine refuses what does not fit and says so.

If something fails, send the traceback and the `MiniMax H3 mode:` line.

## Known limits

* Previews only: the Open-H3 refiner was built for the FL2VA transformer. `H3_REF2VA_UPSCALE=1` lets the upscale run
  anyway, as an experiment (it is not expected to match FL2VA quality).
* Reference videos and reference audio (a voice) are not wired yet: the pipeline supports them, the server does not.
* The Turbo LoRA was trained at 544p; 384p previews are cheaper but may look softer.
* The `detailed_description` the engine writes is short. The official guide asks for a long, chronological one, so
  the prompt rewriter is where the quality will come from.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `H3_MODE` | `fl2va` | `fl2va` or `ref2va` |
| `H3_DIT_FILE` | Ref2VA / FL2VA pruned INT8 ConvRot, by mode | transformer file inside the models folder |
| `H3_REF2V_LORA` | `<LoRA folder>/ref2v_turbo/minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors` | preview LoRA of Ref2VA mode |
| `H3_MAX_REFS` | `4` | most reference images per clip |
| `H3_REF2VA_UPSCALE` | `0` | `1` allows the 2K upscale in Ref2VA mode |

`test_ref2va.py` options: `--prompt --dialogue --language --sound --music --ref (repeat) --ref-names --ref-size --start --end --duration --aspect --resolution --seed`.

`build_video_engine.py` carries the changes (`REF2VA_PATCHES`) so that a newer version of the notebook can be converted
the same way; `video_engine.py` in this folder is already the result.
