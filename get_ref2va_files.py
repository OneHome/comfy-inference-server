#!/usr/bin/env python3
"""Downloads the two files that the Ref2VA mode of the video server needs (run it once per Kaggle session, after the
notebook's Step 1, which leaves the other weights in /kaggle/tmp):

    %cd /kaggle/working
    !python comfy-inference-server/get_ref2va_files.py

  * MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors   the Ref2VA transformer, the same pruned INT8 ConvRot kind of
                                                              file as the FL2VA one (about 20 GB) -> <models folder>
  * minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16    the LightX2V Ref2V Turbo 4-step LoRA made for Ref2VA
                                                              -> <LoRA folder>/ref2v_turbo/

Both come from the Hugging Face repo that Step 1 already uses (DeepBeepMeep/MiniMax-H3), at the same pinned revision
(the newest one as a fallback). Options:

    --check          only say what is there and what is missing
    --delete-fl2va   delete the FL2VA transformer first when the disk is too full for a second one (Step 1 downloads it
                     again whenever you want the normal mode back)
"""
import argparse
import json
import os
import shutil
import sys

REPO = "DeepBeepMeep/MiniMax-H3"
REVISION = "5905b204b04d019c69707d89d49be4c8ae1b134f"   # the revision of the notebook's Step 1 (28 Oct 2026)
DIT = "MiniMax-H3-Ref2VA-pruned_rank8_int8_convrot.safetensors"
FL2VA_DIT = "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
LORA_REMOTE = "loras/minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors"
LORA_NAME = os.path.basename(LORA_REMOTE)
NEEDED_FREE_GB = 24   # the transformer plus some room to work


def folders():
    settings = {}
    if os.path.exists("h3_2k_settings.json"):
        with open("h3_2k_settings.json") as f:
            settings = json.load(f)
    models = os.environ.get("H3_MODELS_DIR") or settings.get("H3_MODELS_DIR") or "/kaggle/tmp/h3_models"
    loras = os.environ.get("H3_LORA_DIR") or settings.get("H3_LORA_DIR") or "/kaggle/tmp/h3_loras"
    return models, loras


def gb(path):
    return os.path.getsize(path) / 1024**3


def download(filename, local_dir):
    """The file from the pinned revision, or from the newest one if that revision does not have it."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(REPO, filename, revision=REVISION, local_dir=local_dir)
    except Exception as error:
        print(f"   (revision {REVISION[:8]}: {type(error).__name__}; trying the newest one)")
        return hf_hub_download(REPO, filename, local_dir=local_dir)


def main():
    parser = argparse.ArgumentParser(description="Download the files of the Ref2VA mode")
    parser.add_argument("--check", action="store_true", help="only report")
    parser.add_argument("--delete-fl2va", action="store_true", help="delete the FL2VA transformer if the disk is too full")
    args = parser.parse_args()

    models, loras = folders()
    dit_path = os.path.join(models, DIT)
    lora_path = os.path.join(loras, "ref2v_turbo", LORA_NAME)
    os.makedirs(models, exist_ok=True)
    os.makedirs(os.path.dirname(lora_path), exist_ok=True)

    print(f"Models folder: {models}\nLoRA folder:   {loras}")
    for label, path in (("Ref2VA transformer", dit_path), ("Ref2V Turbo LoRA ", lora_path)):
        print(f"  {label}: " + (f"present ({gb(path):.2f} GB)" if os.path.exists(path) else "MISSING"))
    free = shutil.disk_usage(models).free / 1024**3
    print(f"Disk free: {free:.0f} GB")
    if args.check:
        return

    if not os.path.exists(dit_path):
        fl2va = os.path.join(models, FL2VA_DIT)
        if free < NEEDED_FREE_GB:
            if args.delete_fl2va and os.path.exists(fl2va):
                print(f"Only {free:.0f} GB free: deleting the FL2VA transformer ({gb(fl2va):.1f} GB).")
                os.remove(fl2va)
            else:
                sys.exit(f"❌ Only {free:.0f} GB free and the Ref2VA transformer needs about {NEEDED_FREE_GB} GB. "
                         f"Run again with --delete-fl2va to free the space of the FL2VA transformer "
                         f"(Step 1 downloads it again when you want the normal mode back).")
        print(f"Downloading {DIT} (about 20 GB)...")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        download(DIT, models)
        if not os.path.exists(dit_path):
            sys.exit(f"❌ The download finished but {dit_path} is not there.")

    if not os.path.exists(lora_path):
        print(f"Downloading {LORA_NAME}...")
        scratch = os.path.join(loras, ".ref2v_download")
        try:
            got = download(LORA_REMOTE, scratch)
            shutil.move(got, lora_path)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print("\n✅ Ready:")
    for path in (dit_path, lora_path):
        print(f"   {gb(path):6.2f} GB  {path}")
    print("\nNow start the server with H3_MODE=ref2va (see the notebook cell in the README).")


if __name__ == "__main__":
    main()
