import os
import sys
import io
import gc
import random
import time
import math
import shutil

import PIL.Image
import PIL.ImageOps
import PIL.ImageFilter

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional
import lark
import safetensors

import folder_paths
from folder_paths import folder_names_and_paths
import comfy
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.controlnet
import comfy.model_management
import comfy.utils
import latent_preview

latent_preview.set_preview_method("latent2rgb")

STATIC = ["model_path", "callback", "device_names", "temporary", "device", "_loaded"]

prompt_grammar = lark.Lark(r"""
!start: (prompt | /[\[\]:]/+)*
prompt: (scheduled | lora | plain | WHITESPACE)*
scheduled: "[" [prompt ":"] prompt ":" [_WHITESPACE] step_specifier [_WHITESPACE]"]"
step_specifier: NUMBER
lora: "<" "lora" ":" filename [ ":" [_WHITESPACE] NUMBER [_WHITESPACE] ] ">"
WHITESPACE: /\s+/
_WHITESPACE: /\s+/
plain: /[^\\\[\]<>:|]+/
filename: /[^<>:|\\]+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def format_float(x):
    return f"{x:.4f}".rstrip('0').rstrip('.')

def model_name(x):
    return x.rsplit('.',1)[0].rsplit(os.path.sep,1)[-1]

class AbortError(RuntimeError):
    pass

class GenerationParameters():
    def __init__(self, comfyui_path):
        comfyui_path = os.path.abspath(comfyui_path)
        self.model_path = os.path.join(comfyui_path, "models")

        self._register_model_folders(self.model_path)

        self.callback = None
        self.temporary = {}
        self.device = None
        self._loaded = {}

        self.device_names = []
        for i in range(torch.cuda.device_count()):
            original = torch.cuda.get_device_name(i)
            name = original
            n = 2
            while name in self.device_names:
                name = original + f" ({n})"
                n += 1
            self.device_names += [name]

        self.device_names += ["CPU"]

    @staticmethod
    def _register_model_folders(model_path):
        if not model_path or not os.path.isdir(model_path):
            return

        for comfy_key in folder_names_and_paths:
            sub = os.path.join(model_path, comfy_key)
            if os.path.isdir(sub):
                folder_paths.add_model_folder_path(comfy_key, sub)

    @staticmethod
    def _get_filenames(folder_key):
        return [os.path.join(folder_key, f) for f in folder_paths.get_filename_list(folder_key)]

    def set_status(self, status, reset=True):
        if self.callback:
            if not self.callback({"type": "status", "data": {"message": status, "reset": reset}}):
                raise AbortError("Aborted")

    def set_progress(self, progress):
        if self.callback:
            if not self.callback({"type": "progress", "data": progress}):
                comfy.model_management.soft_empty_cache()
                raise AbortError("Aborted")

    def on_step(self, progress, latents=None):
        step = self.current_step
        total = self.total_steps
        rate = progress.get("rate", 0) or 0
        remaining = (total - step) / rate if rate and total else 0

        if "n" in progress:
            self.current_step += 1

        progress = {"current": step, "total": total, "rate": rate, "remaining": remaining, "unit": "it/s"}

        if latents is not None and self.preview:
            try:
                previewer = latent_preview.get_previewer(self.model.load_device, self.model.model.latent_format)
                if previewer:
                    preview_image = previewer.decode_latent_to_preview_image("JPEG", latents)
                    if preview_image is not None:
                        fmt, img, max_size = preview_image
                        bytesio = io.BytesIO()
                        img.save(bytesio, format="JPEG")
                        progress["previews"] = [bytesio.getvalue()]
            except Exception:
                pass

        self.set_progress(progress)

    def on_complete(self, images, metadata):
        if self.callback:
            self.set_status("Fetching")

            id = random.randrange(2147483646)
            self.temporary[id] = (images, metadata)
            images_data = []
            for i in images:
                bytesio = io.BytesIO()
                im = i.copy()
                im.thumbnail((256,256), PIL.Image.Resampling.LANCZOS)
                im.save(bytesio, format="JPEG")
                images_data += [bytesio.getvalue()]
            self.callback({"type": "temporary", "data": {"id": id, "images": images_data, "metadata": metadata, "type": "JPEG"}})

        comfy.model_management.soft_empty_cache()

    def fetch(self, id):
        if id in self.temporary:
            images, metadata = self.temporary[id]
            del self.temporary[id]

            images_data = []
            for i in images:
                bytesio = io.BytesIO()
                i.save(bytesio, format="PNG")
                images_data += [bytesio.getvalue()]

            return {"type": "result", "data": {"images": images_data, "metadata": metadata, "type": "PNG"}}
        return None

    def reset(self):
        for attr in list(self.__dict__.keys()):
            if not attr in STATIC:
                delattr(self, attr)

    def __getattr__(self, item):
        return None

    @staticmethod
    def _bytes_to_image(value):
        img = PIL.Image.open(io.BytesIO(value))
        img = PIL.ImageOps.exif_transpose(img)
        if img.mode == 'RGBA':
            img = PIL.Image.alpha_composite(PIL.Image.new('RGBA', img.size, (0, 0, 0)), img)
            img = img.convert("RGB")
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img

    @staticmethod
    def _bytes_to_mask(value):
        img = PIL.Image.open(io.BytesIO(value))
        img = PIL.ImageOps.exif_transpose(img)
        if img.mode == 'RGBA':
            img = img.split()[-1]
        else:
            img = img.convert("L")
        return img

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key == "image":
                if type(value) == list:
                    value = [self._bytes_to_image(v) if type(v) in (bytes, bytearray) else v for v in value]
                elif type(value) in (bytes, bytearray):
                    value = self._bytes_to_image(value)
            if key == "mask":
                if type(value) == list:
                    value = [self._bytes_to_mask(v) if type(v) in (bytes, bytearray) else v for v in value]
                elif type(value) in (bytes, bytearray):
                    value = self._bytes_to_mask(value)
            if key == "control" and value:
                # list of {"image": bytes, "model": str, "strength": float, "start": float, "end": float}
                units = []
                for u in value:
                    u = dict(u)
                    if type(u.get("image")) in (bytes, bytearray):
                        u["image"] = self._bytes_to_image(u["image"])
                    if type(u.get("mask")) in (bytes, bytearray):
                        u["mask"] = self._bytes_to_mask(u["mask"])
                    units.append(u)
                value = units
            if key in STATIC:
                continue
            setattr(self, key, value)

    def _resolve_model_path(self, name):
        if not name:
            return None

        folder_key, bare = name.replace("\\", "/").split("/", 1)
        return folder_paths.get_full_path_or_raise(folder_key, bare)

    def _load_unet(self, name):
        path = self._resolve_model_path(name)
        self.set_status("Loading UNET")
        model = comfy.sd.load_diffusion_model(path)
        return path, model

    def _load_clip(self, name, clip_type=None):
        path = self._resolve_model_path(name)
        if clip_type is None:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        self.set_status("Loading CLIP")
        model = comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        return path, model

    def _load_upscale_model(self, name):
        from spandrel import ModelLoader, ImageModelDescriptor
        path = self._resolve_model_path(name)
        cached = self._loaded.get("upscaler")
        if cached and cached[0] == path:
            return cached[1]
        self.set_status("Loading Upscaler")
        try:
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
            model = ModelLoader().load_from_state_dict(sd)
            if not model or not isinstance(model, ImageModelDescriptor):
                raise RuntimeError()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load a Upscaler from '{model_name(name)}'."
            ) from exc
        model = model.eval()
        self._loaded["upscaler"] = (path, model)
        return model

    def _load_vae(self, name):
        path = self._resolve_model_path(name)
        self.set_status("Loading VAE")
        sd, _metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return path, vae

    def _load_models(self):
        """Load UNet, CLIP, and VAE."""
        if self.checkpoint:
            ckpt_path = self._resolve_model_path(self.checkpoint)
            cached_unet = self._loaded.get("unet")
            cached_clip = self._loaded.get("clip")
            cached_vae = self._loaded.get("vae")
            if all(cached and cached[0] == ckpt_path for cached in (cached_unet, cached_clip, cached_vae)):
                self.model = cached_unet[1]
                self.clip_model = cached_clip[1]
                self.vae_model = cached_vae[1]
                return

            self.set_status("Loading Checkpoint")
            model, clip, vae, _clipvision = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            self.model = model
            self.clip_model = clip
            self.vae_model = vae
            self._validate_loaded_models(checkpoint=True)
            self._loaded["unet"] = (ckpt_path, model)
            self._loaded["clip"] = (ckpt_path, clip)
            self._loaded["vae"] = (ckpt_path, vae)
        else:
            clip_type = getattr(comfy.sd.CLIPType, (self.clip_type or "stable_diffusion").upper(),
                                comfy.sd.CLIPType.STABLE_DIFFUSION)
            unet_path, self.model = self._load_unet(self.unet)
            clip_path, self.clip_model = self._load_clip(self.clip, clip_type=clip_type)
            vae_path, self.vae_model = self._load_vae(self.vae)
            self._validate_loaded_models(checkpoint=False)
            self._loaded["unet"] = (unet_path, self.model)
            self._loaded["clip"] = (clip_path, self.clip_model)
            self._loaded["vae"] = (vae_path, self.vae_model)

    def _validate_loaded_models(self, checkpoint=False):
        if self.model is None:
            raise RuntimeError(
                f"Failed to load a UNet from '{model_name(self.checkpoint if checkpoint else self.unet)}'."
            )
        if self.clip_model is None:
            if checkpoint:
                raise RuntimeError(
                    f"Failed to load a CLIP from '{model_name(self.checkpoint)}'."
                )
            else:
                raise RuntimeError(
                    f"Failed to load a CLIP from '{model_name(self.clip)}'. The file may not be a valid CLIP model, or the clip_type '{self.clip_type}' may be wrong."
                )
        if self.vae_model is None:
            raise RuntimeError(
                f"Failed to load a VAE from '{model_name(self.checkpoint if checkpoint else self.vae)}'."
            )

    def _collect_loras(self, node, out):
        if isinstance(node, lark.Tree):
            if node.data == "lora":
                name = str(node.children[0].children[0])
                strength = 1.0
                if len(node.children) > 1 and node.children[1] is not None:
                    strength = float(node.children[1])
                out.append((name, strength))
            else:
                for child in node.children:
                    self._collect_loras(child, out)

    def _parse_prompt_node(self, node, step, total):
        def _is_triggered(specifier, step, total):
            specifier = float(specifier)
            if specifier < 1.0:
                return specifier < (step + 1) / total
            return specifier < (step + 1)
            
        if isinstance(node, lark.Token):
            return str(node)
        if not isinstance(node, lark.Tree):
            return str(node)

        if node.data == "lora":
            return ""

        if node.data == "scheduled":
            if len(node.children) == 3:
                prompt_a = node.children[0]
                prompt_b = node.children[1]
            else:
                prompt_a = None
                prompt_b = node.children[0]
            specifier = node.children[-1].children[0]
            active = prompt_b if _is_triggered(specifier, step, total) else prompt_a
            return self._parse_prompt_node(active, step, total) if active is not None else ""

        return "".join(self._parse_prompt_node(child, step, total) for child in node.children)

    def _parse_prompt(self, prompt, steps):
        if not prompt or not prompt.strip():
            return ([{"prompt": "", "start_percent": 0.0, "end_percent": 1.0}], [])

        tree = prompt_grammar.parse(prompt)

        loras_raw = []
        self._collect_loras(tree, loras_raw)
        seen = set()
        loras = []
        for name, strength in loras_raw:
            if name not in seen:
                seen.add(name)
                loras.append((name, strength))

        schedule = []
        segment_start = 0
        segment_text = self._parse_prompt_node(tree, 0, steps).strip()
        for step in range(1, steps):
            text = self._parse_prompt_node(tree, step, steps).strip()
            if text != segment_text:
                schedule.append({
                    "prompt": segment_text,
                    "start_percent": segment_start / steps,
                    "end_percent": step / steps,
                })
                segment_start = step
                segment_text = text
        schedule.append({
            "prompt": segment_text,
            "start_percent": segment_start / steps,
            "end_percent": 1.0,
        })

        return schedule, loras

    def _encode_prompt_scheduled(self, clip, text, steps):
        """Parse prompt scheduling syntax and return a ComfyUI conditioning list
        with start_percent/end_percent set on each segment, plus the LoRA list."""
        schedule, loras = self._parse_prompt(text, steps)

        if len(schedule) == 1 and schedule[0]["start_percent"] == 0.0:
            # No scheduling — fast path, single encode.
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(schedule[0]["prompt"]))
            return cond, loras

        conditioning = []
        for seg in schedule:
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(seg["prompt"]))
            for entry in cond:
                n = [entry[0], entry[1].copy()]
                n[1]["start_percent"] = seg["start_percent"]
                n[1]["end_percent"]   = seg["end_percent"]
                conditioning.append(n)

        return conditioning, loras

    def _load_loras(self, loras):
        """Apply a list of (name, strength) LoRAs to clones of self.model and
        self.clip_model.  Returns the patched (model, clip) pair."""
        model = self.model
        clip  = self.clip_model

        for name, strength in loras:
            lora_path = None
            for candidate in folder_paths.get_filename_list("loras"):
                stem = os.path.splitext(candidate)[0]
                if stem == name or stem.endswith("/" + name):
                    lora_path = folder_paths.get_full_path("loras", candidate)
                    break
            if not lora_path:
                raise RuntimeError(f"LoRA not found: {name}")

            lora_sd, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora_sd, strength, strength, lora_metadata=lora_metadata
            )

        return model, clip

    def _load_controlnet(self, name):
        path = self._resolve_model_path(name)
        key = "controlnet:" + path
        cached = self._loaded.get(key)
        if cached:
            return cached[1]
        self.set_status("Loading ControlNet")
        keys = self._safetensors_keys(path)
        if keys is not None and "lllite_conditioning1.conv1.weight" in keys:
            # Anima LLLite: a "model patch", not a classic ControlNet
            controlnet = self._load_model_patch(name, keys)
        else:
            controlnet = comfy.controlnet.load_controlnet(path)
            if controlnet is None:
                controlnet = self._load_model_patch(name, keys)
        # stored as (path, obj) so _evict_other_models can iterate it like the others
        self._loaded[key] = (path, controlnet)
        return controlnet

    @staticmethod
    def _safetensors_keys(path):
        if not path.lower().endswith(".safetensors"):
            return None
        try:
            with safetensors.safe_open(path, framework="pt") as f:
                return list(f.keys())
        except Exception:
            return None

    def _load_model_patch(self, name, keys=None):
        """Load a file from models/controlnet as a ComfyUI model patch (ModelPatchLoader)."""
        import comfy_extras.nodes_model_patch as nodes_model_patch

        # ModelPatchLoader only looks in "model_patches": make it see the controlnet folders too
        registered = set(folder_paths.get_folder_paths("model_patches"))
        for folder in folder_paths.get_folder_paths("controlnet"):
            if folder not in registered:
                folder_paths.add_model_folder_path("model_patches", folder)

        bare = name.replace("\\", "/").split("/", 1)[-1]
        try:
            patch = nodes_model_patch.ModelPatchLoader().load_model_patch(bare)[0]
        except Exception as e:
            if keys and any(k.startswith("lllite") for k in keys) and "lllite_conditioning1.conv1.weight" not in keys:
                raise RuntimeError(
                    f"'{model_name(name)}' looks like a kohya ControlNet-LLLite for SD/SDXL. "
                    "ComfyUI core doesn't support that format (it needs a custom node), only classic "
                    "ControlNets and Anima LLLite."
                )
            if isinstance(e, UnboundLocalError):
                raise RuntimeError(
                    f"'{model_name(name)}' has a format this ComfyUI version can't load "
                    "(it is neither a ControlNet nor a known model patch)."
                )
            raise RuntimeError(f"'{model_name(name)}' is not a ControlNet and could not be loaded as a model patch: {e}")
        return patch

    def _apply_model_patch(self, model, model_patch, image, strength, start, end, mask=None):
        """Apply a model patch to the MODEL (same as ComfyUI's AnimaLLLiteApply node)."""
        import comfy.ldm.anima.lllite
        import comfy_extras.nodes_model_patch as nodes_model_patch

        if not isinstance(model_patch.model, comfy.ldm.anima.lllite.AnimaLLLite):
            raise RuntimeError("This model patch type isn't supported yet (only Anima LLLite is).")
        return nodes_model_patch.AnimaLLLiteApply().apply_patch(model, model_patch, image, strength, start, end, mask)[0]

    def _apply_controlnets(self, model, positive, negative, width, height, reference_size=None, extent=None, default_mask=None):
        """Apply every ControlNet unit in self.control.
        Classic ControlNets are attached to the conditioning (same logic as ComfyUI's
        ControlNetApplyAdvanced node); model patches (Anima LLLite) patch the MODEL.
        If reference_size/extent are given (inpainting crop), control images that match
        reference_size are cropped with the same extent so they stay aligned.
        Returns (model, positive, negative)."""
        units = self.control or []
        used = set()

        for unit in units:
            image = unit.get("image")
            name = unit.get("model")
            strength = float(unit.get("strength", 1.0))
            start = float(unit.get("start", 0.0))
            end = float(unit.get("end", 1.0))
            if image is None or not name or strength == 0:
                continue

            controlnet = self._load_controlnet(name)
            used.add("controlnet:" + self._resolve_model_path(name))

            if extent is not None and reference_size is not None and image.size == reference_size:
                image = image.crop(extent)
            mask = unit.get("mask")
            if mask is not None and extent is not None and reference_size is not None and mask.size == reference_size:
                mask = mask.crop(extent)
            image = self._upscale([image], width, height)

            if hasattr(controlnet, "model_patches_to"):
                # model patch (Anima LLLite): needs the IMAGE as (1,H,W,3) and an optional (1,H,W) mask
                import numpy as np
                image_t = self._pil_to_image_tensor(image)
                mask_t = None
                if mask is None and default_mask is not None and getattr(controlnet.model, "cond_in_channels", 3) == 4:
                    mask = default_mask  # 4-channel (inpainting) patch: use the request's mask (already cropped/scaled)
                if mask is not None:
                    mask = self._upscale_single(mask, transforms.InterpolationMode.NEAREST, width, height)
                    mask_t = torch.from_numpy(np.asarray(mask.convert("L")).astype(np.float32) / 255.0)[None]
                    mask_t = mask_t.to(device=image_t.device, dtype=image_t.dtype)
                model = self._apply_model_patch(model, controlnet, image_t, strength, start, end, mask_t)
                continue

            hint = self._pil_to_image_tensor(image).movedim(-1, 1)  # (1,3,H,W)

            cnets = {}
            out = []
            for conditioning in (positive, negative):
                c = []
                for t in conditioning:
                    d = t[1].copy()
                    prev = d.get("control", None)
                    if prev in cnets:
                        c_net = cnets[prev]
                    else:
                        c_net = controlnet.copy().set_cond_hint(
                            hint, strength, (start, end), vae=self.vae_model
                        )
                        c_net.set_previous_controlnet(prev)
                        cnets[prev] = c_net
                    d["control"] = c_net
                    d["control_apply_to_uncond"] = False
                    c.append([t[0], d])
                out.append(c)
            positive, negative = out

        # drop cached ControlNets this request didn't use (they can be big)
        for key in [k for k in self._loaded if k.startswith("controlnet:") and k not in used]:
            del self._loaded[key]

        return model, positive, negative

    def _run_upscale_model_once(self, upscale_model, image_tensor):
        """Run one pass of a spandrel upscale model on a (B,H,W,3) tensor.
        The model must already be on the target device. Returns a (B,H',W',3) tensor."""
        output_device = comfy.model_management.intermediate_device()
        device = comfy.model_management.get_torch_device()
        # tiled_scale returns its output on output_device, which may be CPU. Move
        # every pass back to the model device so repeated model upscaling works.
        in_img = image_tensor.movedim(-1, -3).to(device)  # (B,3,H,W)

        tile = 512
        overlap = 32
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2],
                    tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a.float()),
                    tile_x=tile, tile_y=tile, overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar, output_device=output_device,
                )
                oom = False
            except Exception as e:
                comfy.model_management.raise_non_oom(e)
                tile //= 2
                if tile < 128:
                    raise e

        return torch.clamp(s.movedim(-3, -1), min=0, max=1.0).to(comfy.model_management.intermediate_dtype())

    def _upscale_with_model(self, upscale_model, image_tensor, target_width=None, target_height=None):
        """Run an ESRGAN/spandrel upscale model on a ComfyUI IMAGE tensor (B,H,W,3).
        Loops until the output is large enough to cover target_width x target_height
        (matching the old sd-inference-server behaviour), then does a final Lanczos
        resize to the exact target. If no target is given, runs exactly one pass.
        Returns a list of PIL images.
        """
        device = comfy.model_management.get_torch_device()

        memory_required = comfy.model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image_tensor.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image_tensor.nelement() * image_tensor.element_size()
        comfy.model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        current = image_tensor.to(device)
        try:
            while True:
                prev_w = current.shape[2]  # (B,H,W,3) → W is dim 2
                current = self._run_upscale_model_once(upscale_model, current)
                if target_width is None or target_height is None:
                    break
                if current.shape[2] >= target_width and current.shape[1] >= target_height:
                    break
                if current.shape[2] <= prev_w:
                    raise RuntimeError(f"Upscaler is not upscaling ({prev_w} -> {current.shape[2]})")
        finally:
            upscale_model.to("cpu")

        pil_images = self._images_to_pil(current)

        if target_width and target_height:
            pil_images = self._upscale(pil_images, target_width, target_height)

        return pil_images

    def _empty_latent(self, width, height, batch_size=1):
        latent_format = self.model.get_model_object("latent_format")
        channels = latent_format.latent_channels
        ratio = latent_format.spacial_downscale_ratio
        dims = getattr(latent_format, "latent_dimensions", 2)

        if dims == 3:
            shape = [batch_size, channels, 1, height // ratio, width // ratio]
        else:
            shape = [batch_size, channels, height // ratio, width // ratio]

        latent = torch.zeros(
            shape,
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        return {"samples": latent, "downscale_ratio_spacial": ratio}

    def _ksampler(self, model, seed, steps, cfg, positive, negative, latent_image,
                  denoise=1.0, start_step=None, last_step=None, force_full_denoise=True, noise_mask=None):
        ds = latent_image.get("downscale_ratio_spacial", None) if isinstance(latent_image, dict) else None
        latent_image = latent_image["samples"] if isinstance(latent_image, dict) else latent_image
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, ds)

        noise = comfy.sample.prepare_noise(latent_image, seed, None)

        callback = self._sampler_callback
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model, noise, steps, cfg, self.sampler, self.scheduler,
            positive, negative, latent_image,
            denoise=denoise, disable_noise=False,
            start_step=start_step, last_step=last_step,
            force_full_denoise=force_full_denoise, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )
        return {"samples": samples}

    def _sampler_callback(self, step, x0, x, total_steps):
        now = time.perf_counter()
        rate = 0
        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            if elapsed > 0:
                rate = 1.0 / elapsed
        self._last_step_time = now

        self.on_step({"n": step, "rate": rate}, latents=x0)

    @staticmethod
    def _vae_decode(vae, samples):
        latent = samples["samples"]
        if hasattr(latent, "is_nested") and latent.is_nested:
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    @staticmethod
    def _vae_encode(vae, pixels):
        """Encode an IMAGE tensor (B, H, W, 3) in [0,1] into a latent (VAEEncode logic)."""
        t = vae.encode(pixels)
        return {"samples": t}

    def _pil_to_image_tensor(self, image):
        """Convert a PIL image (or list thereof) into a ComfyUI IMAGE tensor (B, H, W, 3) in [0,1]."""
        import numpy as np

        if not isinstance(image, list):
            image = [image]

        tensors = []
        for im in image:
            im = PIL.ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            arr = np.asarray(im).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr)[None,])

        tensor = torch.cat(tensors, dim=0)

        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()
        return tensor.to(device=device, dtype=dtype)

    def _images_to_pil(self, images):
        """Convert a ComfyUI IMAGE tensor (B, H, W, 3) in [0,1] to a list of PIL images."""
        images = images.cpu().float()
        images = torch.clamp(images, 0.0, 1.0)
        images = (images * 255.0).round().to(torch.uint8).numpy()
        return [PIL.Image.fromarray(images[i]) for i in range(images.shape[0])]

    @staticmethod
    def _upscale_single(img, mode, width, height):
        """Resize a single PIL image to fit (width, height), preserving aspect ratio then center-cropping."""
        rw = width / img.size[0]
        rh = height / img.size[1]
        if abs(rw - rh) < 0.01:
            z = min(width, height)
        elif rw > rh:
            z = width
        else:
            z = height

        resize = transforms.transforms.Resize(z, mode)
        img = resize(img)

        dx = int((img.size[0] - width) * 0.5)
        dy = int((img.size[1] - height) * 0.5)
        img = torchvision.transforms.functional.crop(img, dy, dx, height, width)
        return img

    def _upscale(self, inputs, width, height):
        return [self._upscale_single(inputs[i], transforms.InterpolationMode.LANCZOS, width, height) for i in range(len(inputs))]

    def _prepare_images(self, inputs, extents, width, height):
        """Crop inputs to extents then resize to (width, height) with Nearest (for masks)."""
        if not inputs:
            return inputs
        outputs = self._apply_extents(inputs, extents)
        outputs = [self._upscale_single(o, transforms.InterpolationMode.NEAREST, width, height) for o in outputs]
        return outputs

    @staticmethod
    def _prepare_mask(mask, blur, expand):
        if expand:
            mask = mask.filter(PIL.ImageFilter.MaxFilter(int(2 * expand + 1)))
        if blur:
            mask = mask.filter(PIL.ImageFilter.GaussianBlur(blur))
        return mask

    @staticmethod
    def _masks_to_noise_mask(masks, ratio, device=None, dtype=None):
        """Convert a list of PIL masks (white = regenerate) into a ComfyUI noise_mask
        tensor of shape (B, 1, H_lat, W_lat) in [0,1], where 1 = denoise, 0 = keep original.
        """
        import numpy as np

        tensors = []
        for m in masks:
            if m is None:
                continue
            w, h = m.size
            w, h = w - (w % ratio), h - (h % ratio)
            m = m.resize((w // ratio, h // ratio), resample=PIL.Image.LANCZOS)
            arr = np.asarray(m).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr)[None, None, :, :])  # (1, 1, H_lat, W_lat)

        if not tensors:
            return None

        mask = torch.cat(tensors, dim=0)
        if device is None:
            device = comfy.model_management.intermediate_device()
        if dtype is None:
            dtype = comfy.model_management.intermediate_dtype()
        return mask.to(device=device, dtype=dtype)

    @staticmethod
    def _get_extents(originals, masks, padding, width, height):
        """Compute crop extents (x1,y1,x2,y2) for each image based on mask bbox,
        padded and aspect-ratio matched to the target (width, height)."""
        def pad_extent(extent, p, src, wrk):
            wrk_w, wrk_h = wrk
            src_w, src_h = src

            if not extent:
                return 0, 0, src_w, src_h

            x1, y1, x2, y2 = extent

            ar = wrk_w / wrk_h
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            rw, rh = min(src_w, (x2 - x1) + p), min(src_h, (y2 - y1) + p)

            if wrk_w / rw < wrk_h / rh:
                w = rw
                h = int(w / ar)
                if h > src_h:
                    h = src_h
                    w = int(h * ar)
            else:
                h = rh
                w = int(h * ar)
                if w > src_w:
                    w = src_w
                    h = int(w / ar)

            x1 = cx - w // 2
            x2 = cx + w - (w // 2)

            if x1 < 0:
                x2 += -x1
                x1 = 0
            if x2 > src_w:
                x1 -= x2 - src_w
                x2 = src_w

            y1 = cy - h // 2
            y2 = cy + h - (h // 2)

            if y1 < 0:
                y2 += -y1
                y1 = 0
            if y2 > src_h:
                y1 -= y2 - src_h
                y2 = src_h

            return int(x1), int(y1), int(x2), int(y2)

        if not masks:
            return [(0, 0, o.size[0], o.size[1]) for o in originals]

        if padding is None:
            padding = 10240

        extents = []
        for i in range(len(masks)):
            if masks[i] is not None:
                extents.append(pad_extent(masks[i].getbbox(), padding, originals[i].size, (width, height)))
            else:
                extents.append((0, 0, originals[i].size[0], originals[i].size[1]))

        return extents

    @staticmethod
    def _apply_extents(inputs, extents):
        return [inputs[i].crop(extents[i]) for i in range(len(inputs))]

    @staticmethod
    def _apply_inpainting(images, originals, masks, extents):
        """Paste generated images back into the originals at the extent locations, masked."""
        outputs = [None] * len(images)
        masked = [None] * len(images)
        for i in range(len(images)):
            image = images[i]
            original = originals[i % len(originals)]
            mask = masks[i % len(masks)]
            extent = extents[i % len(extents)]
            x1, y1, x2, y2 = extent

            image = image.convert("RGBA")
            mask = mask.convert("L")

            masked[i] = image.copy()
            masked[i].putalpha(mask.resize(image.size))

            ew, eh = x2 - x1, y2 - y1
            if (ew, eh) != images[i].size:
                image = image.resize((ew, eh))
                mask = mask.resize((ew, eh))

            outputs[i] = original.copy()
            outputs[i].paste(image, extent, mask)
        return outputs, masked

    def _get_metadata(self, width, height, batch_size, seeds, mode="txt2img"):
        metadata = []
        for i in range(batch_size):
            m = {
                "mode": mode,
                "width": width,
                "height": height,
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "seed": seeds[i] if i < len(seeds) else seeds[-1],
                "steps": self.steps,
                "scale": format_float(self.scale),
                "sampler": self.sampler,
                "scheduler": self.scheduler
            }
            if self.checkpoint:
                m["model"] = model_name(self.checkpoint)
            else:
                m["UNET"] = model_name(self.unet)
                m["CLIP"] = model_name(self.clip)
                m["VAE"] = model_name(self.vae)
                m["clip_type"] = self.clip_type
            if mode == "img2img":
                m["strength"] = format_float(self.strength)
                if self.upscaler:
                    m["upscaler"] = model_name(self.upscaler)
            metadata.append(m)
        return metadata

    def _set_device(self):
        if not self.device_name or self.device_name == "CPU":
            self.device = torch.device("cpu")
            return

        if self.device_name in self.device_names:
            idx = self.device_names.index(self.device_name)
            if idx < torch.cuda.device_count():
                self.device = torch.device(idx)
                return

        self.device = comfy.model_management.get_torch_device()

    def _evict_other_models(self):
        keep_models = []
        for _, obj in self._loaded.values():
            if obj is None:
                continue
            patcher = obj if hasattr(obj, "model_patches_to") else getattr(obj, "patcher", None)
            if patcher is not None:
                keep_models.append(patcher)

        if not keep_models:
            comfy.model_management.unload_all_models()
            return

        keep_loaded = [comfy.model_management.LoadedModel(m) for m in keep_models]
        for device in comfy.model_management.get_all_torch_devices():
            comfy.model_management.free_memory(1e30, device, keep_loaded=keep_loaded)
        comfy.model_management.soft_empty_cache()

    @torch.inference_mode()
    def txt2img(self):
        self.set_status("Configuring")

        self._set_device()

        self.set_status("Loading")
        self._load_models()

        self.set_status("Encoding")
        positive, loras = self._encode_prompt_scheduled(self.clip_model, self.prompt, self.steps)
        negative, _     = self._encode_prompt_scheduled(self.clip_model, self.negative_prompt, self.steps)

        model, clip = self._load_loras(loras)
        self._evict_other_models()

        model, positive, negative = self._apply_controlnets(model, positive, negative, self.width, self.height)

        batch_size = 1
        seeds = [self.seed + i for i in range(batch_size)]

        latent = self._empty_latent(self.width, self.height, batch_size)

        self.current_step = 0
        self.total_steps = self.steps
        self._last_step_time = None

        metadata = self._get_metadata(self.width, self.height, batch_size, seeds)

        self.set_status("Generating")
        sampled = self._ksampler(
            model, self.seed, self.steps, self.scale,
            positive, negative, latent, denoise=1.0,
        )

        self.set_status("Decoding")
        images = self._vae_decode(self.vae_model, sampled)
        pil_images = self._images_to_pil(images)

        self.on_complete(pil_images, metadata)

        comfy.model_management.soft_empty_cache()

    @torch.inference_mode()
    def img2img(self):
        self.set_status("Configuring")

        self._set_device()

        self.set_status("Loading")
        self._load_models()

        self.set_status("Encoding")
        positive, loras = self._encode_prompt_scheduled(self.clip_model, self.prompt, self.steps)
        negative, _     = self._encode_prompt_scheduled(self.clip_model, self.negative_prompt, self.steps)

        model, clip = self._load_loras(loras)
        self._evict_other_models()

        self.set_status("Preparing")
        images = self.image if isinstance(self.image, list) else [self.image]
        masks = list(self.mask) if self.mask else []
        width, height = self.width, self.height

        for i in range(len(masks)):
            if masks[i]:
                masks[i] = self._upscale_single(masks[i], transforms.InterpolationMode.LANCZOS, images[i].width, images[i].height)

        extents = self._get_extents(images, masks, self.padding, width, height)

        for i in range(len(masks)):
            if masks[i] is None:
                masks[i] = PIL.Image.new("L", (images[i].width, images[i].height), color=255)

        original_images = images
        images = self._apply_extents(images, extents)
        masks = self._prepare_images(masks, extents, width, height)
        masks = [self._prepare_mask(mask, self.mask_blur, self.mask_expand) for mask in masks]

        batch_size = len(images)
        seeds = [self.seed + i for i in range(batch_size)]

        strength = max(0.0, min(1.0, float(self.strength)))
        start_step = int(self.steps * (1.0 - strength))
        last_step = self.steps

        self.current_step = 0
        self.total_steps = last_step - start_step
        self._last_step_time = None

        metadata = self._get_metadata(width, height, batch_size, seeds, mode="img2img")

        model, positive, negative = self._apply_controlnets(
            model, positive, negative, width, height,
            reference_size=original_images[0].size, extent=extents[0],
            default_mask=masks[0] if masks else None,
        )

        self.set_status("Upscaling")
        if self.upscaler:
            sr_model = self._load_upscale_model(self.upscaler)
            image_tensor = self._pil_to_image_tensor(images)
            upscaled_images = self._upscale_with_model(sr_model, image_tensor, width, height)
        else:
            upscaled_images = self._upscale(images, width, height)

        self.set_status("Encoding")
        pixels = self._pil_to_image_tensor(upscaled_images)
        latent = self._vae_encode(self.vae_model, pixels)

        noise_mask = None
        if self.mask:
            latent_format = self.model.get_model_object("latent_format")
            ratio = latent_format.spacial_downscale_ratio
            noise_mask = self._masks_to_noise_mask(masks, ratio)

        self.set_status("Generating")
        sampled = self._ksampler(
            model, self.seed, self.steps, self.scale,
            positive, negative, latent, denoise=1.0,
            start_step=start_step, last_step=last_step,
            force_full_denoise=True, noise_mask=noise_mask,
        )

        self.set_status("Decoding")
        out_images = self._vae_decode(self.vae_model, sampled)
        pil_images = self._images_to_pil(out_images)

        if self.mask:
            outputs, _masked = self._apply_inpainting(pil_images, original_images, masks, extents)
            pil_images = [outputs[i] if i < len(self.mask) and self.mask[i] else pil_images[i] for i in range(len(pil_images))]

        self.on_complete(pil_images, metadata)

        comfy.model_management.soft_empty_cache()

    @torch.inference_mode()
    def upscale(self):
        self.set_status("Configuring")
        self._set_device()

        sr_model = None
        if self.upscaler:
            self.set_status("Loading Upscaler")
            sr_model = self._load_upscale_model(self.upscaler)

        images = self.image if isinstance(self.image, list) else [self.image]
        batch_size = len(images)
        seeds = [self.seed + i if self.seed else i for i in range(batch_size)]

        metadata = []
        for i in range(batch_size):
            m = {"mode": "upscale", "seed": seeds[i]}
            if self.upscaler:
                m["upscaler"] = model_name(self.upscaler)
            if self.width:
                m["width"] = self.width
            if self.height:
                m["height"] = self.height
            metadata.append(m)

        self.set_status("Upscaling")
        if sr_model:
            image_tensor = self._pil_to_image_tensor(images)  # (B,H,W,3)
            pil_images = self._upscale_with_model(sr_model, image_tensor, self.width, self.height)
            if overscale > 1.0:
                pil_images = self._upscale(pil_images, self.width, self.height)
        else:
            pil_images = self._upscale(images, self.width, self.height)

        self.on_complete(pil_images, metadata)
        comfy.model_management.soft_empty_cache()

    def _model_file_path(self, relative_path):
        if not isinstance(relative_path, str) or not relative_path:
            return ""

        relative_path = relative_path.replace("\\", os.path.sep)
        if os.path.isabs(relative_path):
            return ""

        path = os.path.abspath(os.path.join(self.model_path, relative_path))
        if os.path.commonpath((self.model_path, path)) != self.model_path:
            return ""
        return path

    def metadata(self):
        self.set_status("Inspecting")

        model = self.model
        file = self._model_file_path(model)
        if not file:
            raise RuntimeError(f"Model '{model}' does not exist.")

        metadata = {}
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}

        if self.callback:
            if not self.callback({"type": "metadata", "data": {"model": model, "metadata": metadata}}):
                raise AbortError("Aborted")

    def manage(self):
        if self.operation == "unload":
            self.set_status("Unloading")
            self._loaded.clear()
            comfy.model_management.unload_all_models()
            gc.collect()

            if torch.cuda.is_available():
                for device_index in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
        elif self.operation == "modify":
            old_file = self._model_file_path(self.old_file)

            if not old_file or not os.path.isfile(old_file):
                self.set_status("Modifying")
                raise RuntimeError(f"File '{old_file}' does not exist.")

            if not self.new_file:
                self.set_status("Deleting")
                os.remove(old_file)
            else:
                self.set_status("Moving")
                new_file = self._model_file_path(self.new_file)
                if old_file != new_file:
                    os.makedirs(os.path.dirname(new_file), exist_ok=True)
                    shutil.move(old_file, new_file)

            self.options()

    def options(self):
        data = {}

        data["UNET"] = self._get_filenames("diffusion_models")
        data["CLIP"] = self._get_filenames("text_encoders")
        data["VAE"] = self._get_filenames("vae")
        data["LoRA"] = self._get_filenames("loras")
        data["upscaler"] = self._get_filenames("upscale_models")
        data["ControlNet"] = self._get_filenames("controlnet")

        data["clip_mode"] = [m.name.lower() for m in comfy.sd.CLIPType]

        data["sampler"] = list(comfy.samplers.SAMPLER_NAMES)
        data["scheduler"] = list(comfy.samplers.SCHEDULER_NAMES)

        data["device"] = self.device_names

        data["model_types"] = {}

        if self.callback:
            if not self.callback({"type": "options", "data": data}):
                raise AbortError("Aborted")
