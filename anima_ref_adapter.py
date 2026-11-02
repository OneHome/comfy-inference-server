"""Anima Reference Adapter (R2I), v2 asymmetric adapters, for the qComfy inference server.

Adapted from ElectronGirl/ComfyUI-Anima-Ref-Adapter (Apache-2.0, https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter).
The adapter weights are distributed separately (CircleStone Non-Commercial License):
https://huggingface.co/ElectronGirl/anima-ref-adapter

What was changed: the ComfyUI node classes were removed; only the one-node "Anima Ref Adapter" path for v2
("asym") adapters is kept (no pilot adapter, no style slot, no Ref Guider); the weights are not downloaded here
(the server does it) and `apply` takes a file path instead of a LoRA name. The attention / reference-stream code is
the original one.

How it works: the reference image goes through the VAE and a "reference stream" (self-attention only, timestep 0,
RoPE coordinates shifted past the target grid) that is run once per image; its per-block keys and values are
reused by every sampling step. The target stream attends to itself and to those keys/values. A rank-64 LoRA, active
only while the reference is attached, teaches the model to read it. Reference guidance runs as a post-CFG patch.
"""
from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.ldm.common_dit
import comfy.patcher_extension
import comfy.samplers
import comfy.utils

SINGLE_SLOTS = {"character": {"t": 1, "offset_h": 1.0, "offset_w": 0}}  # adapters trained before slots
OPTIONS_REF_KEY = "anima_ref_latents"  # transformer_options: reference latents
NO_REF_KEY = "anima_ref_off"  # transformer_options: run this call without the reference
WRAPPER_KEY = "anima_ref_adapter"
HOOKS_ATTR = "_anima_ref_adapter_hooks"  # on each hooked module: {adapter key: hook handle}
_adapters: dict[tuple, "_Adapter"] = {}  # adapter key -> loaded weights
_active = {"key": None}  # key of the adapter whose LoRA the hooks apply right now


def _adapter_key(path: str, strength: float) -> tuple:
    return os.path.abspath(path), os.path.getmtime(path), float(strength)


class _Adapter:
    """LoRA weights applied through forward hooks, only while this adapter's key is active.

    Hooks are tracked on the modules themselves and match by key, not by object identity, so a
    loader node that runs again (e.g. after the upstream model chain changed) keeps its LoRA working.
    """

    def __init__(self, path: str, strength: float):
        self.key = _adapter_key(path, strength)
        tensors = comfy.utils.load_torch_file(path, safe_load=True)
        self.weights = {}
        for key, down in tensors.items():
            if not key.endswith(".lora_down.weight"):
                continue
            name = key[len("diffusion_model."):-len(".lora_down.weight")]
            up = tensors[key.replace("lora_down", "lora_up")]
            alpha = float(tensors.get(key.replace("lora_down.weight", "alpha"), torch.tensor(down.shape[0])))
            self.weights[name] = [down, up, strength * alpha / down.shape[0]]
        if not self.weights:
            raise ValueError(f"{path} has no lora_down/lora_up weights")
        metadata = _read_metadata(path)
        self.slots = metadata.get("slots") or SINGLE_SLOTS
        self.asym = metadata.get("arch") == "asym"
        self.ref_lora = metadata.get("ref_lora", True)
        self.ref_cache: dict = {}  # reference fingerprint -> per-block (k, v), asym adapters only

    @classmethod
    def get(cls, path: str, strength: float) -> "_Adapter":
        key = _adapter_key(path, strength)
        if key not in _adapters:
            _adapters[key] = cls(path, strength)
        return _adapters[key]

    def install(self, diffusion_model) -> None:
        for name, entry in self.weights.items():
            module = diffusion_model.get_submodule(name)
            installed = getattr(module, HOOKS_ATTR, None)
            if installed is None:
                installed = {}
                setattr(module, HOOKS_ATTR, installed)
            if self.key in installed:
                continue

            def hook(module, inputs, output, entry=entry, key=self.key):
                if _active["key"] != key:
                    return output
                x = inputs[0]
                if entry[0].device != x.device or entry[0].dtype != x.dtype:
                    entry[0] = entry[0].to(device=x.device, dtype=x.dtype)
                    entry[1] = entry[1].to(device=x.device, dtype=x.dtype)
                return output + F.linear(F.linear(x, entry[0]), entry[1]) * entry[2]

            installed[self.key] = module.register_forward_hook(hook)


def _read_metadata(path: str) -> dict:
    try:
        with safe_open(path, "pt") as handle:
            return json.loads((handle.metadata() or {}).get("anima_ref_adapter", "{}"))
    except Exception:  # not a safetensors file, or no metadata: a single-reference adapter
        return {}


def _frame_positions(t_index, height, width, h0, w0, device):
    hh, ww = torch.meshgrid(torch.arange(height, device=device) + h0, torch.arange(width, device=device) + w0,
                            indexing="ij")
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def _rope_from_positions(pe, t, h, w):
    """VideoRopePosition3DEmb.generate_embeddings for explicit per-token (t, h, w) positions."""
    device = t.device
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _block_mods(block, emb, adaln):
    """(shift, scale, gate) [B, T, 1, 1, D] for self-attention, cross-attention and MLP (Block.forward)."""
    out = []
    for modulation in (block.adaln_modulation_self_attn, block.adaln_modulation_cross_attn, block.adaln_modulation_mlp):
        m = modulation(emb) + adaln if block.use_adaln_lora else modulation(emb)
        out.append(tuple(t[:, :, None, None, :] for t in m.chunk(3, dim=-1)))
    return out


def _norm(x, layer_norm, scale, shift):
    return layer_norm(x) * (1 + scale) + shift


def _asym_block(block, xt, xr, emb_t, emb_r, adaln_t, adaln_r, context, rope_t, rope_r, ref_kv=None,
                transformer_options=None):
    """Block.forward with asymmetric attention (train/anima_env.asym_block). Returns (xt, xr, (k_r, v_r))."""
    attn = block.self_attn
    k_r = v_r = None
    if xr is not None:
        sa, _, mlp = _block_mods(block, emb_r, adaln_r)
        residual_dtype, compute_dtype = xr.dtype, emb_r.dtype
        B, T, H, W, _ = xr.shape
        n = _norm(xr, block.layer_norm_self_attn, sa[1], sa[0]).to(compute_dtype).flatten(1, 3)
        q_r, k_r, v_r = attn.compute_qkv(n, None, rope_emb=rope_r)
        out = attn.compute_attention(q_r, k_r, v_r).reshape(B, T, H, W, -1)
        xr = torch.addcmul(xr, sa[2].to(residual_dtype), out.to(residual_dtype))
        n = _norm(xr, block.layer_norm_mlp, mlp[1], mlp[0]).to(compute_dtype)
        xr = torch.addcmul(xr, mlp[2].to(residual_dtype), block.mlp(n).to(residual_dtype))
    elif ref_kv is not None:
        k_r, v_r = ref_kv
    if xt is not None:
        options = transformer_options or {}
        sa, ca, mlp = _block_mods(block, emb_t, adaln_t)
        residual_dtype, compute_dtype = xt.dtype, emb_t.dtype
        B, T, H, W, _ = xt.shape
        n = _norm(xt, block.layer_norm_self_attn, sa[1], sa[0]).to(compute_dtype).flatten(1, 3)
        q_t, k_t, v_t = attn.compute_qkv(n, None, rope_emb=rope_t)
        n_q = q_t.shape[1]
        if k_r is not None:
            k_t, v_t = torch.cat([k_t, k_r.to(k_t)], dim=1), torch.cat([v_t, v_r.to(v_t)], dim=1)
        if k_t.shape[1] != n_q and "compute_attention" in attn.__dict__:
            # Another node replaced this block's attention (e.g. Anima Safe PAG, which blends each query's
            # output with its own value, so it assumes as many queries as keys). Pad the queries with zeros
            # to the key length and keep the target's rows: queries are independent, and a query's "own"
            # value is then exactly its target token's.
            q_t = torch.cat([q_t, q_t.new_zeros(q_t.shape[0], k_t.shape[1] - n_q, *q_t.shape[2:])], dim=1)
        out = attn.compute_attention(q_t, k_t, v_t, transformer_options=options)[:, :n_q].reshape(B, T, H, W, -1)
        xt = torch.addcmul(xt, sa[2].to(residual_dtype), out.to(residual_dtype))
        n = _norm(xt, block.layer_norm_cross_attn, ca[1], ca[0]).to(compute_dtype).flatten(1, 3)
        out = block.cross_attn(n, context, rope_emb=None, transformer_options=options).reshape(B, T, H, W, -1)
        xt = torch.addcmul(xt, ca[2].to(residual_dtype), out.to(residual_dtype))
        n = _norm(xt, block.layer_norm_mlp, mlp[1], mlp[0]).to(compute_dtype)
        xt = torch.addcmul(xt, mlp[2].to(residual_dtype), block.mlp(n).to(residual_dtype))
    return xt, xr, (k_r, v_r)


def _frame_embed(dm, latent, t_index, h0, w0, sigma):
    """Patch embedding, RoPE and timestep embedding of one frame [B, C, 1, H, W]."""
    x_emb, _, _ = dm.prepare_embedded_sequence(latent, fps=None, padding_mask=None)
    _, _, hp, wp, _ = x_emb.shape
    rope = _rope_from_positions(dm.pos_embedder, *_frame_positions(t_index, hp, wp, h0, w0, latent.device))
    # the timestep embedding sets the compute dtype, so take it before the residual stream goes to fp32
    emb, adaln = dm.t_embedder[1](dm.t_embedder[0](sigma.reshape(-1, 1).float()).to(x_emb.dtype))
    if x_emb.dtype == torch.float16:
        x_emb = x_emb.float()  # residual stream in fp32, as MiniTrainDIT._forward does for fp16
    return x_emb, rope.unsqueeze(1).unsqueeze(0), dm.t_embedding_norm(emb), adaln


def _encode_ref_asym(dm, ref, ref_offset):
    """Per-block reference keys/values: the reference stream run once."""
    xr, rope_r, emb_r, adaln_r = _frame_embed(dm, ref, 1, *ref_offset, torch.zeros(ref.shape[0], device=ref.device))
    cache = []
    for block in dm.blocks:
        _, xr, kv = _asym_block(block, None, xr, None, emb_r, None, adaln_r, None, None, rope_r)
        cache.append(kv)
    return cache


def _asym_forward(dm, x, timesteps, context, ref_kv, transformer_options):
    """MiniTrainDIT._forward for the target frame alone, attending to cached reference keys/values."""
    orig_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    sigma = timesteps.reshape(x.shape[0], -1)[:, 0]
    xt, rope_t, emb_t, adaln_t = _frame_embed(dm, x[:, :, :1], 0, 0, 0, sigma)
    batch = xt.shape[0]
    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        k, v = ref_kv[index]
        kv = (k.expand(batch, *k.shape[1:]), v.expand(batch, *v.shape[1:])) if k.shape[0] != batch else (k, v)
        xt, _, _ = _asym_block(block, xt, None, emb_t, None, adaln_t, None, context, rope_t, None, kv, transformer_options)
    out = dm.unpatchify(dm.final_layer(xt.to(context.dtype), emb_t, adaln_lora_B_T_3D=adaln_t))[:, :, :1]
    return out[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]


def _asym_ref_kv(adapter: _Adapter, dm, refs, x, offset_ratio):
    """Cached reference keys/values for this reference and target height (the RoPE shift depends on it)."""
    ref = max(refs, key=lambda r: r.shape[-1] * r.shape[-2])  # the full-size encode; any size works here
    ref = ref.unsqueeze(2) if ref.ndim == 4 else ref
    ref = comfy.ldm.common_dit.pad_to_patch_size(ref, (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial))
    if ref.shape[0] > 1 and torch.equal(ref, ref[:1].expand_as(ref)):
        ref = ref[:1]  # the same reference in every batch item (the usual case): encode it once
    hp = -(-x.shape[-2] // dm.patch_spatial)
    offset = (round(adapter.slots["character"]["offset_h"] * offset_ratio * hp), int(adapter.slots["character"].get("offset_w", 0)))
    key = (id(dm), tuple(ref.shape), offset, x.dtype, x.device, float(ref.float().sum()), float(ref.flatten()[::997].float().sum()))
    if key not in adapter.ref_cache:
        # ~1 GB of keys/values per reference at 1 MP, invisible to ComfyUI's memory manager: keep only the
        # current one (the cache is also emptied when each sampling run ends, see _make_outer_sample_wrapper).
        adapter.ref_cache.clear()
        previous = _active["key"]
        _active["key"] = adapter.key if adapter.ref_lora else None  # the reference stream with or without the LoRA
        try:
            adapter.ref_cache[key] = _encode_ref_asym(dm, ref.to(device=x.device, dtype=x.dtype), offset)
        finally:
            _active["key"] = previous
    return adapter.ref_cache[key]


def _make_outer_sample_wrapper(adapter: _Adapter):
    """Free the reference cache when a sampling run ends, so its VRAM is not held between runs or
    between the chains of a workflow (as comfy_extras/nodes_easycache.py does for its cache)."""
    def wrapper(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            adapter.ref_cache.clear()
    return wrapper


def _flat_dims(x):
    return tuple(range(1, x.ndim))


def _correct(guided, pos, correction, rescale_phi):
    """APG (drop the guidance component parallel to the conditional prediction) or CFG rescale, in x0 space."""
    if correction == "apg":
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        parallel = (update * unit).sum(dim=_flat_dims(pos), keepdim=True) * unit
        return pos + (update - parallel).to(pos.dtype)
    if correction == "rescale":
        ratio = pos.std(dim=_flat_dims(pos), keepdim=True) / guided.std(dim=_flat_dims(guided), keepdim=True)
        return rescale_phi * guided * ratio + (1 - rescale_phi) * guided
    return guided


def _in_window(model_sampling, sigma, start, end) -> bool:
    """Whether a step lies in [start, end] of the schedule (fractions of the denoising, as in ComfyUI's
    ControlNet start/end percent; sigma falls as sampling goes on)."""
    if start <= 0.0 and end >= 1.0:
        return True
    sigma = float(sigma.flatten()[0])
    return model_sampling.percent_to_sigma(end) - 1e-5 <= sigma <= model_sampling.percent_to_sigma(start) + 1e-5


def _make_wrapper(adapter: _Adapter, offset_ratio: float):
    """Diffusion-model wrapper, asymmetric (v2) path of the original wrapper."""
    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        options = kwargs.get("transformer_options", {})
        refs = [] if options.get(NO_REF_KEY) else options.get(OPTIONS_REF_KEY, [])
        if not refs:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        dm = executor.class_obj
        adapter.install(dm)  # no-op once hooked; covers a model reloaded since apply() ran
        with torch.no_grad():
            ref_kv = _asym_ref_kv(adapter, dm, refs, x, offset_ratio)
        previous = _active["key"]
        _active["key"] = adapter.key
        try:
            return _asym_forward(dm, x, timesteps, context, ref_kv, kwargs.get("transformer_options", {}))
        finally:
            _active["key"] = previous
    return wrapper


def apply(model, vae, image, adapter_path, ref_guidance=4.5, correction="apg", strength=1.0, ref_end=1.0,
          ref_megapixels=1.0):
    """MODEL + VAE + reference IMAGE ([B, H, W, 3] float tensor) + adapter file -> patched MODEL (a clone)."""
    adapter = _Adapter.get(adapter_path, strength)
    if not adapter.asym:
        raise RuntimeError("Only the v2 reference adapters (anima_ref_adapter_v2_*) are supported, not the pilot one.")
    adapter.install(model.get_model_object("diffusion_model"))

    pixels = image[:1].movedim(-1, 1)
    scale = (ref_megapixels * 1024 * 1024 / (pixels.shape[-1] * pixels.shape[-2])) ** 0.5
    width, height = (max(16, round(d * scale / 16) * 16) for d in (pixels.shape[-1], pixels.shape[-2]))
    process_latent_in = model.get_model_object("process_latent_in")
    # a half-size encode too (the original keeps it for samplers with a half-resolution stage)
    refs = [process_latent_in(vae.encode(comfy.utils.common_upscale(pixels, w, h, "lanczos", "disabled").movedim(1, -1)))
            for w, h in ((width, height), (width // 16 * 8, height // 16 * 8))]
    m = model.clone()
    m.model_options["transformer_options"][OPTIONS_REF_KEY] = refs
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, _make_wrapper(adapter, 1.0))
    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, _make_outer_sample_wrapper(adapter))

    def post_cfg(args):
        guided, pos = args["denoised"], args["cond_denoised"]
        if ref_guidance != 1.0 and _in_window(args["model"].model_sampling, args["sigma"], 0.0, ref_end):
            # reference effect = (a branch with the reference) - (the same branch without it); the uncond branch,
            # or the cond branch when cfg is 1 and there is no uncond
            cond, ref = (args["uncond"], args["uncond_denoised"]) if args["uncond"] is not None else (args["cond"], pos)
            options = args["model_options"].copy()
            options["transformer_options"] = {**options["transformer_options"], NO_REF_KEY: True}
            (base,) = comfy.samplers.calc_cond_batch(args["model"], [cond], args["input"], args["sigma"], options)
            guided = guided + (ref_guidance - 1.0) * (ref - base)
        return _correct(guided, pos, correction, 0.5)

    m.set_model_sampler_post_cfg_function(post_cfg)
    return m
