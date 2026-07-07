"""
Muse Seed Scout
================
Generates several cheap, low-resolution Stage-1-only candidates using Muse
Director V1's own Stage 1 recipe (same helper functions: prompt relay, guide
application, audio/lipsync mask) so you can preview a handful of seeds before
committing to a full Director run. Does not modify muse_director_v1.py — it
only imports its existing module-level helper functions.

Usage:
  1. Wire the same model/clip/vae/audio_vae + prompt + start image you'd give
     Director, plus a short duration_frames (this is a scouting preview, not
     the final clip).
  2. Run once with all four use_seed_N toggles off — this generates and caches
     the 4 candidates and purges VRAM afterward.
  3. Flip exactly one use_seed_N toggle on. Re-running the graph at this point
     does NOT regenerate the candidates (they're cached against every other
     input) — it just re-selects and outputs that candidate's seed instantly.
  4. Wire the `seed` output into Muse Director V1's `seed` widget — right-click
     Director and choose "Convert seed to input" first.
"""

import gc
import logging
import math

import torch

import folder_paths
import comfy.model_management as mm

from comfy_extras.nodes_custom_sampler import (
    CFGGuider, KSamplerSelect, BasicScheduler, RandomNoise, SamplerCustomAdvanced,
)
from comfy_extras.nodes_lt import (
    LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent,
)

from .muse_director_v1 import (
    _encode_relay, _apply_guide, _build_audio_latent, _crop_conditioning,
    _zero_out_conditioning, _unpack, _resize_image, _compress_image,
)

log = logging.getLogger(__name__)

# Keyed by node unique_id so multiple Scout nodes in one workflow don't collide.
_SCOUT_CACHE = {}


def _fp(x):
    """Cheap fingerprint for cache-key comparison — doesn't need to be a real
    hash, just needs to change when the input meaningfully changes."""
    if x is None:
        return None
    if torch.is_tensor(x):
        try:
            xf = x.float()
            return (tuple(x.shape), round(xf.mean().item(), 6), round(xf.std().item(), 6))
        except Exception:
            return (tuple(x.shape), id(x))
    return x


class MuseSeedScout:
    """Cheap 4-seed Stage-1-only scouting node for Muse Director V1."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model":     ("MODEL",),
                "clip":      ("CLIP",),
                "vae":       ("VAE",),
                "audio_vae": ("VAE",),

                "global_prompt":   ("STRING", {"multiline": True, "default": ""}),
                "local_prompts":   ("STRING", {"default": ""}),
                "segment_lengths": ("STRING", {"default": ""}),
                "epsilon": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),

                "frame_rate":      ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "duration_frames": ("INT",   {"default": 25, "min": 9, "max": 240, "step": 8,
                                              "tooltip": "Keep this short — it's a scouting preview, not the final clip."}),
                "custom_width":  ("INT", {"default": 960, "min": 64, "max": 4096, "step": 32}),
                "custom_height": ("INT", {"default": 544, "min": 64, "max": 4096, "step": 32}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "crop", "pad"],
                                  {"default": "maintain aspect ratio"}),
                "divisible_by":    ("INT", {"default": 32, "min": 1, "max": 256}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 51}),

                "generate_audio":  ("BOOLEAN", {"default": True}),
                "custom_audio_on": ("BOOLEAN", {"default": False}),
                "lipsync":         ("BOOLEAN", {"default": True}),

                "ic_lora_name":     (["None"] + loras, {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "stage1_steps": ("INT",   {"default": 8,   "min": 1, "max": 50}),
                "cfg":          ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),

                "guide_scale_by":            ("FLOAT", {"default": 0.5, "min": 0.01, "max": 8.0, "step": 0.01}),
                "guide_upscale_method":      (["bicubic", "bilinear", "nearest-exact", "area", "bislerp"], {"default": "bicubic"}),
                "guide_image_attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guide_crop":                (["center", "disabled"], {"default": "center"}),
                "guide_auto_snap_ic_grid":   ("BOOLEAN", {"default": True}),
                "guide_use_tiled_encode":    ("BOOLEAN", {"default": False}),
                "guide_tile_size":           ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "guide_tile_overlap":        ("INT", {"default": 64,  "min": 16, "max": 256, "step": 16}),

                "seed_1": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_2": ("INT", {"default": 2, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_3": ("INT", {"default": 3, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_4": ("INT", {"default": 4, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),

                "use_seed_1": ("BOOLEAN", {"default": False}),
                "use_seed_2": ("BOOLEAN", {"default": False}),
                "use_seed_3": ("BOOLEAN", {"default": False}),
                "use_seed_4": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "start_image":  ("IMAGE",),
                "custom_audio": ("AUDIO",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("seed", "preview_1", "preview_2", "preview_3", "preview_4")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"
    DESCRIPTION = (
        "Generates 4 cheap low-res Stage-1-only candidates (reusing Muse Director "
        "V1's own Stage 1 helper functions) so you can pick a seed before committing "
        "to a full Director run. Wire the seed output into Director's seed input "
        "(convert it to an input first). Flip exactly one use_seed_N toggle on to "
        "select a candidate — re-running after only a toggle change is instant, "
        "not a full regenerate."
    )

    def execute(
        self, model, clip, vae, audio_vae,
        global_prompt, local_prompts, segment_lengths, epsilon,
        frame_rate, duration_frames, custom_width, custom_height,
        resize_method, divisible_by, img_compression,
        generate_audio, custom_audio_on, lipsync,
        ic_lora_name, ic_lora_strength,
        stage1_steps, cfg,
        guide_scale_by, guide_upscale_method, guide_image_attn_strength,
        guide_crop, guide_auto_snap_ic_grid, guide_use_tiled_encode,
        guide_tile_size, guide_tile_overlap,
        seed_1, seed_2, seed_3, seed_4,
        use_seed_1, use_seed_2, use_seed_3, use_seed_4,
        start_image=None, custom_audio=None, unique_id=None,
    ):
        seeds = [int(seed_1), int(seed_2), int(seed_3), int(seed_4)]
        toggles = [use_seed_1, use_seed_2, use_seed_3, use_seed_4]

        gen_key = (
            id(model), id(clip), id(vae), id(audio_vae),
            global_prompt, local_prompts, segment_lengths, epsilon,
            frame_rate, duration_frames, custom_width, custom_height,
            resize_method, divisible_by, img_compression,
            generate_audio, custom_audio_on, lipsync,
            ic_lora_name, ic_lora_strength, stage1_steps, cfg,
            guide_scale_by, guide_upscale_method, guide_image_attn_strength,
            guide_crop, guide_auto_snap_ic_grid, guide_use_tiled_encode,
            guide_tile_size, guide_tile_overlap,
            tuple(seeds),
            _fp(start_image),
            _fp(custom_audio.get("waveform")) if (custom_audio_on and custom_audio) else None,
        )

        cached = _SCOUT_CACHE.get(unique_id)
        if cached is not None and cached["gen_key"] == gen_key:
            log.info("[MuseSeedScout] Cache hit — reusing cached candidates, no regeneration.")
            frames = cached["frames"]
        else:
            log.info("[MuseSeedScout] Generating 4 fresh candidates (seeds=%s)...", seeds)
            frames = self._generate_candidates(
                model, clip, vae, audio_vae,
                global_prompt, local_prompts, segment_lengths, epsilon,
                frame_rate, duration_frames, custom_width, custom_height,
                resize_method, divisible_by, img_compression,
                generate_audio, custom_audio_on, lipsync,
                ic_lora_name, ic_lora_strength,
                stage1_steps, cfg,
                guide_scale_by, guide_upscale_method, guide_image_attn_strength,
                guide_crop, guide_auto_snap_ic_grid, guide_use_tiled_encode,
                guide_tile_size, guide_tile_overlap,
                seeds, start_image, custom_audio,
            )
            _SCOUT_CACHE[unique_id] = {"gen_key": gen_key, "frames": frames}
            mm.soft_empty_cache(force=True)
            gc.collect()
            log.info("[MuseSeedScout] VRAM purged after scouting pass.")

        picked = [i for i, t in enumerate(toggles) if t]
        if len(picked) == 0:
            chosen_idx = 0
            log.info("[MuseSeedScout] No seed selected yet — defaulting to seed_1. "
                      "Flip a use_seed_N toggle on and re-run to lock one in.")
        else:
            if len(picked) > 1:
                log.warning("[MuseSeedScout] More than one use_seed_N toggle is on — using the first: seed_%d",
                            picked[0] + 1)
            chosen_idx = picked[0]

        chosen_seed = seeds[chosen_idx]
        log.info("[MuseSeedScout] Output seed = %d (candidate %d)", chosen_seed, chosen_idx + 1)
        return (chosen_seed, frames[0], frames[1], frames[2], frames[3])

    def _generate_candidates(
        self, model, clip, vae, audio_vae,
        global_prompt, local_prompts, segment_lengths, epsilon,
        frame_rate, duration_frames, custom_width, custom_height,
        resize_method, divisible_by, img_compression,
        generate_audio, custom_audio_on, lipsync,
        ic_lora_name, ic_lora_strength,
        stage1_steps, cfg,
        guide_scale_by, guide_upscale_method, guide_image_attn_strength,
        guide_crop, guide_auto_snap_ic_grid, guide_use_tiled_encode,
        guide_tile_size, guide_tile_overlap,
        seeds, start_image, custom_audio,
    ):
        # Same halving convention as Director's own Stage 1 (muse_director_v1.py).
        s1_w = max(divisible_by, (custom_width // 2 // divisible_by) * divisible_by)
        s1_h = max(divisible_by, (custom_height // 2 // divisible_by) * divisible_by)

        ltxv_len = int(math.ceil((duration_frames - 1) / 8.0) * 8) + 1
        latent_t = ((ltxv_len - 1) // 8) + 1

        pre_latent = {"samples": torch.zeros(
            [1, 128, latent_t, s1_h // 32, s1_w // 32],
            device=mm.intermediate_device(),
        )}

        # ── Guide data: single start image, no full timeline ────────────────
        guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": frame_rate}
        if start_image is not None:
            tensor = _resize_image(start_image, s1_w, s1_h, resize_method, divisible_by)
            if img_compression > 0:
                tensor = _compress_image(tensor, img_compression)
            guide_data["images"].append(tensor)
            guide_data["insert_frames"].append(0)
            guide_data["strengths"].append(1.0)
        else:
            tensor = torch.zeros((1, s1_h, s1_w, 3), dtype=torch.float32)
            guide_data["images"].append(tensor)
            guide_data["insert_frames"].append(0)
            guide_data["strengths"].append(0.0)

        # ── Conditioning + guide are seed-independent — build once ──────────
        patched_model, positive = _encode_relay(
            model, clip, pre_latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )
        zero_neg = _zero_out_conditioning(positive)
        cond_pos, cond_neg = _unpack(LTXVConditioning.execute(positive, zero_neg, frame_rate))

        pos1, neg1, lat1 = cond_pos, cond_neg, pre_latent
        try:
            pos1, neg1, lat1, patched_model = _apply_guide(
                cond_pos, cond_neg, vae, pre_latent, guide_data, patched_model,
                motion_guide_data=None,
                ic_lora_name=ic_lora_name, ic_lora_strength=ic_lora_strength,
                scale_by=guide_scale_by, upscale_method=guide_upscale_method,
                image_attention_strength=guide_image_attn_strength,
                crop=guide_crop, auto_snap_ic_grid=guide_auto_snap_ic_grid,
                use_tiled_encode=guide_use_tiled_encode,
                tile_size=guide_tile_size, tile_overlap=guide_tile_overlap,
            )
        except Exception as exc:
            log.warning("[MuseSeedScout] Guide application failed, using plain conditioning: %s", exc)
            pos1, neg1 = cond_pos, cond_neg
            lat1 = pre_latent

        # ── Audio latent — also seed-independent — build once ───────────────
        audio_out = None
        effective_custom_audio_on = custom_audio_on
        if custom_audio_on and custom_audio is None:
            log.warning("[MuseSeedScout] custom_audio_on is True but no audio was wired in — "
                        "falling back to generated/empty audio for this scouting pass.")
            effective_custom_audio_on = False
        elif custom_audio_on:
            audio_out = custom_audio

        audio_latent = _build_audio_latent(
            audio_vae, audio_out, ltxv_len, frame_rate,
            effective_custom_audio_on, generate_audio,
        )
        if "samples" in audio_latent:
            s = audio_latent["samples"]
            if effective_custom_audio_on and lipsync:
                mask = torch.zeros(s.shape[0], s.shape[2], s.shape[3], dtype=torch.float32, device=s.device)
            else:
                mask = torch.ones(s.shape[0], s.shape[2], s.shape[3], dtype=torch.float32, device=s.device)
            audio_latent = {**audio_latent, "noise_mask": mask}
        has_audio = "samples" in audio_latent

        guider1 = _unpack(CFGGuider.execute(patched_model, pos1, neg1, cfg))[0]
        sampler = _unpack(KSamplerSelect.execute("euler"))[0]
        sigmas1 = _unpack(BasicScheduler.execute(patched_model, "linear_quadratic", stage1_steps, 1.0))[0]

        if has_audio:
            av1_lat = _unpack(LTXVConcatAVLatent.execute(lat1, audio_latent))[0]
        else:
            av1_lat = lat1

        frames = []
        for i, sd in enumerate(seeds):
            # Clone per-candidate so no sampler call can mutate state shared
            # with the next one.
            # Reused as-is across candidates — Director's own real Stage 1 uses
            # the same pattern (no defensive clone; av1_lat["samples"] can be a
            # NestedTensor from LTXVConcatAVLatent, which has no .clone()).
            noise1 = _unpack(RandomNoise.execute(int(sd)))[0]
            out1, _ = _unpack(SamplerCustomAdvanced.execute(noise1, guider1, sampler, sigmas1, av1_lat))
            if has_audio:
                out1_vid, _aud1 = _unpack(LTXVSeparateAVLatent.execute(out1))
            else:
                out1_vid = out1
            _pos1_c, _neg1_c, vid1 = _crop_conditioning(pos1, neg1, out1_vid)

            decoded = vae.decode(vid1["samples"])
            if decoded.ndim == 5:
                decoded = decoded.squeeze(0)
            if decoded.ndim == 3:
                decoded = decoded.unsqueeze(0)
            frames.append(decoded.cpu())
            log.info("[MuseSeedScout] Candidate %d/4 done (seed=%d, %d frames).", i + 1, sd, decoded.shape[0])

        return frames


NODE_CLASS_MAPPINGS = {
    "MuseSeedScout": MuseSeedScout,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseSeedScout": "Muse Seed Scout",
}
