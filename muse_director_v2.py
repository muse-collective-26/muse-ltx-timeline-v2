"""
Muse Collective LTX Timeline V2
================================
Fork of V1 with an added Seed Hunt feature — everything else is identical
to muse_director_v1.py (V1 is left completely untouched).

Seed Hunt:
  seed_hunt toggle ON + all use_seed_hunt_N toggles OFF
    -> runs a Stage-1-only pass at Stage 1's real resolution, 4x with 4
       different seeds (seed_hunt_1..4), using the real timeline's own
       guide/audio data so multi-segment timelines just work. Returns 4
       preview clips on the seed_hunt_preview_1..4 outputs (normal outputs
       are placeholders) AND caches each candidate's actual Stage 1 latent
       in memory (_SEED_HUNT_CANDIDATES), keyed by candidate index.
  seed_hunt toggle ON + exactly one use_seed_hunt_N toggle ON
    -> commits to that candidate. Chunk 1 skips a fresh Stage 1 sampler
       pass entirely and instead carries the cached candidate's real
       Stage 1 output straight into Stage 2 (upscale + partial refine) —
       matching Tristan's original ltx23SeedHunter workflow: the seed
       number never has to "match" across resolutions, because Stage 2 is
       refining the actual picked content, not regenerating from scratch.
       If the cache is empty (e.g. server restarted since scouting), falls
       back to overriding `seed` and running a fresh Stage 1 as before.
  seed_hunt toggle OFF
    -> behaves exactly like V1.

Architecture mirrors V7 (reference-frame latent extension for chunk continuity)
with a clean Muse-branded timeline UI and three audio toggles:
  generate_audio  — LTX generates ambient/sfx from [SOUNDS] prompts
  custom_audio_on — use audio file(s) from the AUDIO timeline track
  motion_guide_on — use motion guide segments from the timeline

Both audio modes can be active simultaneously; their waveforms are mixed.
"""

import gc
import json
import logging
import math
import os
import random
import base64
import io as _io

import numpy as np
import torch
import torch.nn.functional as F
import av
from PIL import Image

import folder_paths
import comfy.model_management as mm
import node_helpers

from comfy_extras.nodes_custom_sampler import (
    CFGGuider, KSamplerSelect, BasicScheduler, RandomNoise, SamplerCustomAdvanced,
)
from comfy_extras.nodes_lt import (
    LTXVConditioning, LTXVConcatAVLatent, LTXVSeparateAVLatent,
)
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

from .muse_prompt_relay import (
    get_raw_tokenizer, map_token_indices, build_segments,
    create_mask_fn, distribute_segment_lengths, convert_to_latent_lengths,
)
from .muse_patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)


# ── Media helpers ─────────────────────────────────────────────────────────────

def _load_image_tensor(seg: dict) -> torch.Tensor:
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?") or b64_str.startswith("/api/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


def _load_video_tensor(seg: dict, frame_rate: float) -> torch.Tensor:
    file_path = _resolve_path(seg.get("imageFile", ""))
    if not file_path or not os.path.exists(file_path):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    trim_start_frames = float(seg.get("trimStart", 0))
    length_frames = float(seg.get("length", 1))
    start_sec = trim_start_frames / frame_rate
    frames = []
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if stream.time_base:
                seek_pts = int((max(0, start_sec - 0.5)) / float(stream.time_base))
            else:
                seek_pts = int((max(0, start_sec - 0.5)) * av.time_base)
            container.seek(seek_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None and stream.time_base:
                    frame_time = float(frame.pts * stream.time_base)
                if frame_time is None:
                    frame_time = 0.0
                if frame_time < start_sec - 0.01:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= int(length_frames):
                    break
    except Exception as exc:
        log.warning("[MuseDirector] Video extract error: %s", exc)
    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)


def _resolve_path(rel: str) -> str:
    """Try input dir, then muse subdir, then whatdreamscost subdir."""
    if not rel:
        return ""
    input_dir = folder_paths.get_input_directory()
    for base in [input_dir,
                 os.path.join(input_dir, "muse"),
                 os.path.join(input_dir, "whatdreamscost")]:
        p = os.path.join(base, os.path.basename(rel))
        if os.path.exists(p):
            return p
    p = os.path.join(input_dir, rel)
    return p if os.path.exists(p) else ""


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int,
                  method: str, divisible_by: int) -> torch.Tensor:
    def snap(val, div):
        return max(div, (val // div) * div)
    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)
    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor
    t_nchw = tensor.permute(0, 3, 1, 2)
    if method == "stretch to fit":
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)
    elif method == "maintain aspect ratio":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        resized = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    elif method in ("pad", "pad green"):
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_l = (tw - new_w) // 2
        pad_t = (th - new_h) // 2
        if method == "pad green":
            resized = torch.zeros((N, C, th, tw), dtype=t_nchw.dtype, device=t_nchw.device)
            resized[:, 0, :, :] = 102 / 255.0
            resized[:, 1, :, :] = 1.0
            resized[:, :, pad_t:pad_t+new_h, pad_l:pad_l+new_w] = inner
        else:
            resized = F.pad(inner, (pad_l, tw - new_w - pad_l, pad_t, th - new_h - pad_t), value=0)
    elif method == "crop":
        ratio = max(tw / W, th / H)
        new_w = int(W * ratio)
        new_h = int(H * ratio)
        inner = F.interpolate(t_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner[:, :, top:top+th, left:left+tw]
    else:
        resized = F.interpolate(t_nchw, size=(th, tw), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    if crf == 0:
        return tensor
    N, H, W, C = tensor.shape
    h = (H // 2) * 2
    w = (W // 2) * 2
    tensor_bytes = (tensor[:, :h, :w, :] * 255.0).byte().cpu().numpy()
    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        for i in range(N):
            frame = av.VideoFrame.from_ndarray(tensor_bytes[i], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()
        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = [f.to_ndarray(format="rgb24") for f in container_r.decode(video=0)]
        container_r.close()
        if not decoded:
            return tensor
        decoded_np = np.stack(decoded).astype(np.float32) / 255.0
        out = tensor.clone()
        dec_N = min(N, len(decoded))
        out[:dec_N, :h, :w] = torch.from_numpy(decoded_np[:dec_N]).to(tensor.device, tensor.dtype)
        return out
    except Exception as exc:
        log.warning("[MuseDirector] img_compression failed: %s", exc)
        return tensor


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _build_combined_audio(timeline_data_str: str, start_frame: int,
                          duration_frames: int, frame_rate: float) -> dict:
    """Load and mix audio segments from timeline JSON into a single waveform."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32),
                   "sample_rate": target_sr}
    if not timeline_data_str:
        return empty_audio
    try:
        data = json.loads(timeline_data_str)
        audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio
    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        af = seg.get("audioFile", "")
        if af:
            file_path = _resolve_path(af)
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        if not buffer and seg.get("audioB64"):
            b64 = seg["audioB64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                buffer = _io.BytesIO(base64.b64decode(b64))
            except Exception:
                pass
        if not buffer:
            continue
        try:
            clip_frames = []
            with av.open(buffer) as container:
                if not container.streams.audio:
                    continue
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
                for frame in container.decode(stream):
                    for rf in resampler.resample(frame):
                        clip_frames.append(torch.from_numpy(rf.to_ndarray()))
                for rf in resampler.resample(None):
                    clip_frames.append(torch.from_numpy(rf.to_ndarray()))
            if not clip_frames:
                continue
            waveform = torch.cat(clip_frames, dim=1)  # [2, samples]
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))
            if start_frames + length_frames <= start_frame:
                continue
            offset = max(0, start_frame - start_frames)
            trim_start_frames += offset
            length_frames = max(1, length_frames - offset)
            start_frames = max(0, start_frames - start_frame)
            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = min(start_sample_src + length_samples, waveform.shape[1])
            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0:
                continue
            clip_waveform = waveform[:, start_sample_src:end_sample_src]
            start_sample_dst = int(start_frames / frame_rate * target_sr)
            if start_sample_dst >= out_waveform.shape[1]:
                continue
            end_sample_dst = start_sample_dst + actual_length
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
            if actual_length <= 0:
                continue
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform
        except Exception as exc:
            log.warning("[MuseDirector] Audio segment error: %s", exc)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


# ── Conditioning ──────────────────────────────────────────────────────────────

def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    locals_list = [p.strip() for p in local_prompts.split("|")]
    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        log.info("[MuseDirector] No local segments — using global prompt exclusively.")
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(global_prompt))
        return model.clone(), conditioning
    for i, p in enumerate(locals_list):
        if not p:
            locals_list[i] = global_prompt.strip() or "video"
    arch, patch_size, temporal_stride = detect_model_type(model)
    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])
    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)
    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)
    log.info("[MuseDirector] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[MuseDirector] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
    log.info("[MuseDirector] Latent: %d frames, %d tokens/frame, segments: %s",
             latent_frames, tokens_per_frame, effective_lengths)
    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)
    patched = model.clone()
    apply_patches(patched, arch, mask_fn)
    return patched, conditioning


# ── Keyframe / guide logic ────────────────────────────────────────────────────

def _get_guide_attention_entries(conditioning):
    for item in conditioning:
        entries = item[1].get("guide_attention_entries", None)
        if entries is not None:
            return entries
    return []


def _set_guide_attention_entries(conditioning, entries):
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": entries}
    )


def _append_guide_attention_entry(conditioning, pre_filter_count, latent_shape,
                                   attention_strength=1.0, attention_mask=None):
    entries = [*_get_guide_attention_entries(conditioning)]
    entries.append({
        "pre_filter_count": int(pre_filter_count),
        "strength": float(attention_strength),
        "pixel_mask": attention_mask,
        "latent_shape": list(latent_shape),
    })
    return _set_guide_attention_entries(conditioning, entries)


def _build_guide_data(tdata: dict, start_frame: int, duration_frames: int,
                      frame_rate: float, custom_width: int, custom_height: int,
                      resize_method: str, divisible_by: int, img_compression: int,
                      guide_strength_str: str):
    """Parse image/video segments from timeline_data into guide_data dict."""
    guide_data = {"images": [], "insert_frames": [], "strengths": [], "frame_rate": frame_rate}
    derived_w, derived_h = custom_width, custom_height

    img_segs = [
        s for s in tdata.get("segments", [])
        if s.get("type", "image") in ("image", "video")
        and (s.get("imageFile") or s.get("imageB64"))
        and int(s.get("start", 0)) < start_frame + duration_frames
        and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
    ]
    img_segs.sort(key=lambda s: s["start"])

    strengths = []
    if guide_strength_str.strip():
        strengths = [float(x.strip()) for x in guide_strength_str.split(",") if x.strip()]

    def snap(val, div):
        return max(div, (val // div) * div)

    for idx, seg in enumerate(img_segs):
        seg_start = int(seg.get("start", 0))
        offset = max(0, start_frame - seg_start)
        if seg.get("type") == "video":
            if offset > 0:
                seg["trimStart"] = float(seg.get("trimStart", 0)) + offset
                seg["length"] = max(1, int(seg.get("length", 1)) - offset)
            tensor = _load_video_tensor(seg, float(frame_rate))
        else:
            tensor = _load_image_tensor(seg)

        src_h, src_w = tensor.shape[1], tensor.shape[2]
        if custom_width > 0 and custom_height > 0:
            tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
        elif custom_width > 0:
            tgt_w = snap(custom_width, divisible_by)
            tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        elif custom_height > 0:
            tgt_h = snap(custom_height, divisible_by)
            tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        else:
            tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)

        if img_compression > 0:
            tensor = _compress_image(tensor, img_compression)

        if idx == 0:
            derived_h = tensor.shape[1]
            derived_w = tensor.shape[2]

        insert_frame = max(0, seg_start - start_frame)
        if seg.get("isEndFrame"):
            insert_frame = max(0, seg_start + int(seg.get("length", 1)) - 1 - start_frame)
        strength = strengths[idx] if idx < len(strengths) else 1.0
        guide_data["images"].append(tensor)
        guide_data["insert_frames"].append(insert_frame)
        guide_data["strengths"].append(float(strength))

    if not guide_data["images"]:
        src_w = derived_w if derived_w > 0 else 768
        src_h = derived_h if derived_h > 0 else 512
        tensor = torch.zeros((1, src_h, src_w, 3), dtype=torch.float32)
        if custom_width > 0 and custom_height > 0:
            tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)
        elif custom_width > 0:
            tgt_w = snap(custom_width, divisible_by)
            tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        elif custom_height > 0:
            tgt_h = snap(custom_height, divisible_by)
            tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
            tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by)
        else:
            tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by)
        guide_data["images"].append(tensor)
        guide_data["insert_frames"].append(0)
        guide_data["strengths"].append(0.0)
        derived_w = tensor.shape[2]
        derived_h = tensor.shape[1]

    return guide_data, derived_w, derived_h


def _build_chunk_local_prompts(tdata: dict, start_frame: int, duration_frames: int,
                                fallback_local_prompts: str, fallback_segment_lengths: str):
    """Scope local_prompts/segment_lengths down to only the segments overlapping this
    chunk's (or seed-hunt scouting window's) [start_frame, start_frame+duration_frames)
    range, with lengths clipped to the overlap. Mirrors the same overlap filter
    _build_guide_data already uses for image guides — without this, every chunk (and
    the seed-hunt scouting pass) was being handed all segments across the FULL timeline
    and squeezing them proportionally into its own (smaller) window, causing later
    segments (e.g. a bench-sit near the end) to bleed into early chunks/scouting."""
    segs = [
        s for s in tdata.get("segments", [])
        if int(s.get("start", 0)) < start_frame + duration_frames
        and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
    ]
    segs.sort(key=lambda s: s["start"])

    if not segs:
        return fallback_local_prompts, fallback_segment_lengths

    prompts = []
    lengths = []
    for s in segs:
        seg_start = int(s.get("start", 0))
        seg_len = int(s.get("length", 1))
        overlap_start = max(seg_start, start_frame)
        overlap_end = min(seg_start + seg_len, start_frame + duration_frames)
        clipped_len = max(1, overlap_end - overlap_start)
        prompts.append(str(s.get("prompt", "")).strip())
        lengths.append(str(clipped_len))

    return "|".join(prompts), ",".join(lengths)


def _build_motion_guide_data(timeline_data: str, start_frame: int, duration_frames: int,
                             frame_rate: float, resize_method: str, motion_guide_on: bool):
    """Parse IC-LoRA Video track segments ('motionSegments' in the timeline JSON)
    into motion_guide_data for MuseGuide. Ported from the original WhatDreamsCost
    LTXDirector node (ltx_director.py) — this construction step was never carried
    over when muse_director_v1.py was rewritten standalone."""
    motion_guide_data = {"segments": [], "frame_rate": float(frame_rate),
                          "duration_frames": int(duration_frames), "resize_method": resize_method}
    try:
        tdata = json.loads(timeline_data) if timeline_data else {}
        motion_segments = tdata.get("motionSegments", []) if motion_guide_on else []
        for seg in motion_segments:
            seg_start = int(seg.get("start", 0))
            length = int(seg.get("length", 1))
            if seg_start >= start_frame + duration_frames or seg_start + length <= start_frame:
                continue
            if not seg.get("videoFile"):
                continue
            offset = max(0, start_frame - seg_start)
            new_start = max(0, seg_start - start_frame)
            clipped_len = min(length - offset, duration_frames - new_start)
            if clipped_len <= 0:
                continue
            clean = dict(seg)
            clean["start"] = new_start
            clean["length"] = clipped_len
            clean["trimStart"] = float(seg.get("trimStart", 0)) + offset
            motion_guide_data["segments"].append(clean)
    except Exception as e:
        log.warning("[MuseDirector] Could not build motion_guide_data: %s", e)
    return motion_guide_data


def _apply_guide(pos, neg, vae, video_latent, guide_data, model,
                 motion_guide_data=None, ic_lora_name="None", ic_lora_strength=1.0,
                 scale_by=0.5, upscale_method="bicubic", image_attention_strength=1.0,
                 crop="center", auto_snap_ic_grid=True, use_tiled_encode=False,
                 tile_size=256, tile_overlap=64):
    """Apply guide data using MuseGuide (no WDC dependency)."""
    from .muse_guide import MuseGuide

    images = (guide_data or {}).get("images", [])
    if not images and not (motion_guide_data and motion_guide_data.get("segments")):
        return pos, neg, video_latent, model

    try:
        result = MuseGuide.execute(
            positive=pos,
            negative=neg,
            vae=vae,
            latent=video_latent,
            guide_data=guide_data,
            motion_guide_data=motion_guide_data,
            model=model,
            ic_lora_name=ic_lora_name,
            ic_lora_strength=ic_lora_strength,
            scale_by=scale_by,
            upscale_method=upscale_method,
            image_attention_strength=image_attention_strength,
            crop=crop,
            auto_snap_ic_grid=auto_snap_ic_grid,
            use_tiled_encode=use_tiled_encode,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        pos_out, neg_out, lat_out, model_out, _ = result
        return pos_out, neg_out, lat_out, model_out
    except Exception as exc:
        log.warning("[MuseDirector] Guide application failed: %s", exc)
        return pos, neg, video_latent, model


def _crop_conditioning(pos, neg, latent):
    """Trim guide keyframes from conditioning + latent after sampling."""
    from .muse_guide import MuseCropGuides
    try:
        result = MuseCropGuides().execute(pos, neg, latent)
        if result and len(result) >= 3:
            return result[0], result[1], result[2]
    except Exception as exc:
        log.warning("[MuseDirector] CropGuides failed: %s", exc)
    return pos, neg, latent


# ── Audio latent builder ──────────────────────────────────────────────────────

def _build_audio_latent(audio_vae, audio_out, ltxv_length, frame_rate,
                        custom_audio_on, generate_audio):
    """Encode audio waveform into latent space and build noise mask."""
    if audio_vae is None:
        return {}
    inner = getattr(audio_vae, "first_stage_model", audio_vae)
    z_channels = audio_vae.latent_channels
    audio_freq = inner.latent_frequency_bins
    num_audio_latents = inner.num_of_latents_from_frames(ltxv_length, float(frame_rate))

    if not custom_audio_on:
        # Generate from scratch — empty latent, all-ones mask set later
        audio_latents = torch.zeros(
            (1, z_channels, num_audio_latents, audio_freq),
            device=mm.intermediate_device(),
        )
        return {"samples": audio_latents, "type": "audio"}

    # Encode custom audio waveform
    waveform = audio_out["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if hasattr(audio_vae, "first_stage_model"):
        latent_samples = audio_vae.encode(waveform.movedim(1, -1))
    else:
        latent_samples = audio_vae.encode({
            "waveform": waveform,
            "sample_rate": audio_out["sample_rate"],
        })
    if latent_samples.numel() == 0:
        raise ValueError("Encoded audio latent is empty.")

    B, C, F_len, H_len = latent_samples.shape
    # 0 = preserve (lip-sync to this speech), 1 = generate new audio
    # When generate_audio=True the caller overwrites this with all-ones anyway
    gap_mask = torch.zeros((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)

    log.info("[MuseDirector] Encoded custom audio latent — noise_mask=zeros (preserve for lip-sync).")
    return {"samples": latent_samples, "type": "audio", "noise_mask": gap_mask}


# ── Chunk save ────────────────────────────────────────────────────────────────

def _save_chunk_mp4(frames, fps, path):
    import av as _av
    frames_u8 = (frames.cpu().float().clamp(0, 1) * 255).byte().numpy()
    H, W = int(frames_u8.shape[1]), int(frames_u8.shape[2])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=int(fps))
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for f in frames_u8:
            avf = _av.VideoFrame.from_ndarray(f, format="rgb24")
            for pkt in stream.encode(avf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    log.info("[MuseDirector] Saved: %s", path)


def _zero_out_conditioning(conditioning):
    c = []
    for t in conditioning:
        d = t[1].copy()
        pooled = d.get("pooled_output", None)
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        lyrics = d.get("conditioning_lyrics", None)
        if lyrics is not None:
            d["conditioning_lyrics"] = torch.zeros_like(lyrics)
        c.append([torch.zeros_like(t[0]), d])
    return c


def _unpack(result):
    return result.args if hasattr(result, "args") else result


# Candidate index (0-3) -> {"vid1", "aud1", "pos1_c", "neg1_c", "seed"}.
# Populated by a Seed Hunt scouting pass, consumed by the next commit run so
# Stage 2 can refine the actual picked latent instead of regenerating Stage 1
# from scratch. Lives for the lifetime of the ComfyUI server process; a
# cache miss (fresh server, or committing without ever scouting) falls back
# to the old "override seed and rerun Stage 1" behavior.
_SEED_HUNT_CANDIDATES = {}


# ── Node ──────────────────────────────────────────────────────────────────────

class MuseDirectorSamplerV2:
    """
    Muse Collective LTX Timeline V2

    Fork of V1 with an added Seed Hunt feature. See module docstring.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model":            ("MODEL",),
                "clip":             ("CLIP",),
                "audio_vae":        ("VAE",),
                "vae":              ("VAE",),
                "spatial_upscaler": ("LATENT_UPSCALE_MODEL",),

                # Timeline hidden widgets (managed by JS)
                "start_second":     ("FLOAT",  {"default": 0.0,   "min": 0.0, "max": 3600.0, "step": 0.01}),
                "end_second":       ("FLOAT",  {"default": 10.0,  "min": 0.0, "max": 3600.0, "step": 0.01}),
                "duration_seconds": ("FLOAT",  {"default": 10.0,  "min": 0.0, "max": 3600.0, "step": 0.01}),
                "start_frame":      ("INT",    {"default": 0,     "min": 0,   "max": 86400}),
                "end_frame":        ("INT",    {"default": 240,   "min": 0,   "max": 86400}),
                "duration_frames":  ("INT",    {"default": 240,   "min": 1,   "max": 86400}),
                "timeline_data":    ("STRING", {"default": "{}"}),
                "local_prompts":    ("STRING", {"default": ""}),
                "segment_lengths":  ("STRING", {"default": ""}),
                "global_prompt":    ("STRING", {"multiline": True, "default": ""}),
                "guide_strength":   ("STRING", {"default": ""}),
                "epsilon":          ("FLOAT",  {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001}),

                # Generation settings
                "frame_rate":    ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "display_mode":  (["seconds", "frames"], {"default": "seconds"}),
                "custom_width":  ("INT",   {"default": 960,  "min": 64,  "max": 4096, "step": 32}),
                "custom_height": ("INT",   {"default": 544,  "min": 64,  "max": 4096, "step": 32}),
                "resize_method": (["maintain aspect ratio", "stretch to fit", "crop", "pad"],
                                  {"default": "maintain aspect ratio"}),
                "divisible_by":  ("INT",   {"default": 32, "min": 1, "max": 256}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 51}),

                # Audio toggles
                "generate_audio":   ("BOOLEAN", {"default": True,
                                                  "tooltip": "LTX generates ambient/sfx audio from [SOUNDS] prompts."}),
                "custom_audio_on":  ("BOOLEAN", {"default": False,
                                                  "tooltip": "Use audio file(s) from the AUDIO timeline track."}),
                "lipsync":          ("BOOLEAN", {"default": True,
                                                  "tooltip": "Sync mouth movements to custom audio. Requires Custom Audio ON and talking head LoRA."}),
                "motion_guide_on":  ("BOOLEAN", {"default": True,
                                                  "tooltip": "Use motion guide segments from the timeline."}),

                # Chunking
                "chunk_duration_seconds": ("FLOAT", {"default": 10.0, "min": 2.0, "max": 120.0, "step": 0.1}),
                "auto_chunk_threshold":   ("FLOAT", {"default": 10.0, "min": 0.0, "max": 3600.0, "step": 0.5}),
                "carry_frames":     ("INT",   {"default": 73,   "min": 1,   "max": 240, "step": 1,
                                               "tooltip": "Reference frames from previous chunk locked at chunk start. 73 ≈ 3s at 24fps."}),
                "carry_strength":   ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.01}),
                "crossfade_frames": ("INT",   {"default": 0,    "min": 0,   "max": 120, "step": 1}),

                # IC-LoRA
                "ic_lora_name":     (["None"] + loras, {"default": "None"}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                # Sampling
                "stage1_steps":   ("INT",   {"default": 8,    "min": 1, "max": 50}),
                "stage2_steps":   ("INT",   {"default": 4,    "min": 1, "max": 50}),
                "stage2_denoise": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg":            ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed":           ("INT",   {"default": 42,   "min": 0,  "max": 0xFFFFFFFFFFFFFFFF}),
                "filename_prefix": ("STRING", {"default": "muse"}),
                "bg_volume":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),

                # Guide settings — at end so existing workflows don't shift
                "guide_scale_by":            ("FLOAT", {"default": 0.5,  "min": 0.01, "max": 8.0,  "step": 0.01}),
                "guide_scale_by_s2":         ("FLOAT", {"default": 1.0,  "min": 0.01, "max": 8.0,  "step": 0.01}),
                "guide_upscale_method":      (["bicubic", "bilinear", "nearest-exact", "area", "bislerp"], {"default": "bicubic"}),
                "guide_image_attn_strength": ("FLOAT", {"default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "guide_crop":                (["center", "disabled"], {"default": "center"}),
                "guide_auto_snap_ic_grid":   ("BOOLEAN", {"default": True}),
                "guide_use_tiled_encode":    ("BOOLEAN", {"default": False}),
                "guide_tile_size":           ("INT",   {"default": 256, "min": 64, "max": 512, "step": 32}),
                "guide_tile_overlap":        ("INT",   {"default": 64,  "min": 16, "max": 256, "step": 16}),

                # Timeline UI placeholder (hidden by JS)
                "timeline_ui": ("STRING", {"default": ""}),

                # Seed Hunt — cheap 4-seed Stage-1-only scouting pass
                "seed_hunt": ("BOOLEAN", {"default": False,
                                          "tooltip": "ON + no candidate chosen: run a 4-seed Stage-1-resolution "
                                                     "preview instead of the full pipeline. ON + one use_seed_hunt_N "
                                                     "chosen: commit to that candidate — Stage 2 refines its actual "
                                                     "cached latent instead of regenerating Stage 1 from scratch."}),
                "seed_hunt_steps":          ("INT", {"default": 6,  "min": 1, "max": 50}),
                "seed_hunt_scale": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05,
                                              "tooltip": "Unused as of 1.0.4 — Seed Hunt now scouts at Stage 1's "
                                                         "real resolution automatically (so the picked candidate's "
                                                         "actual latent can carry forward into Stage 2). Kept as a "
                                                         "widget only so older saved workflows still load correctly."}),
                "seed_hunt_1": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                         "tooltip": "Unused as of 1.0.4 — scouting now draws a fresh random seed "
                                                    "for each candidate every run instead of reusing these fixed "
                                                    "values (the actual latent carries forward on commit, so the "
                                                    "seed number no longer needs to be fixed or reproducible)."}),
                "seed_hunt_2": ("INT", {"default": 2, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_hunt_3": ("INT", {"default": 3, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seed_hunt_4": ("INT", {"default": 4, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "use_seed_hunt_1": ("BOOLEAN", {"default": False}),
                "use_seed_hunt_2": ("BOOLEAN", {"default": False}),
                "use_seed_hunt_3": ("BOOLEAN", {"default": False}),
                "use_seed_hunt_4": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "bg_audio":    ("AUDIO",),
                "base_model":  ("MODEL", {"tooltip": "Base model without talking-head LoRA. Connect the UNETLoader output directly here so the ambient audio pass generates sounds without speech."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
                     "AUDIO", "AUDIO", "AUDIO", "AUDIO", "IMAGE")
    RETURN_NAMES = ("last_chunk_frames", "audio", "stage1_frames",
                     "seed_hunt_preview_1", "seed_hunt_preview_2",
                     "seed_hunt_preview_3", "seed_hunt_preview_4",
                     "seed_hunt_audio_1", "seed_hunt_audio_2",
                     "seed_hunt_audio_3", "seed_hunt_audio_4",
                     "reference_image")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"
    DESCRIPTION = (
        "Muse Collective LTX Timeline V2 — fully standalone LTX 2.3 AV Director + Infinite Sampler, "
        "with an added Seed Hunt scouting pass (seed_hunt toggle). "
        "Reference-frame latent extension for seamless multi-chunk generation. "
        "No WhatDreamsCost dependency."
    )

    def execute(
        self,
        model, clip, audio_vae, vae, spatial_upscaler,
        start_second, end_second, duration_seconds,
        start_frame, end_frame, duration_frames,
        timeline_data, local_prompts, segment_lengths,
        global_prompt, guide_strength, epsilon,
        frame_rate, display_mode, custom_width, custom_height,
        resize_method, divisible_by, img_compression,
        generate_audio, custom_audio_on, lipsync, motion_guide_on,
        chunk_duration_seconds, auto_chunk_threshold,
        carry_frames, carry_strength, crossfade_frames,
        ic_lora_name, ic_lora_strength,
        stage1_steps, stage2_steps, stage2_denoise, cfg,
        seed, filename_prefix,
        guide_scale_by=0.5, guide_scale_by_s2=1.0, guide_upscale_method="bicubic",
        guide_image_attn_strength=1.0, guide_crop="center", guide_auto_snap_ic_grid=True,
        guide_use_tiled_encode=False, guide_tile_size=256, guide_tile_overlap=64,
        bg_volume=1.0, bg_audio=None, base_model=None, timeline_ui="",
        seed_hunt=False, seed_hunt_steps=6,
        seed_hunt_scale=0.25,
        seed_hunt_1=1, seed_hunt_2=2, seed_hunt_3=3, seed_hunt_4=4,
        use_seed_hunt_1=False, use_seed_hunt_2=False, use_seed_hunt_3=False, use_seed_hunt_4=False,
    ):
        if not isinstance(ic_lora_name, str):
            ic_lora_name = "None"
        stage1_steps = max(1, int(stage1_steps))
        stage2_steps = max(1, int(stage2_steps))

        # Parse timeline
        try:
            tdata = json.loads(timeline_data) if timeline_data and timeline_data.strip() not in ("", "{}") else {}
        except Exception:
            tdata = {}

        # Sync global_prompt from timeline if not connected
        if not global_prompt:
            global_prompt = tdata.get("global_prompt", "")

        log.info("[MuseDirector] global_prompt: %r", global_prompt)

        # ── Seed Hunt ────────────────────────────────────────────────────────
        seed_hunt_cached_candidate = None
        if seed_hunt:
            hunt_toggles = [use_seed_hunt_1, use_seed_hunt_2, use_seed_hunt_3, use_seed_hunt_4]
            hunt_picked = [i for i, t in enumerate(hunt_toggles) if t]

            if not hunt_picked:
                # Now that a picked candidate carries its own actual latent
                # forward (not just its seed number), there's no reason for
                # scouting to keep reusing the same seed_hunt_1..4 values —
                # fresh random seeds each scout run means a different batch
                # of 4 to choose from, with zero effect on Stage 2 quality.
                hunt_seeds = [random.getrandbits(48) for _ in range(4)]
                # Scout the same length the real run's first chunk will actually
                # generate — not an independent fixed value — so seed choice is
                # made on a fair preview of the real generation, not a shorter
                # or longer one that can genuinely diverge in content.
                _scout_total_duration = max(0.0, end_second - start_second)
                _scout_use_chunks = (auto_chunk_threshold <= 0.0) or (_scout_total_duration > auto_chunk_threshold)
                _scout_effective_chunk = chunk_duration_seconds if _scout_use_chunks else _scout_total_duration
                _scout_chunk_seconds = min(_scout_effective_chunk, _scout_total_duration) if _scout_total_duration > 0 else 1.0
                scout_duration_frames = max(9, int(round(_scout_chunk_seconds * frame_rate)))

                log.info("[MuseDirector] Seed Hunt ON, no candidate chosen — running scouting pass "
                          "(seeds=%s, duration_frames=%d matching the real first chunk) at Stage 1's "
                          "real resolution, skipping the full pipeline.",
                          hunt_seeds, scout_duration_frames)
                previews, previews_audio = self._run_seed_hunt(
                    model, clip, audio_vae, vae,
                    tdata, timeline_data, start_frame,
                    global_prompt, local_prompts, segment_lengths, guide_strength, epsilon,
                    frame_rate, scout_duration_frames, custom_width, custom_height,
                    resize_method, divisible_by, img_compression,
                    generate_audio, custom_audio_on, lipsync, motion_guide_on,
                    ic_lora_name, ic_lora_strength, seed_hunt_steps, cfg,
                    guide_scale_by, guide_upscale_method, guide_image_attn_strength,
                    guide_crop, guide_auto_snap_ic_grid, guide_use_tiled_encode,
                    guide_tile_size, guide_tile_overlap, hunt_seeds,
                    base_model=base_model,
                )
                mm.soft_empty_cache(force=True)
                gc.collect()
                dummy_frame = torch.zeros((1, 64, 64, 3))
                dummy_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}
                return (dummy_frame, dummy_audio, dummy_frame,
                        previews[0], previews[1], previews[2], previews[3],
                        previews_audio[0], previews_audio[1], previews_audio[2], previews_audio[3],
                        dummy_frame)
            else:
                if len(hunt_picked) > 1:
                    log.warning("[MuseDirector] Seed Hunt: more than one use_seed_hunt_N is on — "
                                "using the first: seed_hunt_%d", hunt_picked[0] + 1)
                picked_idx = hunt_picked[0]
                seed_hunt_cached_candidate = _SEED_HUNT_CANDIDATES.get(picked_idx)
                if seed_hunt_cached_candidate is not None:
                    # Seed comes from the cache — it's the actual random seed
                    # that produced this candidate, not seed_hunt_N's current
                    # widget value (which no longer drives scouting).
                    seed = seed_hunt_cached_candidate["seed"]
                    log.info("[MuseDirector] Seed Hunt: candidate %d chosen — reusing its cached Stage 1 "
                              "latent (seed=%d), skipping a fresh Stage 1 sampler pass for chunk 1.",
                              picked_idx + 1, seed)
                else:
                    fallback_seeds = [int(seed_hunt_1), int(seed_hunt_2), int(seed_hunt_3), int(seed_hunt_4)]
                    seed = fallback_seeds[picked_idx]
                    log.warning("[MuseDirector] Seed Hunt: candidate %d chosen but no cached latent found "
                                "(server restarted, or seed hunt never scouted) — falling back to a fresh "
                                "Stage 1 with seed=%d.", picked_idx + 1, seed)

                # Committing consumes the cache — what we need for this run is
                # already pulled into seed_hunt_cached_candidate above, so the
                # dict itself (all 4 candidates' full latents, including the 3
                # never picked) is now dead weight. Clear it and reclaim the
                # memory before the expensive Stage 2 work starts, rather than
                # letting it linger in VRAM/RAM for the rest of this run.
                if _SEED_HUNT_CANDIDATES:
                    _SEED_HUNT_CANDIDATES.clear()
                    mm.soft_empty_cache(force=True)
                    gc.collect()
                    log.info("[MuseDirector] Seed Hunt cache cleared after commit.")

        total_duration = end_second - start_second
        use_chunks = (auto_chunk_threshold <= 0.0) or (total_duration > auto_chunk_threshold)
        effective_chunk = chunk_duration_seconds if use_chunks else total_duration
        mode = "chunked" if use_chunks else "single-pass"
        log.info("[MuseDirector] %s mode — %.1fs total", mode, total_duration)

        chunks = []
        t = start_second
        while t < end_second - 0.01:
            end = min(t + effective_chunk, end_second)
            chunks.append((t, end))
            t = end
        log.info("[MuseDirector] %d chunk(s), carry_frames=%d", len(chunks), carry_frames)

        output_dir = folder_paths.get_output_directory()
        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_full.mp4")):
            counter += 1

        # Pre-load bg audio segments from timeline
        _bg_tracks = []
        try:
            _tl_bg_vol = float(tdata.get("bgAudioVolume", 1.0))
            for _bseg in tdata.get("bgAudioSegments", []):
                _af = _bseg.get("audioFile", "")
                if not _af:
                    continue
                _ap = _resolve_path(_af)
                if not _ap:
                    continue
                try:
                    import soundfile as _sf
                    _data, _sr = _sf.read(_ap, dtype="float32", always_2d=True)
                    _bg_tracks.append((
                        torch.from_numpy(_data.T),
                        _sr,
                        float(_bseg.get("start", 0)),
                        float(_bseg.get("length", 1)),
                        float(_bseg.get("trimStart", 0)),
                        _tl_bg_vol,
                    ))
                    log.info("[MuseDirector] BG pre-loaded: %s", _af)
                except Exception as exc:
                    log.warning("[MuseDirector] BG pre-load failed: %s", exc)
        except Exception as exc:
            log.warning("[MuseDirector] BG pre-parse failed: %s", exc)

        all_frames = []
        all_s1_frames = []
        all_waveforms = []
        all_bg_waveforms = []
        audio_sample_rate = 44100
        live_pixel_frames = None
        reference_image_out = None
        color_ref_mean = None
        color_ref_std = None

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_seed = seed
            # V2 change from V1: Stage 2 uses the same seed as Stage 1 (V1 used
            # seed - 1) so a Seed Hunt candidate's noise carries through to the
            # real Stage 2 refine — otherwise the committed run's Stage 2 uses
            # a seed that was never previewed during scouting.
            s2_seed = seed

            raw_s_fr = int(chunk_start * frame_rate)
            e_fr = int(chunk_end * frame_rate)

            if chunk_idx > 0 and live_pixel_frames is not None:
                ref_pixel_count = min(carry_frames, live_pixel_frames.shape[0], raw_s_fr)
            else:
                ref_pixel_count = 0

            overlap_frames = ref_pixel_count
            s_fr = raw_s_fr - ref_pixel_count
            gen_start = s_fr / frame_rate

            log.info(
                "[MuseDirector] Chunk %d/%d  %.2f→%.2f s  (ref=%d px → %.3fs locked)  seed=%d",
                chunk_idx + 1, len(chunks), chunk_start, chunk_end,
                ref_pixel_count, gen_start, chunk_seed,
            )

            s1_w = max(divisible_by, (custom_width  // 2 // divisible_by) * divisible_by)
            s1_h = max(divisible_by, (custom_height // 2 // divisible_by) * divisible_by)

            n_chunk_frames = e_fr - s_fr
            ltxv_len = int(math.ceil((n_chunk_frames - 1) / 8.0) * 8) + 1
            latent_t = ((ltxv_len - 1) // 8) + 1
            pre_latent = {"samples": torch.zeros(
                [1, 128, latent_t, s1_h // 32, s1_w // 32],
                device=mm.intermediate_device(),
            )}

            chunk_dur = e_fr / frame_rate - gen_start

            # ── Build guide data ─────────────────────────────────────────────
            chunk_s_frame = s_fr
            chunk_dur_frames = e_fr - s_fr
            guide_data, derived_w, derived_h = _build_guide_data(
                tdata, chunk_s_frame, chunk_dur_frames,
                frame_rate, s1_w, s1_h, resize_method, divisible_by,
                img_compression, guide_strength,
            )
            guide_data["timeline_data"] = timeline_data
            guide_data["start_frame"] = chunk_s_frame
            guide_data["duration_frames"] = chunk_dur_frames
            guide_data["resize_method"] = resize_method

            if reference_image_out is None and guide_data["images"]:
                reference_image_out = guide_data["images"][0]

            motion_guide_data = _build_motion_guide_data(
                timeline_data, chunk_s_frame, chunk_dur_frames,
                frame_rate, resize_method, motion_guide_on,
            )
            log.info("[MuseDirector] Motion guide: %d segment(s) (motion_guide_on=%s)",
                     len(motion_guide_data["segments"]), motion_guide_on)

            # ── Build conditioning ───────────────────────────────────────────
            chunk_local_prompts, chunk_segment_lengths = _build_chunk_local_prompts(
                tdata, chunk_s_frame, chunk_dur_frames, local_prompts, segment_lengths,
            )
            patched_model, positive = _encode_relay(
                model, clip, pre_latent,
                global_prompt, chunk_local_prompts, chunk_segment_lengths, epsilon,
            )

            # ── Build audio waveform + latent ────────────────────────────────
            combined_audio = _build_combined_audio(
                timeline_data, chunk_s_frame, ltxv_len, float(frame_rate),
            )
            audio_latent = _build_audio_latent(
                audio_vae, combined_audio, ltxv_len, frame_rate,
                custom_audio_on, generate_audio,
            )

            # Decide noise_mask: 0=preserve audio (lip-sync), 1=generate audio here
            if "samples" in audio_latent:
                s = audio_latent["samples"]
                if custom_audio_on and lipsync:
                    # Preserve the encoded speech — model syncs lips to it
                    mask = torch.zeros(s.shape[0], s.shape[2], s.shape[3], dtype=torch.float32, device=s.device)
                    log.info("[MuseDirector] Lipsync ON — zeros mask, model preserves speech.")
                else:
                    # Generate audio (no custom audio, or custom audio with lipsync off)
                    mask = torch.ones(s.shape[0], s.shape[2], s.shape[3], dtype=torch.float32, device=s.device)
                    log.info("[MuseDirector] Lipsync OFF / no custom audio — ones mask, LTX generates.")
                audio_latent = {**audio_latent, "noise_mask": mask}

            # ── Stage 1 ─────────────────────────────────────────────────────
            # LTXVConditioning first, then DirectorGuide (same order as V7)
            zero_neg = _zero_out_conditioning(positive)
            cond_pos, cond_neg = _unpack(LTXVConditioning.execute(positive, zero_neg, frame_rate))

            # Keep the pre-LoRA (relay-patched only) model so Stage 2 can load the
            # IC-LoRA fresh onto it too, instead of double-applying on top of Stage 1's
            # already-patched model (matches the original WDC LTXDirector behavior,
            # where both stages' model inputs trace back to the same clean source).
            relay_model = patched_model

            pos1, neg1, lat1 = cond_pos, cond_neg, pre_latent
            try:
                pos1, neg1, lat1, patched_model = _apply_guide(
                    cond_pos, cond_neg, vae, pre_latent, guide_data, patched_model,
                    motion_guide_data=motion_guide_data,
                    ic_lora_name=ic_lora_name, ic_lora_strength=ic_lora_strength,
                    scale_by=guide_scale_by, upscale_method=guide_upscale_method,
                    image_attention_strength=guide_image_attn_strength,
                    crop=guide_crop, auto_snap_ic_grid=guide_auto_snap_ic_grid,
                    use_tiled_encode=guide_use_tiled_encode,
                    tile_size=guide_tile_size, tile_overlap=guide_tile_overlap,
                )
            except Exception as exc:
                log.warning("[MuseDirector] Guide application failed, using plain conditioning: %s", exc)
                pos1, neg1 = cond_pos, cond_neg
                lat1 = pre_latent

            # Reference-frame lock
            ref_t = 0
            if live_pixel_frames is not None and ref_pixel_count > 0:
                try:
                    import comfy.utils as _cu
                    ref_px = live_pixel_frames[-ref_pixel_count:].float()
                    ref_s1 = _cu.common_upscale(
                        ref_px.movedim(-1, 1), s1_w, s1_h, "bilinear", "disabled"
                    ).movedim(1, -1)
                    ref_lat = vae.encode(ref_s1[:, :, :, :3])
                    ref_t = ref_lat.shape[2]
                    lat1_samples = lat1["samples"].clone()
                    cap = min(ref_t, lat1_samples.shape[2])
                    lat1_samples[:, :, :cap] = ref_lat[:, :, :cap].to(lat1_samples.device, lat1_samples.dtype)
                    ref_mask = torch.ones(
                        [1, 1, lat1_samples.shape[2], 1, 1],
                        device=lat1_samples.device, dtype=torch.float32,
                    )
                    ref_mask[:, :, :cap] = 0.0
                    lat1 = {**lat1, "samples": lat1_samples, "noise_mask": ref_mask}
                    log.info("[MuseDirector] Reference lock S1: %d px → %d latent frames frozen",
                             ref_pixel_count, cap)
                except Exception as exc:
                    log.warning("[MuseDirector] Reference lock S1 failed: %s", exc)

            if chunk_idx == 0 and seed_hunt_cached_candidate is not None:
                # Skip a fresh Stage 1 sampler pass entirely — carry the
                # picked scout candidate's actual generated latent forward
                # into Stage 2 instead, matching Tristan's ltx23SeedHunter
                # workflow (composition comes from the real content, not
                # from a seed number reproducing it at a different shape).
                vid1 = seed_hunt_cached_candidate["vid1"]
                aud1 = seed_hunt_cached_candidate["aud1"]
                pos1_c = seed_hunt_cached_candidate["pos1_c"]
                neg1_c = seed_hunt_cached_candidate["neg1_c"]
                has_audio = isinstance(aud1, dict) and "samples" in aud1
                sampler = _unpack(KSamplerSelect.execute("euler"))[0]  # Stage 2 reuses this below
            else:
                guider1 = _unpack(CFGGuider.execute(patched_model, pos1, neg1, cfg))[0]
                sampler  = _unpack(KSamplerSelect.execute("euler"))[0]
                sigmas1  = _unpack(BasicScheduler.execute(patched_model, "linear_quadratic", stage1_steps, 1.0))[0]
                noise1   = _unpack(RandomNoise.execute(chunk_seed))[0]
                has_audio = "samples" in audio_latent
                if has_audio:
                    av1_lat = _unpack(LTXVConcatAVLatent.execute(lat1, audio_latent))[0]
                else:
                    av1_lat = lat1
                out1, _ = _unpack(SamplerCustomAdvanced.execute(noise1, guider1, sampler, sigmas1, av1_lat))
                if has_audio:
                    out1_vid, aud1 = _unpack(LTXVSeparateAVLatent.execute(out1))
                else:
                    out1_vid = out1
                    aud1 = {}

                pos1_c, neg1_c, vid1 = _crop_conditioning(pos1, neg1, out1_vid)

            # Stage1 decode for comparison
            try:
                s1_decoded = vae.decode(vid1["samples"])
                if s1_decoded.ndim == 5:
                    s1_decoded = s1_decoded.squeeze(0)
                if s1_decoded.ndim == 3:
                    s1_decoded = s1_decoded.unsqueeze(0)
                if overlap_frames > 0:
                    s1_decoded = s1_decoded[overlap_frames:]
                nominal_s1 = int(round((chunk_end - chunk_start) * frame_rate))
                if s1_decoded.shape[0] > nominal_s1:
                    s1_decoded = s1_decoded[:nominal_s1]
                all_s1_frames.append(s1_decoded.cpu())
            except Exception as exc:
                log.warning("[MuseDirector] Stage1 decode failed: %s", exc)

            # ── Stage 2 ─────────────────────────────────────────────────────
            vid_up = _unpack(LTXVLatentUpsampler.execute(vid1, spatial_upscaler, vae))[0]

            # Stage 2: DirectorGuide only, no LTXVConditioning (conditioning already baked in from S1)
            try:
                pos2, neg2, lat2, _ = _apply_guide(
                    pos1_c, neg1_c, vae, vid_up, guide_data, relay_model,
                    motion_guide_data=motion_guide_data,
                    ic_lora_name=ic_lora_name, ic_lora_strength=ic_lora_strength,
                    scale_by=guide_scale_by_s2, upscale_method=guide_upscale_method,
                    image_attention_strength=guide_image_attn_strength,
                    crop=guide_crop, auto_snap_ic_grid=guide_auto_snap_ic_grid,
                    use_tiled_encode=guide_use_tiled_encode,
                    tile_size=guide_tile_size, tile_overlap=guide_tile_overlap,
                )[:4]
            except Exception as exc:
                log.warning("[MuseDirector] Stage2 guide failed: %s", exc)
                pos2, neg2 = pos1_c, neg1_c
                lat2 = vid_up
            # relay_model was only needed to give Stage 2 a fresh (pre-LoRA) model to
            # patch — release the reference now so it doesn't linger held in scope
            # for the rest of the chunk loop.
            del relay_model

            # Reference-frame lock Stage2
            if live_pixel_frames is not None and ref_pixel_count > 0 and ref_t > 0:
                try:
                    import comfy.utils as _cu2
                    s2_h = lat2["samples"].shape[3] * 32
                    s2_w = lat2["samples"].shape[4] * 32
                    ref_px2 = live_pixel_frames[-ref_pixel_count:].float()
                    ref_s2 = _cu2.common_upscale(
                        ref_px2.movedim(-1, 1), s2_w, s2_h, "bilinear", "disabled"
                    ).movedim(1, -1)
                    ref_lat2 = vae.encode(ref_s2[:, :, :, :3])
                    cap2 = min(ref_lat2.shape[2], lat2["samples"].shape[2])
                    lat2_samples = lat2["samples"].clone()
                    lat2_samples[:, :, :cap2] = ref_lat2[:, :, :cap2].to(lat2_samples.device, lat2_samples.dtype)
                    ref_mask2 = torch.ones(
                        [1, 1, lat2_samples.shape[2], 1, 1],
                        device=lat2_samples.device, dtype=torch.float32,
                    )
                    ref_mask2[:, :, :cap2] = 0.0
                    lat2 = {**lat2, "samples": lat2_samples, "noise_mask": ref_mask2}
                    log.info("[MuseDirector] Reference lock S2: %d latent frames frozen", cap2)
                except Exception as exc:
                    log.warning("[MuseDirector] Reference lock S2 failed: %s", exc)

            guider2  = _unpack(CFGGuider.execute(patched_model, pos2, neg2, cfg))[0]
            sigmas2  = _unpack(BasicScheduler.execute(patched_model, "linear_quadratic",
                                                       stage2_steps, stage2_denoise))[0]
            noise2   = _unpack(RandomNoise.execute(s2_seed))[0]
            if has_audio:
                if custom_audio_on and lipsync:
                    # Re-apply the same "preserve audio" lock used for Stage 1 —
                    # aud1 is a sampler *output* and doesn't carry the original
                    # noise_mask forward, so without this Stage 2 is free to
                    # regenerate the audio-video correlation and can lose the
                    # lip-sync that Stage 1 established.
                    aud1_s = aud1["samples"]
                    aud1_mask = torch.zeros(
                        aud1_s.shape[0], aud1_s.shape[2], aud1_s.shape[3],
                        dtype=torch.float32, device=aud1_s.device,
                    )
                    aud1 = {**aud1, "noise_mask": aud1_mask}
                av2_lat = _unpack(LTXVConcatAVLatent.execute(lat2, aud1))[0]
            else:
                av2_lat = lat2
            out2, _ = _unpack(SamplerCustomAdvanced.execute(noise2, guider2, sampler, sigmas2, av2_lat))
            if has_audio:
                out2_nosemask = {k: v for k, v in out2.items() if k != "noise_mask"}
                out2_vid, _ = _unpack(LTXVSeparateAVLatent.execute(out2_nosemask))
            else:
                out2_vid = out2
            _, _, vid_final = _crop_conditioning(pos2, neg2, out2_vid)

            # ── VAE decode ───────────────────────────────────────────────────
            lat_shape = list(vid_final["samples"].shape)
            frames = vae.decode(vid_final["samples"])
            log.info("[MuseDirector] latent %s → decoded %s", lat_shape, list(frames.shape))
            if frames.ndim == 5:
                frames = frames.squeeze(0)
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)

            if overlap_frames > 0:
                frames = frames[overlap_frames:]

            if all_frames and (frames.shape[1] != all_frames[0].shape[1] or frames.shape[2] != all_frames[0].shape[2]):
                import comfy.utils as _cu
                frames = _cu.common_upscale(
                    frames.movedim(-1, 1), all_frames[0].shape[2], all_frames[0].shape[1],
                    "lanczos", "disabled"
                ).movedim(1, -1).clamp(0, 1)

            nominal_frames = int(round((chunk_end - chunk_start) * frame_rate))
            if frames.shape[0] > nominal_frames:
                frames = frames[:nominal_frames]

            # Color match
            if chunk_idx == 0:
                _ref_frames = frames.float()
                color_ref_mean = _ref_frames.mean(dim=(0, 1, 2))
                color_ref_std  = _ref_frames.std(dim=(0, 1, 2)).clamp(min=1e-5)
            elif all_frames:
                try:
                    # float16 instead of float32, and chained in-place ops
                    # instead of each arithmetic step allocating its own new
                    # tensor -- the original float32, non-in-place version
                    # held ~5-6 separate full-chunk-sized tensors alive at
                    # once (src, corrected, and the intermediates each "+"/
                    # "*"/"-" in the expression silently created), which is
                    # what was failing to allocate on longer chunks. This
                    # keeps only 2 full-size tensors alive (src, corrected)
                    # at roughly a quarter the memory each. Color-grading
                    # math doesn't need float32 precision -- these are RGB
                    # values in 0-1 range being averaged/rescaled, not
                    # anything numerically sensitive.
                    src = frames.to(torch.float16)
                    src_mean = src.mean(dim=(0, 1, 2))
                    src_std  = src.std(dim=(0, 1, 2)).clamp(min=1e-5)
                    # color_ref_mean/std were captured from chunk 0's decode —
                    # by the time chunk N's color match runs, dynamic VRAM
                    # loading may have moved tensors between devices, so
                    # re-home them onto src's device/dtype rather than
                    # assuming they still match (a silent device-mismatch
                    # exception here was the likely reason this was never
                    # visibly correcting anything).
                    ref_mean = color_ref_mean.to(device=src.device, dtype=src.dtype)
                    ref_std  = color_ref_std.to(device=src.device, dtype=src.dtype)
                    corrected = src.sub(src_mean).div_(src_std).mul_(ref_std).add_(ref_mean).clamp_(0.0, 1.0)
                    # Ease from full correction right at the chunk boundary
                    # down to a low resting correction over the first
                    # second, then hold there for the rest of the chunk.
                    # Previously only the first n_blend frames were blended
                    # and everything after snapped back to 100% correction,
                    # creating its own visible seam a second into the chunk
                    # instead of a smooth transition.
                    ramp_len = min(int(frame_rate), frames.shape[0])
                    floor = 0.15
                    blend = torch.full((frames.shape[0],), floor, device=frames.device, dtype=torch.float16)
                    blend[:ramp_len] = torch.linspace(1.0, floor, ramp_len, device=frames.device, dtype=torch.float16)
                    corrected.mul_(blend[:, None, None, None])
                    src.mul_(1.0 - blend[:, None, None, None])
                    frames = corrected.add_(src).to(all_frames[-1].dtype)
                except Exception as exc:
                    log.warning("[MuseDirector] Color match failed: %s", exc, exc_info=True)

            # Save the per-chunk safety-net file using the fully processed
            # frames (post overlap-trim, post color-match) so it matches
            # exactly what ends up in the real assembled output -- previously
            # this saved before the overlap trim, which duplicated the
            # carried-over reference frames into every chunk after the first
            # (extra ~3s per chunk, visible repeat/stutter at every boundary).
            out_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_{chunk_idx:03d}.mp4")
            try:
                _save_chunk_mp4(frames, frame_rate, out_path)
            except Exception as exc:
                log.warning("[MuseDirector] Chunk save failed: %s", exc)

            # Store at float16 instead of float32 -- halves the memory of the
            # final full-video assembly (torch.cat(all_frames, dim=0) at the
            # end of the run), which is what crashed on an 85s/1920x1088 run
            # (51GB requested at float32, ~25.6GB at float16). All the float
            # math this chunk needed (trim, resize-correction, color-match
            # blend) is already done above this point -- this is purely a
            # storage-precision change for what's about to sit in RAM for the
            # rest of the run, not a change to any processing.
            frames = frames.clamp(0.0, 1.0).to(torch.float16)

            all_frames.append(frames)
            # Only concatenate enough of the tail to cover carry_frames worth of
            # pixels -- NOT the entire history. Re-concatenating everything
            # generated so far, every single chunk, is what caused an unbounded
            # (and eventually crashing) memory allocation on long videos.
            # Walking backward until there are enough frames gives the exact
            # same result (only the tail ever mattered) at a fraction of the
            # memory/time cost.
            tail_chunks = []
            tail_len = 0
            for f in reversed(all_frames):
                tail_chunks.append(f)
                tail_len += f.shape[0]
                if tail_len >= carry_frames:
                    break
            tail_chunks.reverse()
            cat_so_far = torch.cat(tail_chunks, dim=0) if len(tail_chunks) > 1 else tail_chunks[0]
            live_pixel_frames = cat_so_far[-carry_frames:].clone().cpu()

            # ── Ambient-only pass ───────────────────────────────────────────────
            # The main pass's model has the talking-head LoRA applied, which is
            # trained on clean dry speech and suppresses [SOUNDS] ambient content
            # regardless of prompt wording or CFG — confirmed by testing, not just
            # theory. So whenever generate_audio is on, run a second lightweight
            # pass on a LoRA-free model (wire base_model = UNETLoader output
            # directly, bypassing the LoRA chain) to actually get ambient/SFX,
            # then mix it under whatever speech source is in use (custom lipsync
            # audio, or the main pass's own generated speech). Runs regardless of
            # custom_audio_on/lipsync now — it used to be gated to only the
            # custom-audio-plus-lipsync case, which silently dropped ambient sound
            # for pure generate_audio-only runs.
            ambient_wav = None
            if generate_audio and audio_vae is not None:
                try:
                    inner_vae_a = getattr(audio_vae, "first_stage_model", audio_vae)
                    num_amb_latents = inner_vae_a.num_of_latents_from_frames(ltxv_len, float(frame_rate))
                    z_ch = audio_vae.latent_channels
                    a_freq = inner_vae_a.latent_frequency_bins
                    amb_audio_samples = torch.zeros(
                        (1, z_ch, num_amb_latents, a_freq),
                        device=mm.intermediate_device(),
                    )
                    amb_noise_mask = torch.ones(
                        (1, num_amb_latents, a_freq),
                        dtype=torch.float32, device=mm.intermediate_device(),
                    )
                    amb_audio_lat = {"samples": amb_audio_samples, "type": "audio", "noise_mask": amb_noise_mask}
                    # Use actual video latent so the model has visual context for sound generation
                    # Add a video noise_mask of ones so the video is treated as fully generated (not locked)
                    vid_ones = torch.ones(
                        (pre_latent["samples"].shape[0], pre_latent["samples"].shape[2],
                         pre_latent["samples"].shape[3], pre_latent["samples"].shape[4]),
                        dtype=torch.float32, device=pre_latent["samples"].device,
                    )
                    amb_vid_lat = {"samples": pre_latent["samples"].clone(), "noise_mask": vid_ones}
                    av_amb = _unpack(LTXVConcatAVLatent.execute(amb_vid_lat, amb_audio_lat))[0]
                    # Build conditioning from [SOUNDS] content only — strips [SPEECH] so the
                    # model generates ambient audio without adding speech on top
                    import re as _re
                    all_prompts = [global_prompt] + (local_prompts if isinstance(local_prompts, list) else [local_prompts])
                    sounds_parts = []
                    for _p in all_prompts:
                        for _m in _re.findall(r'\[SOUNDS?\][^\[|]*', _p, _re.IGNORECASE):
                            sounds_parts.append(_m.strip())
                    amb_text = ' '.join(sounds_parts) if sounds_parts else "ambient background sounds"
                    amb_cond = clip.encode_from_tokens_scheduled(clip.tokenize(amb_text))
                    amb_zero_neg = _zero_out_conditioning(amb_cond)
                    amb_cond_pos, amb_cond_neg = _unpack(LTXVConditioning.execute(amb_cond, amb_zero_neg, frame_rate))
                    # Use base_model (no LoRA) if wired, otherwise fall back to the main model
                    amb_model = base_model if base_model is not None else model
                    amb_guider = _unpack(CFGGuider.execute(amb_model, amb_cond_pos, amb_cond_neg, cfg))[0]
                    amb_sampler = _unpack(KSamplerSelect.execute("euler"))[0]
                    amb_steps = stage1_steps
                    amb_sigmas = _unpack(BasicScheduler.execute(amb_model, "linear_quadratic", amb_steps, 1.0))[0]
                    amb_noise = _unpack(RandomNoise.execute(chunk_seed + 99999))[0]
                    amb_out, _ = _unpack(SamplerCustomAdvanced.execute(amb_noise, amb_guider, amb_sampler, amb_sigmas, av_amb))
                    _, amb_aud = _unpack(LTXVSeparateAVLatent.execute(amb_out))
                    if isinstance(amb_aud, dict) and "samples" in amb_aud:
                        amb_decoded = inner_vae_a.decode(amb_aud["samples"].cpu().float())
                        if amb_decoded.shape[1] == 1:
                            amb_decoded = amb_decoded.expand(-1, 2, -1)
                        ambient_wav = amb_decoded.cpu().float()
                        amb_sr = getattr(inner_vae_a, "output_sample_rate", 44100)
                        if amb_sr != 44100:
                            import torchaudio as _ta
                            ambient_wav = _ta.functional.resample(ambient_wav, amb_sr, 44100)
                        log.info("[MuseDirector] Ambient pass decoded: shape %s", list(ambient_wav.shape))
                except Exception as exc:
                    log.warning("[MuseDirector] Ambient pass failed: %s", exc)

            # ── Audio source selection ────────────────────────────────────────
            decoded_wav = None
            if generate_audio and has_audio and isinstance(aud1, dict) and "samples" in aud1:
                try:
                    inner_vae = audio_vae.first_stage_model
                    aud_samples = aud1["samples"].cpu().float()
                    decoded_wav = inner_vae.decode(aud_samples)
                    if decoded_wav.shape[1] == 1:
                        decoded_wav = decoded_wav.expand(-1, 2, -1)
                    decoded_wav = decoded_wav.cpu().float()
                    audio_sr = getattr(inner_vae, "output_sample_rate", 44100)
                    if audio_sr != 44100:
                        import torchaudio
                        decoded_wav = torchaudio.functional.resample(decoded_wav, audio_sr, 44100)
                        audio_sr = 44100
                    log.info("[MuseDirector] Decoded generated audio: shape %s sr=%d",
                             list(decoded_wav.shape), audio_sr)
                except Exception as exc:
                    log.warning("[MuseDirector] Audio decode failed: %s", exc)
                    decoded_wav = None

            custom_wav = combined_audio.get("waveform") if (custom_audio_on and isinstance(combined_audio, dict)) else None
            audio_sr = combined_audio.get("sample_rate", 44100) if isinstance(combined_audio, dict) else 44100

            if generate_audio and custom_audio_on and lipsync and custom_wav is not None:
                # All three on: clean custom speech + ambient from separate pass
                if ambient_wav is not None:
                    min_len = min(ambient_wav.shape[-1], custom_wav.shape[-1])
                    mixed = custom_wav[..., :min_len] + ambient_wav[..., :min_len] * 0.25
                    combined_audio = {"waveform": mixed, "sample_rate": audio_sr}
                    log.info("[MuseDirector] Audio: lipsync speech + ambient layer")
                else:
                    combined_audio = {"waveform": custom_wav, "sample_rate": audio_sr}
                    log.info("[MuseDirector] Audio: lipsync speech only (ambient pass failed)")
            elif decoded_wav is not None and custom_wav is not None:
                min_len = min(decoded_wav.shape[-1], custom_wav.shape[-1])
                mixed = decoded_wav[..., :min_len] + custom_wav[..., :min_len]
                combined_audio = {"waveform": mixed, "sample_rate": audio_sr}
                log.info("[MuseDirector] Audio: mixed generated + custom")
            elif decoded_wav is not None and ambient_wav is not None:
                # Pure generate_audio mode: decoded_wav is the main pass's own
                # speech (talking-head LoRA, ambient-suppressed) — layer the
                # LoRA-free ambient pass underneath it.
                min_len = min(decoded_wav.shape[-1], ambient_wav.shape[-1])
                mixed = decoded_wav[..., :min_len] + ambient_wav[..., :min_len] * 0.25
                combined_audio = {"waveform": mixed, "sample_rate": audio_sr}
                log.info("[MuseDirector] Audio: generated speech + ambient layer")
            elif decoded_wav is not None:
                combined_audio = {"waveform": decoded_wav, "sample_rate": audio_sr}
                log.info("[MuseDirector] Audio: generated only")
            elif custom_wav is not None:
                combined_audio = {"waveform": custom_wav, "sample_rate": audio_sr}
                log.info("[MuseDirector] Audio: custom only")
            else:
                log.info("[MuseDirector] Audio: silence")

            # Trim + cap audio
            if isinstance(combined_audio, dict) and "waveform" in combined_audio:
                waveform = combined_audio["waveform"]
                audio_sample_rate = combined_audio.get("sample_rate", 44100)
                if overlap_frames > 0:
                    trim_samples = int(overlap_frames * audio_sample_rate / frame_rate)
                    waveform = waveform[:, :, trim_samples:]
                nominal_samples = int((chunk_end - chunk_start) * audio_sample_rate)
                if waveform.shape[-1] > nominal_samples:
                    waveform = waveform[:, :, :nominal_samples]
                all_waveforms.append(waveform)

            # Per-chunk bg audio
            if _bg_tracks:
                chunk_speech_len = waveform.shape[-1] if (isinstance(combined_audio, dict) and "waveform" in combined_audio) else int((chunk_end - gen_start) * audio_sample_rate)
                bg_chunk_out = torch.zeros((2, chunk_speech_len), dtype=torch.float32)
                for (_bg_raw, _bg_sr, _seg_start_fr, _seg_len_fr, _seg_trim_fr, _vol) in _bg_tracks:
                    _bg_w = _bg_raw
                    if _bg_sr != audio_sample_rate:
                        import torchaudio as _ta2
                        _bg_w = _ta2.functional.resample(_bg_w.unsqueeze(0), _bg_sr, audio_sample_rate).squeeze(0)
                    _offset_fr = max(0.0, s_fr - _seg_start_fr)
                    _eff_trim_fr = _seg_trim_fr + _offset_fr
                    _eff_len_fr = max(1.0, _seg_len_fr - _offset_fr)
                    _dst_start_fr = max(0.0, _seg_start_fr - s_fr)
                    _src_start = int(_eff_trim_fr / frame_rate * audio_sample_rate)
                    _src_end = min(_src_start + int(_eff_len_fr / frame_rate * audio_sample_rate), _bg_w.shape[-1])
                    _dst_start = int(_dst_start_fr / frame_rate * audio_sample_rate)
                    if _src_end > _src_start and _dst_start < chunk_speech_len:
                        _clip = _bg_w[:, _src_start:_src_end]
                        if overlap_frames > 0:
                            _bg_trim = int(overlap_frames * audio_sample_rate / frame_rate)
                            _clip = _clip[:, _bg_trim:]
                        _avail = min(_clip.shape[-1], chunk_speech_len - _dst_start)
                        if _clip.shape[0] == 1:
                            _clip = _clip.expand(2, -1)
                        elif _clip.shape[0] > 2:
                            _clip = _clip[:2, :]
                        bg_chunk_out[:, _dst_start:_dst_start + _avail] += _clip[:, :_avail] * _vol
                all_bg_waveforms.append(bg_chunk_out.unsqueeze(0))

            mm.soft_empty_cache()

        # ── Assemble output ────────────────────────────────────────────────────
        if not all_frames:
            all_frames = [torch.zeros((1, custom_height, custom_width, 3))]

        if all_waveforms:
            full_audio = {"waveform": torch.cat(all_waveforms, dim=2), "sample_rate": audio_sample_rate}
        else:
            full_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}

        if all_bg_waveforms:
            try:
                full_bg_w = torch.cat(all_bg_waveforms, dim=2)
                lip_w = full_audio["waveform"]
                lip_len = lip_w.shape[-1]
                if full_bg_w.shape[-1] > lip_len:
                    full_bg_w = full_bg_w[..., :lip_len]
                elif full_bg_w.shape[-1] < lip_len:
                    full_bg_w = torch.nn.functional.pad(full_bg_w, (0, lip_len - full_bg_w.shape[-1]))
                if full_bg_w.shape[1] != lip_w.shape[1]:
                    full_bg_w = full_bg_w.expand(-1, lip_w.shape[1], -1) if full_bg_w.shape[1] == 1 else full_bg_w[:, :lip_w.shape[1], :]
                full_audio = {"waveform": lip_w + full_bg_w.to(lip_w.device, lip_w.dtype), "sample_rate": audio_sample_rate}
            except Exception as exc:
                log.warning("[MuseDirector] BG audio mix failed: %s", exc)

        if bg_audio is not None and "waveform" in bg_audio:
            try:
                lip_w = full_audio["waveform"]
                bg_w = bg_audio["waveform"].clone()
                bg_sr = bg_audio.get("sample_rate", audio_sample_rate)
                if bg_sr != audio_sample_rate:
                    import torchaudio
                    bg_w = torchaudio.functional.resample(bg_w, bg_sr, audio_sample_rate)
                lip_len = lip_w.shape[-1]
                if bg_w.shape[-1] > lip_len:
                    bg_w = bg_w[..., :lip_len]
                elif bg_w.shape[-1] < lip_len:
                    bg_w = torch.nn.functional.pad(bg_w, (0, lip_len - bg_w.shape[-1]))
                if bg_w.shape[1] != lip_w.shape[1]:
                    bg_w = bg_w.expand(-1, lip_w.shape[1], -1) if bg_w.shape[1] == 1 else bg_w[:, :lip_w.shape[1], :]
                full_audio = {"waveform": lip_w + bg_w.to(lip_w.device, lip_w.dtype) * bg_volume,
                              "sample_rate": audio_sample_rate}
            except Exception as exc:
                log.warning("[MuseDirector] Wire BG audio mix failed: %s", exc)

        n_cf = int(crossfade_frames) if crossfade_frames else 0
        if len(all_frames) > 1 and n_cf > 0:
            result = [all_frames[0]]
            for i in range(1, len(all_frames)):
                prev, curr = result[-1], all_frames[i]
                n = min(n_cf, prev.shape[0], curr.shape[0])
                if n > 0:
                    alphas = torch.linspace(0.0, 1.0, n + 2, device=prev.device)[1:-1]
                    blended = prev.clone()
                    for j in range(n):
                        a = float(alphas[j])
                        blended[-n + j] = ((1 - a) * prev[-n + j] + a * curr[j]).clamp(0, 1)
                    result[-1] = blended
                    result.append(curr[n:])
                else:
                    result.append(curr)
            full_video = torch.cat(result, dim=0)
        else:
            full_video = torch.cat(all_frames, dim=0)

        full_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_full.mp4")
        # Use ffmpeg concat to avoid loading all frames into CPU RAM (can be 20GB+ for long videos)
        chunk_paths = [
            os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_{i:03d}.mp4")
            for i in range(len(chunks))
            if os.path.exists(os.path.join(output_dir, f"{filename_prefix}_{counter:05d}_{i:03d}.mp4"))
        ]
        if len(chunk_paths) > 1:
            try:
                import subprocess, tempfile
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as flist:
                    for cp in chunk_paths:
                        flist.write(f"file '{cp}'\n")
                    flist_path = flist.name
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", flist_path,
                     "-c", "copy", full_path],
                    check=True, capture_output=True
                )
                os.unlink(flist_path)
                log.info("[MuseDirector] Full video saved via ffmpeg concat: %s", full_path)
            except Exception as exc:
                log.warning("[MuseDirector] ffmpeg concat failed, falling back to in-memory save: %s", exc)
                try:
                    _save_chunk_mp4(full_video, frame_rate, full_path)
                except Exception as exc2:
                    log.warning("[MuseDirector] Full video save failed: %s", exc2)
        else:
            try:
                _save_chunk_mp4(full_video, frame_rate, full_path)
            except Exception as exc:
                log.warning("[MuseDirector] Full video save failed: %s", exc)

        # Once full_video exists it holds its own copy of every chunk's data --
        # the source list is now a redundant second copy of the same frames.
        # Free it here, before building the full-length stage1_frames preview
        # below, instead of letting both the full Stage2 video and the full
        # Stage1 video be resident in RAM at the same time.
        all_frames.clear()

        s1_video = torch.cat(all_s1_frames, dim=0) if all_s1_frames else full_video
        _dummy_scout = torch.zeros((1, 64, 64, 3))
        _dummy_scout_audio = {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}
        return (full_video, full_audio, s1_video,
                _dummy_scout, _dummy_scout, _dummy_scout, _dummy_scout,
                _dummy_scout_audio, _dummy_scout_audio, _dummy_scout_audio, _dummy_scout_audio,
                reference_image_out if reference_image_out is not None else _dummy_scout)

    def _run_seed_hunt(
        self, model, clip, audio_vae, vae,
        tdata, timeline_data, start_frame,
        global_prompt, local_prompts, segment_lengths, guide_strength, epsilon,
        frame_rate, seed_hunt_duration_frames, custom_width, custom_height,
        resize_method, divisible_by, img_compression,
        generate_audio, custom_audio_on, lipsync, motion_guide_on,
        ic_lora_name, ic_lora_strength, seed_hunt_steps, cfg,
        guide_scale_by, guide_upscale_method, guide_image_attn_strength,
        guide_crop, guide_auto_snap_ic_grid, guide_use_tiled_encode,
        guide_tile_size, guide_tile_overlap, hunt_seeds,
        base_model=None,
    ):
        """Stage-1-only pass, run once per seed in hunt_seeds, at Stage 1's
        own real resolution (half of custom_width/custom_height) — the exact
        same shape the real Stage 1 would use. Reuses the exact same helper
        functions the real Stage 1 above uses, against the real timeline
        data, so multi-segment timelines, the real audio track, and IC-LoRA
        all behave the same as a full run would. Each candidate's actual
        Stage 1 output latent is cached into _SEED_HUNT_CANDIDATES so a
        later commit can carry it straight into Stage 2 instead of
        regenerating Stage 1 from scratch."""
        s1_w = max(divisible_by, (custom_width  // 2 // divisible_by) * divisible_by)
        s1_h = max(divisible_by, (custom_height // 2 // divisible_by) * divisible_by)

        ltxv_len = int(math.ceil((seed_hunt_duration_frames - 1) / 8.0) * 8) + 1
        latent_t = ((ltxv_len - 1) // 8) + 1
        pre_latent = {"samples": torch.zeros(
            [1, 128, latent_t, s1_h // 32, s1_w // 32],
            device=mm.intermediate_device(),
        )}

        guide_data, _dw, _dh = _build_guide_data(
            tdata, start_frame, seed_hunt_duration_frames,
            frame_rate, s1_w, s1_h, resize_method, divisible_by,
            img_compression, guide_strength,
        )
        guide_data["timeline_data"] = timeline_data
        guide_data["start_frame"] = start_frame
        guide_data["duration_frames"] = seed_hunt_duration_frames
        guide_data["resize_method"] = resize_method

        motion_guide_data = _build_motion_guide_data(
            timeline_data, start_frame, seed_hunt_duration_frames,
            frame_rate, resize_method, motion_guide_on,
        )

        hunt_local_prompts, hunt_segment_lengths = _build_chunk_local_prompts(
            tdata, start_frame, seed_hunt_duration_frames, local_prompts, segment_lengths,
        )
        patched_model, positive = _encode_relay(
            model, clip, pre_latent, global_prompt, hunt_local_prompts, hunt_segment_lengths, epsilon,
        )

        combined_audio = _build_combined_audio(
            timeline_data, start_frame, ltxv_len, float(frame_rate),
        )
        audio_latent = _build_audio_latent(
            audio_vae, combined_audio, ltxv_len, frame_rate,
            custom_audio_on, generate_audio,
        )
        if "samples" in audio_latent:
            s = audio_latent["samples"]
            if custom_audio_on and lipsync:
                mask = torch.zeros(s.shape[0], s.shape[2], s.shape[3], dtype=torch.float32, device=s.device)
            else:
                mask = torch.ones(s.shape[0], s.shape[2], s.shape[3], dtype=torch.float32, device=s.device)
            audio_latent = {**audio_latent, "noise_mask": mask}
        has_audio = "samples" in audio_latent

        zero_neg = _zero_out_conditioning(positive)
        cond_pos, cond_neg = _unpack(LTXVConditioning.execute(positive, zero_neg, frame_rate))

        pos1, neg1, lat1 = cond_pos, cond_neg, pre_latent
        try:
            pos1, neg1, lat1, patched_model = _apply_guide(
                cond_pos, cond_neg, vae, pre_latent, guide_data, patched_model,
                motion_guide_data=motion_guide_data,
                ic_lora_name=ic_lora_name, ic_lora_strength=ic_lora_strength,
                scale_by=guide_scale_by, upscale_method=guide_upscale_method,
                image_attention_strength=guide_image_attn_strength,
                crop=guide_crop, auto_snap_ic_grid=guide_auto_snap_ic_grid,
                use_tiled_encode=guide_use_tiled_encode,
                tile_size=guide_tile_size, tile_overlap=guide_tile_overlap,
            )
        except Exception as exc:
            log.warning("[MuseDirector] Seed Hunt guide application failed, using plain conditioning: %s", exc)
            pos1, neg1 = cond_pos, cond_neg
            lat1 = pre_latent

        guider1 = _unpack(CFGGuider.execute(patched_model, pos1, neg1, cfg))[0]
        sampler = _unpack(KSamplerSelect.execute("euler"))[0]
        sigmas1 = _unpack(BasicScheduler.execute(patched_model, "linear_quadratic", int(seed_hunt_steps), 1.0))[0]

        if has_audio:
            av1_lat = _unpack(LTXVConcatAVLatent.execute(lat1, audio_latent))[0]
        else:
            av1_lat = lat1

        # Ambient pass — same LoRA-free-model technique as the main commit
        # path (see execute()'s chunk loop). Computed once, not per-candidate:
        # it uses a fresh empty pre_latent for visual context (not any
        # candidate's actual generated video), so it's identical regardless
        # of which of the 4 seeds is being previewed. Without this, scouted
        # previews never reflected what a commit would actually sound like.
        ambient_wav = None
        if generate_audio and audio_vae is not None:
            try:
                inner_vae_a = getattr(audio_vae, "first_stage_model", audio_vae)
                num_amb_latents = inner_vae_a.num_of_latents_from_frames(ltxv_len, float(frame_rate))
                z_ch = audio_vae.latent_channels
                a_freq = inner_vae_a.latent_frequency_bins
                amb_audio_samples = torch.zeros(
                    (1, z_ch, num_amb_latents, a_freq),
                    device=mm.intermediate_device(),
                )
                amb_noise_mask = torch.ones(
                    (1, num_amb_latents, a_freq),
                    dtype=torch.float32, device=mm.intermediate_device(),
                )
                amb_audio_lat = {"samples": amb_audio_samples, "type": "audio", "noise_mask": amb_noise_mask}
                vid_ones = torch.ones(
                    (pre_latent["samples"].shape[0], pre_latent["samples"].shape[2],
                     pre_latent["samples"].shape[3], pre_latent["samples"].shape[4]),
                    dtype=torch.float32, device=pre_latent["samples"].device,
                )
                amb_vid_lat = {"samples": pre_latent["samples"].clone(), "noise_mask": vid_ones}
                av_amb = _unpack(LTXVConcatAVLatent.execute(amb_vid_lat, amb_audio_lat))[0]
                import re as _re
                all_prompts = [global_prompt] + (local_prompts if isinstance(local_prompts, list) else [local_prompts])
                sounds_parts = []
                for _p in all_prompts:
                    for _m in _re.findall(r'\[SOUNDS?\][^\[|]*', _p, _re.IGNORECASE):
                        sounds_parts.append(_m.strip())
                amb_text = ' '.join(sounds_parts) if sounds_parts else "ambient background sounds"
                amb_cond = clip.encode_from_tokens_scheduled(clip.tokenize(amb_text))
                amb_zero_neg = _zero_out_conditioning(amb_cond)
                amb_cond_pos, amb_cond_neg = _unpack(LTXVConditioning.execute(amb_cond, amb_zero_neg, frame_rate))
                amb_model = base_model if base_model is not None else model
                amb_guider = _unpack(CFGGuider.execute(amb_model, amb_cond_pos, amb_cond_neg, cfg))[0]
                amb_sampler = _unpack(KSamplerSelect.execute("euler"))[0]
                amb_sigmas = _unpack(BasicScheduler.execute(amb_model, "linear_quadratic", int(seed_hunt_steps), 1.0))[0]
                amb_noise = _unpack(RandomNoise.execute(int(hunt_seeds[0]) + 99999))[0]
                amb_out, _ = _unpack(SamplerCustomAdvanced.execute(amb_noise, amb_guider, amb_sampler, amb_sigmas, av_amb))
                _, amb_aud = _unpack(LTXVSeparateAVLatent.execute(amb_out))
                if isinstance(amb_aud, dict) and "samples" in amb_aud:
                    amb_decoded = inner_vae_a.decode(amb_aud["samples"].cpu().float())
                    if amb_decoded.shape[1] == 1:
                        amb_decoded = amb_decoded.expand(-1, 2, -1)
                    ambient_wav = amb_decoded.cpu().float()
                    amb_sr = getattr(inner_vae_a, "output_sample_rate", 44100)
                    if amb_sr != 44100:
                        import torchaudio as _ta
                        ambient_wav = _ta.functional.resample(ambient_wav, amb_sr, 44100)
                    log.info("[MuseDirector] Seed Hunt ambient pass decoded: shape %s", list(ambient_wav.shape))
            except Exception as exc:
                log.warning("[MuseDirector] Seed Hunt ambient pass failed: %s", exc)

        previews = []
        previews_audio = []
        for i, sd in enumerate(hunt_seeds):
            # Reused as-is across candidates — same pattern Director's own real
            # Stage 1 uses (no defensive clone; av1_lat["samples"] can be a
            # NestedTensor from LTXVConcatAVLatent, which has no .clone()).
            noise1 = _unpack(RandomNoise.execute(int(sd)))[0]
            out1, _ = _unpack(SamplerCustomAdvanced.execute(noise1, guider1, sampler, sigmas1, av1_lat))
            if has_audio:
                out1_vid, aud1_out = _unpack(LTXVSeparateAVLatent.execute(out1))
            else:
                out1_vid = out1
                aud1_out = {}
            _pos1_c, _neg1_c, vid1 = _crop_conditioning(pos1, neg1, out1_vid)

            _SEED_HUNT_CANDIDATES[i] = {
                "vid1": vid1, "aud1": aud1_out,
                "pos1_c": _pos1_c, "neg1_c": _neg1_c, "seed": int(sd),
            }

            decoded = vae.decode(vid1["samples"])
            if decoded.ndim == 5:
                decoded = decoded.squeeze(0)
            if decoded.ndim == 3:
                decoded = decoded.unsqueeze(0)
            previews.append(decoded.cpu())

            # Decode this candidate's audio the same way the real pipeline does,
            # so lipsync (or generated audio) can actually be judged per seed.
            decoded_wav = None
            if has_audio and isinstance(aud1_out, dict) and "samples" in aud1_out:
                try:
                    inner_vae = audio_vae.first_stage_model
                    # Match whatever device the VAE's weights are actually on —
                    # unlike the real pipeline's single-pass flow, this runs 4x
                    # in a loop without ComfyUI's model manager relocating the
                    # VAE back to CPU in between, so it can still be on GPU here.
                    vae_device = next(inner_vae.parameters()).device
                    aud_samples = aud1_out["samples"].to(vae_device).float()
                    decoded_wav = inner_vae.decode(aud_samples)
                    if decoded_wav.shape[1] == 1:
                        decoded_wav = decoded_wav.expand(-1, 2, -1)
                    decoded_wav = decoded_wav.cpu().float()
                    audio_sr = getattr(inner_vae, "output_sample_rate", 44100)
                    if audio_sr != 44100:
                        import torchaudio
                        decoded_wav = torchaudio.functional.resample(decoded_wav, audio_sr, 44100)
                except Exception as exc:
                    log.warning("[MuseDirector] Seed Hunt candidate %d audio decode failed: %s", i + 1, exc)
                    decoded_wav = None
            if decoded_wav is None:
                decoded_wav = torch.zeros(1, 2, 1)
            if ambient_wav is not None:
                min_len = min(decoded_wav.shape[-1], ambient_wav.shape[-1])
                decoded_wav = decoded_wav[..., :min_len] + ambient_wav[..., :min_len] * 0.25
            previews_audio.append({"waveform": decoded_wav, "sample_rate": 44100})

            log.info("[MuseDirector] Seed Hunt candidate %d/4 done and cached (seed=%d, %d frames).",
                     i + 1, sd, decoded.shape[0])

        return previews, previews_audio


NODE_CLASS_MAPPINGS = {
    "MuseDirectorSamplerV2": MuseDirectorSamplerV2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseDirectorSamplerV2": "Muse Collective LTX Timeline V2",
}
