# Muse Collective LTX Timeline V2

This is a fork of [muse-collective-26/muse-ltx-timeline](https://github.com/muse-collective-26/muse-ltx-timeline) (V1). V1 is untouched and still included as-is — this repo adds a second node, **Muse Collective LTX Timeline V2**, with one new feature: **Seed Hunt**.

## Why

LTX 2.3 is far more sensitive to seed than to prompt wording — the fastest path to a good result is usually trying a handful of seeds cheaply and picking the best one, rather than iterating on the prompt. Seed Hunt bakes that workflow into the node itself, using the node's own real timeline data (images, audio, motion guide) so multi-segment timelines just work with no extra wiring.

## How it works

- Flip **`seed_hunt`** on, leave all four **`use_seed_hunt_1..4`** toggles off, and run — the node generates 4 cheap, low-resolution Stage-1-only candidates (seeds `seed_hunt_1..4`) instead of running the full pipeline. Output ports `seed_hunt_preview_1..4` show each candidate.
- Look at the 4 previews, then flip exactly one `use_seed_hunt_N` on and run again — the node now runs the completely normal full pipeline (Stage 1 + Stage 2, your real resolution/duration/audio settings), using that candidate's seed for **both** stages.
- Re-running with only a toggle changed doesn't regenerate the 4 candidates — they're cached against every other input, so flipping a toggle is instant.

## Settings added on top of V1

| Setting | Purpose |
|---|---|
| `seed_hunt` | Master toggle for the feature. Off = identical to V1. |
| `seed_hunt_1..4` | The 4 candidate seeds to compare. |
| `use_seed_hunt_1..4` | Flip exactly one on to select a candidate and run the full pipeline with it. |
| `seed_hunt_steps` | Sampler steps for the cheap scouting pass (independent of `stage1_steps`). |
| `seed_hunt_duration_frames` | Scouting clip length — independent of your real duration. |
| `seed_hunt_scale` | Scouting resolution as a fraction of `custom_width`/`custom_height` (default 0.25) — follows whatever orientation (portrait/landscape) you're actually using, without affecting the committed run's real resolution. |

## Also included

`Muse Seed Scout` — an earlier, standalone version of the same idea that runs *outside* Director and only takes a single reference image (rather than the full timeline). Superseded by V2's built-in Seed Hunt for anything with more than one timeline segment, but kept as a lighter-weight option for single-image setups.

## One deliberate difference from V1

V1 uses `seed - 1` for Stage 2's noise seed (separate from Stage 1's). V2 uses the **same** seed for both stages, so a Seed Hunt candidate's look carries through faithfully to the full-resolution committed run.
