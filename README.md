# Qwen3.8-27B on AMD Radeon AI PRO R9700 — Vulkan vs ROCm vs HipFire

**Bottom line: on this card, llama.cpp's Vulkan (RADV) backend beats ROCm (HIP) for decode on the Qwen3.8-27B Q4_K_M 262K recipe — same build commit, same harness. Winner config stays MTP3 on both backends. Both llama.cpp backends beat HipFire 0.3.0 by 54–80%.**

Measured on a single Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GiB), llama.cpp `d222767` (same commit, HIP vs Vulkan-only builds), full 262,144-token q8_0 KV context, Q4_K_M base model (16.46 GB) + baked-in MTP head.

## Headline results (TG tok/s — decode)

| row | short (479t) | 8K | 32K | 64K |
|---|---:|---:|---:|---:|
| **mtp3 (winner)** | **68.5** / 55.1 | **62.8** / 50.9 | **50.5** / 40.2 | **40.9** / 31.2 |
| nospec | 28.9 / 28.8 | 30.1 / 27.6 | 27.6 / 24.0 | 25.0 / 19.4 |
| HipFire best lane | 40.6 | 34.9 | 28.0 | 26.5 |

First number = Vulkan, second = ROCm. HipFire = best of AR/DFlash/ngram lanes (hipfire 0.3.0, MQ4V2).

- Vulkan MTP3 decode: **+24% short, +23% @8K, +26% @32K, +31% @64K vs ROCm**.
- Vulkan prefill: slower on short prompts (−10–20%), **faster on deep prompts** (+13% @32K, +20–29% @64K).
- MTP depth optimum = 3 on both backends; on Vulkan depth 4/5 win at ≥8K ctx but collapse on short prompts (backend-specific draft behavior).
- Full tables: [BENCHMARKS.md](BENCHMARKS.md) · reproduce: [RECIPES.md](RECIPES.md) · pitfalls: [ISSUES.md](ISSUES.md)

## Files

- `BENCHMARKS.md` — complete 10-row sweep, PP + TG, all spec modes (MTP 1–5, ngram-simple/mod, DFlash2)
- `RECIPES.md` — Vulkan build, bench harness, production llama-swap routes
- `ISSUES.md` — every trap hit (device indexing, env pinning, sampling caveats, proxy gotchas)
- `scripts/` — the exact harness, sweep script, route config, and raw result JSONs

## Trust notes

- Same llama.cpp commit (`d222767`) for both backends; Vulkan build = `-DGGML_VULKAN=ON -DGGML_HIP=OFF`, RADV mesa 26.1.8.
- Same harness, same prompts, same seed: fox-filler corpus (479 / 12,981 / 39,428 / 52,386 tokens), temp 0, seed 42, thinking disabled.
- Production sampling filters (temp 0.7–1.0, presence penalties) cost ~20% decode vs the temp-0 bench — expected, not a bug.
- No model weights are included; the Q4_K_M GGUF is from the official unsloth repo.

License: Apache-2.0
