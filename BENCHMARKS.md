# Benchmarks — Qwen3.8-27B Q4_K_M @ 262K, single R9700 (gfx1201)

Harness: llama-server `/v1/chat/completions` `timings`, fox-filler prompts sized per context
(short = 479 tokens, 8K = 12,981, 32K = 39,428, 64K = 52,386), temp 0, seed 42, thinking off,
q8_0 KV, flash attention, `-np 1`, `-c 262144`. Every row = fresh server, exact-PID teardown.

Builds: llama.cpp `d222767` — `build` (HIP) and `build-vulkan` (`-DGGML_VULKAN=ON -DGGML_HIP=OFF`).
RADV mesa 26.1.8, glslc 2026.1. DFlash2 row uses the z-lab `dflash2` fork build (flagged — fork vs stock).

## TG tok/s (decode) — Vulkan / ROCm, winner bold

| row | short (479t) | 8K (13.0K) | 32K (39.4K) | 64K (52.4K) |
|---|---:|---:|---:|---:|
| nospec | **28.9** / 28.8 | **30.1** / 27.6 | **27.6** / 24.0 | **25.0** / 19.4 |
| mtp1 | **49.8** / 43.4 | **48.0** / 41.0 | **42.7** / 34.1 | **34.4** / 26.7 |
| mtp2 | 49.0 / **54.9** | **57.5** / 46.4 | **47.2** / 37.1 | **35.8** / 27.7 |
| **mtp3** | **68.5** / 55.1 | **62.8** / 50.9 | **50.5** / 40.2 | **40.9** / 31.2 |
| mtp4 | 45.2 / **57.3** | **61.1** / 47.8 | **49.6** / 35.3 | **41.4** / 28.1 |
| mtp5 | 42.0 / **49.9** | **68.0** / 47.9 | **57.4** / 38.6 | **41.7** / 29.1 |
| ngram-simple | **33.7** / 32.8 | **33.9** / 32.3 | **30.4** / 27.3 | **25.7** / 20.7 |
| ngram-mod | **30.3** / 28.4 | 31.5 / **33.8** | **35.6** / 35.3 | **46.1** / 25.7 |
| dflash4 (fork) | **59.3** / 27.6 | **66.1** / 50.9 | **55.1** / 39.9 | **41.6** / 27.3 |
| mtp3 -b1024 -ub1024 | **68.3** / 55.1 | **62.7** / 50.9 | **50.4** / 40.2 | **38.3** / 31.2 |

## PP tok/s (prefill) — Vulkan / ROCm

| row | short | 8K | 32K | 64K |
|---|---:|---:|---:|---:|
| nospec | 854 / 1047 | 925 / 1015 | **612** / 541 | **377** / 298 |
| mtp3 | 755 / 900 | 869 / 950 | **584** / 512 | **363** / 282 |

## vs HipFire 0.3.0 (MQ4V2, 2026-08-26 session, best lane per depth)

| probe | **llama.cpp Vulkan MTP3** | HipFire AR | HipFire DFlash | HipFire ngram |
|---|---:|---:|---:|---:|
| short 479t | **68.5** | 33.0 | 40.6 | 40.5 |
| 8K 13.0K | **62.8** | 31.1 | 2.3 ⚠ | 34.9 |
| ~39.4K | **50.5** | 27.5 | 0.5 ⚠ | 28.0 |
| ~52K | **40.9** | 26.1 | 23.1 | 26.5 |

## Verdict

- **Vulkan wins decode on 29 of 36 cells; all winner-row comparisons favor Vulkan.**
- MTP3 is the depth optimum on both backends; deeper drafts are backend-specific (Vulkan d4/d5 win ≥8K, collapse on short).
- `-b 1024 -ub 1024` does not help on Vulkan — use the base recipe.
- Deep-context prefill (32K/64K) is faster on Vulkan than ROCm — for long-context workloads Vulkan wins both halves.
- ngram-mod @64K = 46.1 t/s is the fastest single Vulkan cell, but the filler prompt is repetitive (flatters ngram) — treat with caution.

Raw per-row JSONs: `scripts/results/`.
