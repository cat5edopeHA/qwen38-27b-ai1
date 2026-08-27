# Recipes — Qwen3.8-27B Vulkan build, bench harness, production routes

## 1. Vulkan-only llama.cpp build (same commit as the HIP build)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout d222767
# Fedora 44 packages (shaderc no longer exists on F44; glslc is its own package):
sudo dnf install -y glslc vulkan-headers vulkan-loader-devel glslang spirv-headers-devel
cmake -B build-vulkan -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j $(nproc) --target llama-server llama-bench
```

- RADV (mesa) 26.1.8 used here. `radv is not a conformant Vulkan implementation` warning is cosmetic.
- **Device numbering:** with an Intel iGPU present, `--list-devices` shows `Vulkan0 = iGPU`, `Vulkan1/2 = R9700s`.
  Pin a card with `GGML_VK_VISIBLE_DEVICES=1` (+ `HIP_VISIBLE_DEVICES=99`). `VK_VISIBLE_DEVICES` (loader-level) does NOT work.

## 2. Bench sweep (llama-server timings, 262K single card)

Model: `Qwen3.8-27B-UD-Q4_K_M.gguf` (16.46 GB, unsloth, MTP head baked in).

```bash
GGML_VK_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=99 ./build-vulkan/bin/llama-server \
  -m ~/Models/qwen38-27b-q4km/Qwen3.8-27B-UD-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 --port 8091 --no-webui -c 262144 \
  --spec-type draft-mtp --spec-draft-n-max 3
python3 scripts/harness.py 8091 result.json 262144
```

- Read `timings.prompt_per_second` / `timings.predicted_per_second` from `/v1/chat/completions`.
- Sweep `--spec-draft-n-max 1..5`; MTP3 is the optimum at 262K on this card (both backends).
- Fit: full 262K + q8_0 KV + Q4_K_M = 24.7/31.9 GiB loaded, ~7.2 GiB spare — fits one 32 GiB card.
- `-b 1024 -ub 1024` does not help on Vulkan (38.3 vs 40.9 t/s @64K) — keep defaults.

## 3. Production llama-swap route (think/nothink pair)

```yaml
healthCheckTimeout: 500
startPort: 18080
logLevel: info
models:
  qwen38-27b-think:
    name: "Qwen3.8 27B Q4 Think (Vulkan single, MTP3, 262K)"
    env: ["GGML_VK_VISIBLE_DEVICES=1", "HIP_VISIBLE_DEVICES=99"]
    cmd: |
      /path/to/build-vulkan/bin/llama-server
      -m /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf
      -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 -c 262144
      --spec-type draft-mtp --spec-draft-n-max 3
      --port ${PORT} --no-webui
    filters:
      setParams:
        temperature: 1.0
        top_p: 0.95
        top_k: 20
        min_p: 0.0
        presence_penalty: 0.0
        repetition_penalty: 1.0
    ttl: 0
  qwen38-27b-nothink:
    # same cmd/env, filters:
    #   temperature: 0.7, top_p: 0.80, top_k: 20, min_p: 0.0,
    #   presence_penalty: 1.5, repetition_penalty: 1.0,
    #   chat_template_kwargs: {enable_thinking: false}
    ttl: 0
```

Launch: `llama-swap --config config.yaml --listen 0.0.0.0:8080`. Verify MTP is live via the response `timings.draft_n` / `draft_n_accepted` fields (not just speed).

**Production sampling filters cost ~20% decode vs the temp-0 bench** (52.5 vs 68.5 t/s short) — expected.

## 4. Reproduce the comparison

1. Build both backends at the same commit (HIP: `-DGGML_HIP=ON`; Vulkan: above).
2. Run the same sweep script twice, swapping the binary + device env (HIP: `HIP_VISIBLE_DEVICES=0`; Vulkan: `GGML_VK_VISIBLE_DEVICES=1`).
3. Compare row-by-row with `scripts/results/` JSONs.

DFlash2 row requires the z-lab `dflash2` fork (stock loader is DFlash1-era and rejects the tensor count) — flagged as fork-vs-stock in the tables.
