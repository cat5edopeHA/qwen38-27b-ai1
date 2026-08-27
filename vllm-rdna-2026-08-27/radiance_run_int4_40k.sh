#!/bin/bash
# Radiance vLLM rootful launcher — Qwen3.8-27B INT4 on GPU1, 40K ctx, no MTP (capacity row)
set -e
mkdir -p /srv/ai/wip/radiance-cache
chmod 777 /srv/ai/wip/radiance-cache
sudo -n podman rm -f vllm-radiance >/dev/null 2>&1 || true
sudo -n podman pull -q docker.io/stilldeadcode/vllm-radiance:0.9.3
VGID=$(getent group video | cut -d: -f3)
sudo -n podman run -d --name vllm-radiance \
  --ulimit memlock=-1:-1 \
  --device /dev/kfd --device /dev/dri \
  --group-add "$VGID" \
  --shm-size 4g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /srv/ai/models:/models:ro,z \
  -v /srv/ai/wip/radiance-cache:/cache:z \
  -p 8002:8000 \
  -e HIP_VISIBLE_DEVICES=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 -e VLLM_ROCM_USE_AITER_MLA=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_FP8BMM=0 \
  -e VLLM_ROCM_USE_AITER_FP4BMM=0 -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -e NCCL_PROTO=Simple \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton -e AITER_ROOT_DIR=/cache/aiter \
  -e TRITON_CACHE_AUTOTUNING=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  docker.io/stilldeadcode/vllm-radiance:0.9.3 \
  /models/qwen38-27b-int4-awq \
  --served-model-name qwen38-27b-int4-awq \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --max-model-len 40960 --max-num-seqs 16 \
  --gpu-memory-utilization 0.97 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --enable-prefix-caching --mamba-cache-mode align \
  --no-async-scheduling \
  --host 0.0.0.0 --port 8000
echo CONTAINER_STARTED
