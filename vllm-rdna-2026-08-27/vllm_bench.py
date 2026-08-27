#!/usr/bin/env python3
"""Standard ai1 vLLM benchmark probe (Qwen3.8-27B lane).

Harness shape matches ai1-model-testing section 6:
  ~200-tok prompt, 128 out, temp 0, thinking OFF (enable_thinking false).
Adds: PP (unique nonce, no prefix-cache hits), TG (scaled), TTFT (streaming),
load-time + VRAM deltas are captured by the caller.

Usage: vllm_bench.py <base_url> <model> [reps=3] [max_tokens=128]
Prints JSON-ish lines per rep + median summary.
Requires: python3 stdlib only (urllib).
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 128

# ~200-token prompt (Qwen tokenizer ≈ 0.6 tok/word for English)
PROMPT = (
    "Write a detailed technical analysis of the tradeoffs between monolithic "
    "and microservice architectures for a small team maintaining a media "
    "automation stack. Consider deployment complexity, failure isolation, "
    "resource utilization on a homelab server, observability requirements, "
    "and the operational burden of keeping dependencies updated. Compare how "
    "each approach handles a sudden spike in CPU-bound transcoding work, and "
    "discuss what the migration path looks like when moving from one style to "
    "the other. Include specific recommendations about when staying monolithic "
    "is the better engineering decision, and reference concrete examples from "
    "self-hosted software ecosystems to support your argument."
)

LONG = PROMPT * 6  # ~1200 tokens for PP probe
SHORT = "Repeat the word test five times."


def call(content, max_tokens, stream=False, nonce=None):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": stream,
    }
    if nonce:
        body["messages"] = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": f"Continue. [nonce {nonce}]"},
        ]
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def call_stream(content, max_tokens):
    """Returns (ttft_s, usage_dict). Times first content-bearing delta."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                choices = chunk.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    ttft = time.time() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    return ttft, usage


def main():
    # warm-up (first request pays inductor/JIT compile + model wake)
    call("hi", 4)
    pp_list, tg_list, ttft_list = [], [], []

    for i in range(REPS):
        # PP: long prompt, 1 token out, unique nonce defeats prefix cache
        nonce = f"{time.time_ns()}"
        t0 = time.time()
        r = call(LONG, 1, nonce=nonce)
        dt = time.time() - t0
        p = r["usage"]["prompt_tokens"]
        pp = p / dt
        pp_list.append(pp)

        # TTFT: streaming, ~200-tok prompt, 64 out
        ttft, u = call_stream(PROMPT, 64)
        ttft_list.append(ttft)

        # TG: 200-tok prompt, MAX_TOK out; subtract scaled prefill time
        t0 = time.time()
        r2 = call(PROMPT, MAX_TOK)
        dt2 = time.time() - t0
        p2 = r2["usage"]["prompt_tokens"]
        c2 = r2["usage"]["completion_tokens"]
        prefill_s = p2 / pp
        tg = c2 / max(dt2 - prefill_s, 1e-6)
        tg_list.append(tg)
        print(f"rep{i+1}: PP={pp:.1f} (p={p}) | TTFT={ttft*1000:.0f}ms | "
              f"TG={tg:.1f} (c={c2}) | finish={r2.get('choices',[{}])[0].get('finish_reason')}",
              flush=True)

    pp_list.sort(); tg_list.sort(); ttft_list.sort()
    print(f"MEDIAN: PP={pp_list[len(pp_list)//2]:.1f} tok/s | "
          f"TG={tg_list[len(tg_list)//2]:.1f} tok/s | "
          f"TTFT={ttft_list[len(ttft_list)//2]*1000:.0f} ms "
          f"(reps={REPS}, max_tokens={MAX_TOK}, temp=0, thinking=off)")


if __name__ == "__main__":
    main()
