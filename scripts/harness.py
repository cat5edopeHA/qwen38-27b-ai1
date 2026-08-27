#!/usr/bin/env python3
"""Harness v2: faster probes, skip ultra-long prefill, focus on TG at various ctx depths.
Usage: harness.py PORT OUTFILE [ctx_size]
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/tmp/harness-out.json"
CTX  = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
BASE = f"http://127.0.0.1:{PORT}"

def get(path):
    try:
        req = urllib.request.Request(BASE + path, method="GET")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def post(path, data, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body), time.time() - t0
    except Exception as e:
        return {"error": str(e)}, time.time() - t0

for i in range(120):
    try:
        req = urllib.request.Request(BASE + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            if r.status == 200: break
    except Exception: pass
    time.sleep(5)
else:
    print("HEALTH_TIMEOUT"); sys.exit(1)
print("server ready", flush=True)

results = {}
props = get("/props")
results["props_n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
print(f"n_ctx={results['props_n_ctx']}", flush=True)

def chat(n_prompt_target, label, max_tokens=128, seed=42):
    actual_target = min(n_prompt_target, int(CTX * 0.70))
    filler = ("The quick brown fox jumps over the lazy dog while reviewing quarterly "
              "telemetry from the orbital array, noting that sensor calibration drift "
              "exceeds tolerance by 0.03 percent on channels seven through twelve. ") * max(1, actual_target // 24)
    payload = {
        "messages": [{"role": "user", "content": filler +
                      "\n\nSummarize the passage in exactly three sentences."}],
        "max_tokens": max_tokens, "temperature": 0, "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
    }
    r, wall = post("/v1/chat/completions", payload)
    if "error" in r:
        print(f"ERROR {label}: {r['error']}", flush=True)
        results[label] = {"error": r["error"]}; return
    t = r.get("timings", {})
    pt = t.get("prompt_n", r.get("usage", {}).get("prompt_tokens", 0))
    ct = t.get("predicted_n", r.get("usage", {}).get("completion_tokens", 0))
    out = {
        "label": label, "prompt_tokens": pt, "completion_tokens": ct,
        "wall_s": round(wall, 2),
        "pp_tok_s": round(t.get("prompt_per_second", 0), 1),
        "tg_tok_s": round(t.get("predicted_per_second", 0), 1),
        "pp_ms": round(t.get("prompt_ms", 0), 0),
        "tg_ms": round(t.get("predicted_ms", 0), 0),
        "finish": r.get("choices", [{}])[0].get("finish_reason"),
    }
    results[label] = out
    print(f"  {label}: pp={out['pp_tok_s']}t/s tg={out['tg_tok_s']}t/s "
          f"(pt={pt} ct={ct} wall={out['wall_s']}s)", flush=True)

# Probes: short (TG-focused), medium, and long (but not >100K to stay in timeout)
chat(300, "short_tg128")
if CTX >= 16384:
    chat(8192, "pp8k_tg128")
if CTX >= 65536:
    chat(32768, "pp32k_tg64", max_tokens=64)
if CTX >= 131072:
    chat(65536, "pp64k_tg64", max_tokens=64)

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved -> {OUT}", flush=True)
