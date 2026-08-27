# Issues & pitfalls — all hit in production on this exact workflow

1. **Intel iGPU steals Vulkan device 0.** `--list-devices` first; R9700s are 1 and 2. Pinning: `GGML_VK_VISIBLE_DEVICES=<idx>` (llama.cpp env). Loader-level `VK_VISIBLE_DEVICES` does NOT work. `GGML_VK_VISIBLE_DEVICES=9` is a valid way to kill the Vulkan backend entirely.
2. **`-dev A,B` does not split** in llama-bench — it benches each device sequentially (single-card + CPU-spill rows). For true splits omit `-dev` and hide the other backend via env.
3. **Deep MTP is backend-specific.** Vulkan d4/d5 win at ≥8K ctx but collapse on short prompts; on ROCm d4/d5 lose everywhere vs d3. Sweep per backend, never import another host's depth optimum.
4. **`-b 1024 -ub 1024` regresses deep-context decode on Vulkan** (38.3 vs 40.9 t/s @64K) — keep defaults.
5. **Production sampling filters override client params** (llama-swap `setParams`): a temp-0 probe through a temp-0.7 route runs at 0.7. Bench against the raw server for backend truth; expect ~20% decode cost on filtered routes.
6. **OpenAI-compatible proxies require `model` in the request body** — omitting it yields `404 no model id could be identified` (llama-swap and hipfire alike).
7. **Repetitive bench prompts flatter ngram speculators** (ngram-mod @64K hit 46.1 t/s on the repetitive fox-filler corpus). Use varied prose for ngram rows.
8. **DFlash2 needs the z-lab fork + stochastic sampling.** Stock llama.cpp (d222767) rejects its tensor count; greedy (temp 0) collapses acceptance. If you see DFlash "collapse" numbers, check temp.
9. **`pgrep -f <pattern>` over SSH self-matches the invoking shell** — your own ssh command line contains the pattern and never "finishes". Use `pgrep -x llama-server` or bracket patterns.
10. **Backgrounded remote sweeps die with the ssh session** unless launched `setsid bash script.sh > log 2>&1 < /dev/null &`.
11. **Mid-prefill "OOM" is usually a client timeout.** A flat 550 s timeout killed a 131K probe at 55K/131K tokens while the log showed healthy progress. Scale probe timeout with context; grep the server log for alloc errors before declaring OOM.
12. **`curl -s` health checks pass on 503** — always use `curl -sf` when waiting for model load.
13. **Stale server on the bench port poisons results** — kill the previous server (exact PID, SIGTERM) before every launch and on every bailout.
