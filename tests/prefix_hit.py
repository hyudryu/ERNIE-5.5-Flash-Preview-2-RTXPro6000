#!/usr/bin/env python3
"""Prefix-cache granularity: a 3-turn chat over a document; prints prompt/cached tokens and TTFT per turn."""
import json, random, sys, time, urllib.request
r = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
W = "amber falcon river copper lantern meadow quartz harbor willow ember orchid canyon".split()
msgs = [{"role": "user", "content": " ".join(r.choice(W) for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)) + "\n\nSay OK."}]
for turn in range(4):
    body = {"model": "ernie-5.5-flash", "messages": msgs, "max_tokens": 40, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time(); ttft = None; text = ""; usage = None
    resp = urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600)
    for line in resp:
        if not line.startswith(b"data: ") or b"[DONE]" in line: continue
        d = json.loads(line[6:])
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            ttft = ttft or time.time() - t; text += d["choices"][0]["delta"]["content"]
        usage = d.get("usage") or usage
    c = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"turn {turn}: prompt {usage['prompt_tokens']} cached {c} recomputed {usage['prompt_tokens'] - (c or 0)} ttft {ttft:.2f}s", flush=True)
    msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": f"Tell me one fact about the number {turn + 3} in two sentences."}]
