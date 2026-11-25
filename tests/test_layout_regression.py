#!/usr/bin/env python3
"""Layout regression check: first-token logprobs and needle retrieval at several
context lengths. Usage: logprob_check.py save ref.json | logprob_check.py cmp ref.json
The first generated token comes from the target model at temperature 0, so its
top logprobs depend only on the prompt and on what the KV caches hold."""
import json, random, sys, urllib.request
URL = "http://localhost:8000/v1/chat/completions"
WORDS = ("amber falcon river copper lantern meadow quartz harbor willow ember orchid canyon "
         "glacier saffron tundra violet granite comet maple prairie sable cobalt juniper").split()

def doc(n_words, seed, needle_at):
    r = random.Random(seed); w = [r.choice(WORDS) for _ in range(n_words)]
    code = 10000 + seed * 7919 % 89999
    w.insert(int(n_words * needle_at), f". The secret code is {code}. ")
    return " ".join(w), code

def ask(prompt, max_tokens, logprobs):
    body = {"model": "ernie-5.5-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if logprobs: body.update(logprobs=True, top_logprobs=5)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))

def run():
    out = {}
    for n_words, seed, at in [(50, 1, 0.5), (6000, 2, 0.1), (40000, 3, 0.05), (40000, 4, 0.9), (130000, 5, 0.03)]:
        d, code = doc(n_words, seed, at)
        p = d + "\n\nWhat is the secret code mentioned in the text above? Reply with the number only."
        a = ask(p, 1, True); b = ask(p, 12, False)
        top = a["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        out[f"{n_words}w"] = {"prompt_tokens": a["usage"]["prompt_tokens"], "code": code,
                              "answer": b["choices"][0]["message"]["content"].strip(),
                              "top": [[t["token"], round(t["logprob"], 4)] for t in top]}
        print(n_words, a["usage"]["prompt_tokens"], code, out[f"{n_words}w"]["answer"], out[f"{n_words}w"]["top"][:3], flush=True)
    return out

mode, path = sys.argv[1], sys.argv[2]
res = run()
if mode == "save":
    json.dump(res, open(path, "w"), indent=1)
else:
    ref = json.load(open(path)); worst = 0.0
    for k, v in res.items():
        r = dict(map(tuple, ref[k]["top"])); c = dict(map(tuple, v["top"]))
        d = max((abs(r[t] - c[t]) for t in r if t in c), default=9.9)
        ok = str(v["code"]) in v["answer"] and ref[k]["top"][0][0] == v["top"][0][0]
        worst = max(worst, d); print(f"{k}: needle+top1 {'OK' if ok else 'FAIL'}, max |dlogprob| {d:.4f}")
    print("worst", worst)
