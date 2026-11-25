#!/usr/bin/env python3
"""Memory stress: long prefills alone and concurrently, then decode on top.
Usage: stress.py "<n_requests>x<prompt_words>[,...]"  e.g. "1x247000,3x213000,8x85000" """
import json, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
WORDS = "amber falcon river copper lantern meadow quartz harbor willow ember orchid canyon glacier saffron".split()
def one(args):
    n_words, seed = args
    r = random.Random(seed); code = r.randint(10000, 99999)
    w = [r.choice(WORDS) for _ in range(n_words)]; w.insert(n_words // 20, f". The secret code is {code}. ")
    body = {"model": "ernie-5.5-flash", "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": " ".join(w) + "\n\nWhat is the secret code in the text above? Give the number, then write three sentences about lighthouses."}]}
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
        return (d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"], str(code) in d["choices"][0]["message"]["content"], round(time.time() - t, 1))
    except Exception as e:
        return ("ERR", str(e)[:200])
seed = int(time.time())
for spec in sys.argv[1].split(","):
    n, words = map(int, spec.split("x")); seed += 100
    with ThreadPoolExecutor(n) as ex: res = list(ex.map(one, [(words, seed + i) for i in range(n)]))
    print(spec, res, flush=True)
