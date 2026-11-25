"""Routing kernel: identical to torch on finite rows; distinct in-range ids on non-finite rows."""
import torch
from vllm_ernie.models.ernie5_5.nvidia import moe

torch.manual_seed(0)
E, K, T = 512, 10, 4096
dev = "cuda"
g = torch.rand(T, E, device=dev) * 2          # sigmoid(l0)+sigmoid(l1) in [0, 2]
g[5] = 1.0                                     # all-ties row
bias = torch.randn(E, device=dev) * 0.05
scale = torch.rand(E, device=dev) + 0.5
w_k, i_k = moe.ernie5_5_routing(None, g, K, True, bias=bias, scale=scale)
w_t, i_t = moe._ernie5_5_routing_torch(g, K, True, bias, scale)
s_k, o_k = i_k.sort(-1); s_t, o_t = i_t.sort(-1)
print("finite rows: id sets equal:", bool((s_k == s_t).all()), "| max weight diff:", float((w_k.gather(1, o_k) - w_t.gather(1, o_t)).abs().max()))
bad = g.clone()
bad[0] = float("nan"); bad[1, ::3] = float("nan"); bad[2] = float("inf"); bad[3] = float("-inf"); bad[4, :7] = float("nan")
w, i = moe.ernie5_5_routing(None, bad, K, True, bias=bias, scale=scale)
ok_range = bool(((i >= 0) & (i < E)).all())
distinct = bool((i.sort(-1).values.diff(dim=-1) > 0).all())
print("non-finite rows: ids in range:", ok_range, "| distinct per row:", distinct, "| row0 ids:", i[0].tolist())
s2, _ = i[6:].sort(-1)
print("untouched rows unchanged:", bool((s2 == s_t[6:]).all()))
