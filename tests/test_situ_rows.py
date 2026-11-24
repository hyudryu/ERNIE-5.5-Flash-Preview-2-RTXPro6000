"""Compare the fused row-wise SITU+quant kernel with the stock two-kernel path."""
import time, torch
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import situ_mul_quant_fp8_packed_rows
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8_packed_for_deepgemm

torch.manual_seed(0)
dev = "cuda"
beta, lbeta = 4.0, 25.0
for (m_sum, d, nreal, std) in [(5120, 1024, 80, 1.0), (32896, 1024, 640, 6.0), (90112, 1024, 81920, 3.0)]:
    x = (torch.randn(m_sum, 2 * d, device=dev) * std).to(torch.bfloat16)
    x[0, :16] = 0.0
    rows = torch.randperm(m_sum, device=dev)[:nreal].to(torch.int32)
    topk = torch.randint(0, 512, (nreal,), device=dev, dtype=torch.int64)
    # invalid slots: negative expert id and a garbage row index that must be ignored
    topk[::7] = -1
    garbage = rows.clone(); garbage[::7] = 2**30
    valid = topk >= 0
    # reference
    ref_act = torch.empty(m_sum, d, dtype=torch.bfloat16, device=dev)
    torch.ops._C.situ_and_mul(ref_act, x, beta, lbeta)
    ref_q = torch.empty(m_sum, d, dtype=torch.float8_e4m3fn, device=dev)
    ref_q, ref_s = per_token_group_quant_fp8_packed_for_deepgemm(ref_act, 128, use_ue8m0=True, out_q=ref_q)
    out_q = torch.zeros(m_sum, d, dtype=torch.float8_e4m3fn, device=dev)
    q, s = situ_mul_quant_fp8_packed_rows(x, out_q, garbage, topk, beta, lbeta, 128)
    r = rows.long()[valid]
    assert int(q.view(torch.uint8)[rows.long()[~valid]].sum()) == 0, 'invalid slots must stay untouched'
    qa = q.view(torch.uint8)[r]; qb = ref_q.view(torch.uint8)[r]
    sa = s[r]; sb = ref_s[r]
    nq = int((qa != qb).sum()); ns = int((sa != sb).sum())
    da = q.float()[r] * 1.0; db = ref_q.float()[r] * 1.0
    print(f"M_sum={m_sum} d={d} real={nreal} std={std}: q-byte mismatches {nq}/{qa.numel()} scale mismatches {ns}/{sa.numel()} "
          f"scale stride ours {s.stride()} ref {ref_s.stride()} shape {tuple(s.shape)} {tuple(ref_s.shape)}")
    if nq:
        idx = (qa != qb).nonzero()[:5]
        print("  examples (ours, ref):", [(float(da[i, j]), float(db[i, j])) for i, j in idx.tolist()])
    for name, fn in (("stock", lambda: (torch.ops._C.situ_and_mul(ref_act, x, beta, lbeta), per_token_group_quant_fp8_packed_for_deepgemm(ref_act, 128, use_ue8m0=True, out_q=ref_q))),
                     ("fused", lambda: situ_mul_quant_fp8_packed_rows(x, out_q, garbage, topk, beta, lbeta, 128))):
        for _ in range(20): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(200): fn()
        torch.cuda.synchronize(); print(f"  {name}: {(time.perf_counter() - t) / 200 * 1e6:.1f} us")
