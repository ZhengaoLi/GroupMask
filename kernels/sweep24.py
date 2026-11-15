"""Correctness + tuning of the Triton 2:4 GEMV on LLaMA-2-7B shapes, side by side with dense cuBLAS
and the tuned group-sparse 1x64 kernel (random 50% 1x64 masks; per-batch best config for each)."""
import json, sys
import torch, triton
from sparse24 import pack24, gemv24
from gsparse import pack, rows_gemv, row_order

batches = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4,8,16").split(",")]
torch.manual_seed(0)
print(torch.cuda.get_device_name(), flush=True)
shapes = [(4096, 4096, 128), (11008, 4096, 64), (4096, 11008, 32)]
bench = lambda f: triton.testing.do_bench(f, warmup=10, rep=50)
data = []
for (O, I, n) in shapes:
    W = torch.randn(O, I, dtype=torch.bfloat16, device="cuda") * 0.02
    v24, m24, W24 = pack24(W)
    bm = torch.rand(O, I // 64, device="cuda") < 0.5
    Wg = (W.reshape(O, I // 64, 64) * bm[:, :, None]).reshape(O, I)
    vg, cg, pg = pack(Wg, bm, 1, 64)
    data.append((O, I, n, W, W24, v24, m24, Wg, vg, cg, pg, row_order(pg)))
    # memory
mem_d = sum(O * I * 2 * n for (O, I, n, *_) in data)
mem_24 = sum((d[5].numel() * 2 + d[6].numel()) * d[2] for d in data)
mem_g = sum((d[8].numel() * 2 + d[9].numel() * 4 + d[10].numel() * 4) * d[2] for d in data)
print(f"memory vs dense: 2:4 {mem_24/mem_d:.4f}  group1x64 {mem_g/mem_d:.4f}", flush=True)
best = {}
for B in batches:
    xs = [torch.randn(B, d[1], dtype=torch.bfloat16, device="cuda") for d in data]
    td = sum(bench(lambda: torch.nn.functional.linear(x, d[3])) * d[2] for x, d in zip(xs, data))
    r24, rg = {}, {}
    for R in [1, 2, 4, 8]:
        for blk in [128, 256, 512, 1024]:
            if B * R * blk // 2 > 8192:
                continue
            for nw in [1, 2, 4, 8]:
                t, ok = 0.0, True
                for x, d in zip(xs, data):
                    O, I, n = d[:3]
                    f = lambda: gemv24(d[5], d[6], x, O, (R, blk, nw))
                    try:
                        if not r24:
                            ref = torch.nn.functional.linear(x.float(), d[4].float())
                            ok = (f().float() - ref).abs().max().item() <= 2e-2 * ref.abs().max().item()
                            if not ok:
                                print("MISMATCH 2:4", R, blk, nw, flush=True)
                        t += bench(f) * n
                    except Exception as e:
                        ok = False
                        break
                if ok:
                    r24[(R, blk, nw)] = t
    for R in [1, 2, 4, 8]:
        for bn in [2, 4, 8, 16]:
            if B * R * bn * 64 > 8192:
                continue
            for nw in [1, 2, 4]:
                for srt in [False, True]:
                    t = 0.0
                    for x, d in zip(xs, data):
                        O, I, n = d[:3]
                        t += bench(lambda: rows_gemv(d[8], d[9], d[10], x, O, 64, R, bn, nw,
                                                     order=d[11] if srt else None)) * n
                    rg[(R, bn, nw, srt)] = t
    (c24, t24), (cg_, tg) = min(r24.items(), key=lambda kv: kv[1]), min(rg.items(), key=lambda kv: kv[1])
    best[B] = list(c24)
    print(f"B={B:2d}: dense {td:6.2f} ms | 2:4 triton {t24:6.2f} ({td/t24:.2f}x) {c24} | "
          f"group 1x64 {tg:6.2f} ({td/tg:.2f}x) {cg_} | group vs 2:4 {t24/tg:.2f}x", flush=True)
print("BEST24", json.dumps(best))
