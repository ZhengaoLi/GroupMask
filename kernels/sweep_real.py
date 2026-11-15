"""Tune the 1xK multi-row kernel (K from the mask metadata, e.g. 64 or 256) on REAL learned masks (8 of 32 LLaMA-2-7B blocks, all 7 projections each),
with and without length-sorted row order. Prints best (R, BLOCK_N, warps) per batch and the speedup vs dense."""
import json, sys
import numpy as np
import torch, triton
from gsparse import pack, rows_gemv, row_order

masks = sys.argv[1]
batches = [int(b) for b in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8,16").split(",")]
meta = json.load(open(masks.replace(".npz", ".json")))["layers"]
z = np.load(masks)
K = meta[next(iter(meta))]["group"][1]
assert meta[next(iter(meta))]["group"][0] == 1
names = [n for n in meta if int(n.split(".layers.")[1].split(".")[0]) % 4 == 1]   # blocks 1,5,...,29
torch.manual_seed(0)
print(torch.cuda.get_device_name(), len(names), "layers, K =", K, flush=True)
layers = []
for n in names:
    O, I = meta[n]["shape"]; g0, g1 = meta[n]["grid"]
    bm = torch.from_numpy(np.unpackbits(z[n.replace(".", "__")])[: g0 * g1].reshape(g0, g1).astype(bool)).cuda()
    W = (torch.randn(O, I, dtype=torch.bfloat16, device="cuda") * 0.02)
    Wm = (W.reshape(O, I // K, K) * bm[:, :, None]).reshape(O, I)
    vals, cols, ptr = pack(Wm, bm, 1, K)
    layers.append((O, I, W, Wm, vals, cols, ptr, row_order(ptr)))
bench = lambda f: triton.testing.do_bench(f, warmup=10, rep=40)
best = {}
for B in batches:
    xs = [torch.randn(B, L[1], dtype=torch.bfloat16, device="cuda") for L in layers]
    td = sum(bench(lambda: torch.nn.functional.linear(x, L[2])) for x, L in zip(xs, layers))
    res = {}
    for R in [1, 2, 4, 8]:
        for bn in [1, 2, 4, 8, 16]:
            if B * R * bn * K > 8192:
                continue
            for nw in [1, 2, 4]:
                for use_order in [False, True]:
                    t = 0.0
                    ok = True
                    for x, (O, I, W, Wm, vals, cols, ptr, order) in zip(xs, layers):
                        f = lambda: rows_gemv(vals, cols, ptr, x, O, K, R, bn, nw, order=order if use_order else None)
                        if not res:  # correctness once
                            ref = torch.nn.functional.linear(x.float(), Wm.float())
                            ok = (f().float() - ref).abs().max().item() <= 2e-2 * ref.abs().max().item()
                        t += bench(f)
                    if ok:
                        res[(R, bn, nw, use_order)] = t
    (R, bn, nw, uo), t = min(res.items(), key=lambda kv: kv[1])
    tno = min(v for k, v in res.items() if not k[3])
    best[B] = [R, bn, nw, uo]
    print(f"B={B:2d}: dense {td:.2f} ms | best unsorted {tno:.2f} ({td/tno:.2f}x) | best overall {t:.2f} ({td/t:.2f}x) "
          f"R={R} BLOCK_N={bn} warps={nw} sorted={uo}", flush=True)
print("BEST_REAL", json.dumps(best))
