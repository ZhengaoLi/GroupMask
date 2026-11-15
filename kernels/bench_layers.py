"""Correctness + per-layer decode benchmark: group-sparse Triton GEMV vs dense cuBLAS vs 2:4 (cuSPARSELt).

With --masks <npz> (from extract_masks.py) every LLaMA-2-7B target linear uses its real learned mask and
the reported totals are the sum over all 224 linears (one decoding step's linear-layer time).
Without --masks, random masks at --keep are used on the 4 distinct LLaMA-2-7B shapes.
Weights are random (values do not affect speed).
"""
import argparse, json, time
import numpy as np
import torch
import triton
from gsparse import pack, bsr_gemv, sparse_gemv, row_order

ap = argparse.ArgumentParser()
ap.add_argument("--masks", default=None)
ap.add_argument("--group", default="1x64")
ap.add_argument("--keep", type=float, default=0.5)
ap.add_argument("--batches", default="1,4,8,16")
ap.add_argument("--no24", action="store_true")
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
print(torch.cuda.get_device_name(), torch.__version__, "triton", triton.__version__, flush=True)

def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)  # ms

layers = []
if a.masks:
    meta = json.load(open(a.masks.replace(".npz", ".json")))["layers"]
    npz = np.load(a.masks)
    for name, m in meta.items():
        O, I = m["shape"]; A, K = m["group"]; g0, g1 = m["grid"]
        bits = np.unpackbits(npz[name.replace(".", "__")])[: g0 * g1].reshape(g0, g1).astype(bool)
        layers.append((name, O, I, A, K, torch.from_numpy(bits)))
else:
    A, K = map(int, a.group.split("x"))
    for name, O, I, n in [("qkvo", 4096, 4096, 128), ("gate_up", 11008, 4096, 64), ("down", 4096, 11008, 32)]:
        layers.append((name, O, I, A, K, torch.rand(O // A, I // K) < a.keep, n))

batches = [int(b) for b in a.batches.split(",")]
cache = {}
res = {b: {"dense": 0.0, "group": 0.0, "2:4": 0.0} for b in batches}
mem = {"dense": 0, "group": 0, "2:4": 0}
checked = False
for li, L in enumerate(layers):
    name, O, I, A, K, bm = L[:6]
    reps = L[6] if len(L) > 6 else 1
    W = torch.randn(O, I, dtype=torch.bfloat16, device=dev) * 0.02
    Wm = (W.reshape(O // A, A, I // K, K) * bm.to(dev)[:, None, :, None]).reshape(O, I)
    vals, cols, ptr = pack(Wm, bm.to(dev), A, K)
    order = row_order(ptr) if A == 1 else None
    mem["dense"] += W.numel() * 2 * reps
    mem["group"] += (vals.numel() * 2 + cols.numel() * 4 + ptr.numel() * 4) * reps
    W24 = None
    if not a.no24:
        try:
            from torch.sparse import to_sparse_semi_structured
            # magnitude 2:4 mask just for timing
            Wg = W.reshape(-1, 4)
            idx = Wg.abs().topk(2, dim=1).indices
            W24d = torch.zeros_like(Wg).scatter_(1, idx, Wg.gather(1, idx)).reshape(O, I)
            W24 = to_sparse_semi_structured(W24d)
            mem["2:4"] += (W24.packed.numel() * W24.packed.element_size() +
                           (W24.meta.numel() * W24.meta.element_size() if W24.meta is not None else 0)) * reps
        except Exception as e:
            if li == 0:
                print("2:4 unavailable:", repr(e)[:200], flush=True)
            a.no24 = True
    for b in batches:
        x = torch.randn(b, I, dtype=torch.bfloat16, device=dev)
        if not checked:
            ref = torch.nn.functional.linear(x.float(), Wm.float())
            out = sparse_gemv(vals, cols, ptr, x, O, A, K, order=order).float()
            err = (out - ref).abs().max().item() / ref.abs().max().item()
            print(f"correctness {name} B={b}: max rel err {err:.2e}", flush=True)
            assert err < 2e-2, "kernel mismatch"
        res[b]["dense"] += bench(lambda: torch.nn.functional.linear(x, W)) * reps
        res[b]["group"] += bench(lambda: sparse_gemv(vals, cols, ptr, x, O, A, K, order=order)) * reps
        if W24 is not None:
            xp = x if b >= 16 else torch.nn.functional.pad(x, (0, 0, 0, 16 - b))
            res[b]["2:4"] += bench(lambda: torch.nn.functional.linear(xp, W24)) * reps
    checked = True
    del W, Wm, vals, cols, ptr, W24
    if (li + 1) % 28 == 0:
        print(f"... {li+1}/{len(layers)} layers", flush=True)

print("\n== linear-layer time for one decoding step (ms), summed over all target linears ==")
for b in batches:
    r = res[b]
    s = f"B={b:2d}  dense {r['dense']:7.3f}  group {r['group']:7.3f} ({r['dense']/r['group']:.2f}x)"
    if r["2:4"]:
        s += f"  2:4 {r['2:4']:7.3f} ({r['dense']/r['2:4']:.2f}x)"
    print(s)
print("\n== linear weight memory (GB) ==")
for k, v in mem.items():
    if v:
        print(f"{k:6s} {v/1e9:.2f} GB ({v/mem['dense']:.3f} of dense)")
if a.out:
    json.dump({"device": torch.cuda.get_device_name(), "res": res, "mem": mem, "masks": a.masks}, open(a.out, "w"), indent=1)
