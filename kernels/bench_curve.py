"""Latency-vs-batch curves on ONE GPU for dense, 2:4, GroupMask 1x64 and unstructured sparse weights, with a
simple bandwidth / compute bottleneck attribution.

Workload: ALL 224 target linears (32 blocks x 7) of LLaMA-2-7B with the REAL learned 1x64 GroupMask masks, so the
total density is the model's ~50%. Layers are processed ONE AT A TIME (build all formats -> time every batch/format ->
free) and the per-layer times/bytes/FLOPs are summed per (format, batch). Weights are random (values do not affect
speed). Unstructured (CSR) uses the same per-layer density as GroupMask; 2:4 is 50% per layer.
"batch" = number of tokens multiplied by the weight in one call (decode batch size, or prompt tokens in prefill).

Formats / kernels (all BF16 activations):
  dense            cuBLAS F.linear
  24_cusparselt    2:4, torch to_sparse_semi_structured (cuSPARSELt, sparse tensor cores); token dim padded to >=16
  24_triton        2:4, our Triton GEMV (same design as the GroupMask GEMV)              [B <= 64]
  gm_gemv          GroupMask 1x64, our Triton GEMV on the compressed format             [B <= 64]
  gm_expand_gemm   GroupMask 1x64, unpack to a dense buffer + cuBLAS (our prefill path)
  unstr_csr        unstructured, same per-layer density as GroupMask, cuSPARSE CSR SpMM (int32 indices)
Derived "deploy" curves (24_best, gm_best) = per-batch minimum over that format's kernels.

Attribution, per (format, batch), summed over the block:
  bytes  = weight-format bytes actually read (+ unpack write/read for gm_expand_gemm) + activations in/out
  flops  = executed FLOPs (dense-equivalent for dense, cuSPARSELt and expand+GEMM; 2*B*nnz for GEMV/CSR kernels)
  t_mem  = bytes / measured copy bandwidth;   t_comp = flops / peak of the unit the kernel uses
           (tensor-core paths: measured dense BF16 GEMM peak; cuSPARSELt: 2x that (sparse TC);
            CUDA-core kernels (Triton GEMV, cuSPARSE CSR): FP32 FMA peak from the spec table)
  bound  = "memory" if t_mem > t_comp else "compute";  roofline_eff = max(t_mem, t_comp) / measured time
"""
import argparse, json, time, gc
import numpy as np
import torch
import triton
from gsparse import pack, sparse_gemv, row_order, bsr_to_dense
from sparse24 import pack24, gemv24

ap = argparse.ArgumentParser()
ap.add_argument("--masks", default="masks/llama2-7b-wiki-1x64.npz")
ap.add_argument("--layers", default="all", help="all or comma list of block ids (smoke test)")
ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
ap.add_argument("--gemv_max", type=int, default=64)
ap.add_argument("--csr_max", type=int, default=1024, help="CSR SpMM is skipped above this batch (too slow)")
ap.add_argument("--rep", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
gpu = torch.cuda.get_device_name()
BF = torch.bfloat16
bench = lambda fn: triton.testing.do_bench(fn, warmup=25, rep=a.rep)  # ms

# ---- measured machine peaks -------------------------------------------------------------------------------
n = 1 << 29                                            # 1 GiB of bf16
src = torch.empty(n, dtype=BF, device=dev); dst = torch.empty_like(src)
bw = 2 * src.numel() * 2 / (bench(lambda: dst.copy_(src)) * 1e-3)          # bytes/s (read + write)
del src, dst
g = torch.randn(8192, 8192, dtype=BF, device=dev)
tc = 2 * 8192 ** 3 / (bench(lambda: g @ g) * 1e-3)                         # FLOP/s dense BF16 tensor core
del g
FP32_SPEC = {"A100": 19.5e12, "H100": 67e12, "B200": 80e12, "5060": 23.7e12, "RTX PRO 6000": 126e12}
fma = next((v for k, v in FP32_SPEC.items() if k in gpu), 20e12)
torch.cuda.empty_cache()
peaks = {"device": gpu, "copy_bw_GBs": bw / 1e9, "dense_bf16_TFLOPs": tc / 1e12, "fp32_fma_TFLOPs_spec": fma / 1e12}
print(json.dumps(peaks), flush=True)

# ---- per-layer build + benchmark ----------------------------------------------------------------------------
from torch.sparse import to_sparse_semi_structured
meta = json.load(open(a.masks.replace(".npz", ".json")))["layers"]
npz = np.load(a.masks)
if a.layers == "all":
    names = list(meta)
else:
    ids = set(a.layers.split(","))
    names = [k for k in meta if k.split("layers.")[1].split(".")[0] in ids]
print(len(names), "linears", flush=True)
batches = [int(b) for b in a.batches.split(",")]
FMTS = ["dense", "24_cusparselt", "24_triton", "gm_gemv", "gm_expand_gemm", "unstr_csr"]
acc = {}      # (fmt,B) -> [ms, bytes, flops, tcomp_s]
failed = set()
tot = {"gm_nnz": 0, "nnz24": 0, "csr_nnz": 0, "dense": 0, "gm_bytes": 0, "t24_bytes": 0, "cs24_bytes": 0, "csr_bytes": 0}
csr_dtype = BF


def build(name):
    m = meta[name]; O, I = m["shape"]; A, K = m["group"]; g0, g1 = m["grid"]
    bm = torch.from_numpy(np.unpackbits(npz[name.replace(".", "__")])[: g0 * g1].reshape(g0, g1).astype(bool)).to(dev)
    W = torch.randn(O, I, dtype=BF, device=dev) * 0.02
    L = {"name": name, "O": O, "I": I, "W": W}
    Wm = (W.reshape(O // A, A, I // K, K) * bm[:, None, :, None]).reshape(O, I)
    vals, cols, ptr = pack(Wm, bm, A, K)
    L["gm"] = (vals, cols, ptr, row_order(ptr), bm.to(torch.uint8).contiguous(), A, K)
    L["gm_nnz"] = vals.numel()
    L["gm_bytes"] = sum(t.numel() * t.element_size() for t in (vals, cols, ptr))
    L["gm_unpack_bytes"] = vals.numel() * 2 + ptr.numel() * 4 + bm.numel()
    v24, m24, Wp = pack24(W)
    L["t24"] = (v24, m24); L["t24_bytes"] = v24.numel() * 2 + m24.numel()
    S = to_sparse_semi_structured(Wp)
    L["cs24"] = S
    L["cs24_bytes"] = S.packed.numel() * S.packed.element_size() + (
        S.meta.numel() * S.meta.element_size() if getattr(S, "meta", None) is not None else 0)
    L["nnz24"] = O * I // 2
    keep = bm.float().mean().item(); L["keep"] = keep
    Mu = torch.rand(O, I, device=dev) < keep
    Wu = (W * Mu).to_sparse_csr()
    Wu = torch.sparse_csr_tensor(Wu.crow_indices().to(torch.int32), Wu.col_indices().to(torch.int32),
                                 Wu.values().to(csr_dtype), size=(O, I))
    L["csr"] = Wu; L["csr_nnz"] = Wu.values().numel(); L["csr_bytes"] = L["csr_nnz"] * 6 + (O + 1) * 4
    del Wm, Wp, Mu
    return L


def run(L, x, fmt, B):
    O, I = L["O"], L["I"]
    act = 2 * B * (I + O)
    if fmt == "dense":
        W = L["W"]; t = bench(lambda: torch.nn.functional.linear(x, W))
        b = O * I * 2 + act; f = 2 * B * O * I; tc_ = f / tc
    elif fmt == "24_cusparselt":
        Bp = max(16, -(-B // 8) * 8)
        xp = torch.nn.functional.pad(x, (0, 0, 0, Bp - B)) if Bp != B else x
        S = L["cs24"]; t = bench(lambda: torch.nn.functional.linear(xp, S))
        b = L["cs24_bytes"] + 2 * Bp * (I + O); f = 2 * Bp * L["nnz24"]; tc_ = f / (2 * tc)
    elif fmt == "24_triton":
        if B > a.gemv_max: return None
        v, m = L["t24"]; t = bench(lambda: gemv24(v, m, x, O))
        b = L["t24_bytes"] + act; f = 2 * B * L["nnz24"]; tc_ = f / fma
    elif fmt == "gm_gemv":
        if B > a.gemv_max: return None
        vals, cols, ptr, order, bm, A, K = L["gm"]
        t = bench(lambda: sparse_gemv(vals, cols, ptr, x, O, A, K, order=order))
        b = L["gm_bytes"] + act; f = 2 * B * L["gm_nnz"]; tc_ = f / fma
    elif fmt == "gm_expand_gemm":
        vals, cols, ptr, order, bm, A, K = L["gm"]
        def fn():
            Wd = bsr_to_dense(vals, cols, ptr, O, I, A, K, bitmap=bm)
            return torch.nn.functional.linear(x, Wd)
        t = bench(fn)
        b = L["gm_unpack_bytes"] + 2 * O * I * 2 + act; f = 2 * B * O * I; tc_ = f / tc
    elif fmt == "unstr_csr":
        if B > a.csr_max: return None
        Wu = L["csr"]; xt = x.t().contiguous().to(csr_dtype)
        t = bench(lambda: torch.sparse.mm(Wu, xt))
        b = L["csr_bytes"] + act; f = 2 * B * L["csr_nnz"]; tc_ = f / fma
    return t, b, f, tc_


# CSR dtype probe on a small random matrix
try:
    t0 = torch.rand(256, 256, device=dev).to(BF).to_sparse_csr()
    torch.sparse.mm(t0, torch.randn(256, 4, dtype=BF, device=dev))
except Exception as e:
    print("csr bf16 unsupported, using fp16:", repr(e)[:120], flush=True)
    csr_dtype = torch.float16
print("csr dtype", csr_dtype, flush=True)

T0 = time.time()
for li, name in enumerate(names):
    L = build(name)
    for k, kk in [("gm_nnz", "gm_nnz"), ("nnz24", "nnz24"), ("csr_nnz", "csr_nnz"), ("gm_bytes", "gm_bytes"),
                  ("t24_bytes", "t24_bytes"), ("cs24_bytes", "cs24_bytes"), ("csr_bytes", "csr_bytes")]:
        tot[k] += L[kk]
    tot["dense"] += L["O"] * L["I"]
    for B in batches:
        x = torch.randn(B, L["I"], dtype=BF, device=dev)
        for fmt in FMTS:
            if (fmt, B) in failed: continue
            try:
                r = run(L, x, fmt, B)
            except Exception as e:
                print(f"{fmt} B={B} failed on {name}: {repr(e)[:150]}", flush=True); failed.add((fmt, B)); continue
            if r is None: continue
            s = acc.setdefault((fmt, B), [0.0, 0, 0, 0.0])
            for j in range(4): s[j] += r[j]
        del x
    del L; gc.collect(); torch.cuda.empty_cache()
    print(f"[{li + 1}/{len(names)}] {name} keep-so-far {tot['gm_nnz'] / tot['dense']:.3f} elapsed {time.time() - T0:.0f}s",
          flush=True)

rows = []
for (fmt, B), (ms, byts, fl, tcomp) in sorted(acc.items(), key=lambda kv: (batches.index(kv[0][1]), FMTS.index(kv[0][0]))):
    if (fmt, B) in failed: continue
    t_s = ms * 1e-3; t_mem = byts / bw
    rows.append({"fmt": fmt, "B": B, "ms": round(ms, 4), "bytes_MB": round(byts / 1e6, 2), "GFLOP": round(fl / 1e9, 3),
                 "achieved_GBs": round(byts / t_s / 1e9, 1), "achieved_TFLOPs": round(fl / t_s / 1e12, 2),
                 "t_mem_ms": round(t_mem * 1e3, 4), "t_comp_ms": round(tcomp * 1e3, 4),
                 "bound": "memory" if t_mem > tcomp else "compute", "roofline_eff": round(max(t_mem, tcomp) / t_s, 3),
                 "device": gpu})
summary = {"peaks": peaks, "csr_dtype": str(csr_dtype), "n_linears": len(names), "rep_ms": a.rep, "csr_max": a.csr_max,
           "failed": sorted(map(str, failed)), "wall_s": time.time() - T0,
           "nnz": {"gm": tot["gm_nnz"], "24": tot["nnz24"], "csr": tot["csr_nnz"], "dense": tot["dense"]},
           "density": {"gm": tot["gm_nnz"] / tot["dense"], "csr": tot["csr_nnz"] / tot["dense"], "24": 0.5},
           "weight_MB": {"dense": tot["dense"] * 2 / 1e6, "gm": tot["gm_bytes"] / 1e6, "24_triton": tot["t24_bytes"] / 1e6,
                         "24_cusparselt": tot["cs24_bytes"] / 1e6, "unstr_csr": tot["csr_bytes"] / 1e6}}
json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
print(json.dumps(summary), flush=True)
