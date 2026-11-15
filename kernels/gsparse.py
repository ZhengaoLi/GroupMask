"""Group-sparse (block-sparse) weight format and a Triton GEMV/small-batch kernel for decoding.

Format (BSR over A x K blocks; 1x64 row segments are A=1, K=64; 8x8 blocks are A=8, K=8):
  vals : bf16 [nnz, A, K]   kept blocks, band-major (band = A consecutive output rows)
  cols : int32 [nnz]        block-column index (input offset = col * K)
  ptr  : int32 [O/A + 1]    band pointer into vals/cols
Memory = kept * O * I * 2 bytes + 4 bytes per kept block (+ ptr). For 1x64 at 50% kept this is
~51.6% of the dense bf16 weight.

y[b, o] = sum over kept blocks in o's band of  vals[n, o % A, :] . x[b, cols[n]*K : cols[n]*K + K]
"""
import json
import os
import torch
import triton
import triton.language as tl


def pack(W: torch.Tensor, block_mask: torch.Tensor, A: int, K: int):
    """W: [O, I] dense (pruned groups may hold any value); block_mask: bool [O/A, I/K]."""
    O, I = W.shape
    assert O % A == 0 and I % K == 0 and block_mask.shape == (O // A, I // K)
    blocks = W.reshape(O // A, A, I // K, K).permute(0, 2, 1, 3)          # [bands, cols, A, K]
    band_idx, col_idx = block_mask.nonzero(as_tuple=True)                   # row-major => band-major
    vals = blocks[band_idx, col_idx].contiguous().to(torch.bfloat16)       # [nnz, A, K]
    counts = block_mask.sum(1).to(torch.int32)
    ptr = torch.zeros(O // A + 1, dtype=torch.int32, device=W.device)
    ptr[1:] = torch.cumsum(counts, 0)
    return vals, col_idx.to(torch.int32).contiguous(), ptr


@triton.jit
def _bsr_gemv_kernel(vals_ptr, cols_ptr, band_ptr, x_ptr, y_ptr,
                     I, O,
                     BATCH: tl.constexpr, A: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    band = tl.program_id(0)
    start = tl.load(band_ptr + band)
    end = tl.load(band_ptr + band + 1)
    ra = tl.arange(0, A)
    rk = tl.arange(0, K)
    rb = tl.arange(0, BATCH)
    acc = tl.zeros((BATCH, A), dtype=tl.float32)
    for s0 in range(start, end, BLOCK_N):
        rn = s0 + tl.arange(0, BLOCK_N)
        mn = rn < end
        c = tl.load(cols_ptr + rn, mask=mn, other=0)                                         # [N]
        w = tl.load(vals_ptr + rn[:, None, None] * (A * K) + ra[None, :, None] * K + rk[None, None, :],
                    mask=mn[:, None, None], other=0.0).to(tl.float32)                         # [N, A, K]
        xv = tl.load(x_ptr + rb[:, None, None] * I + c[None, :, None] * K + rk[None, None, :],
                     mask=mn[None, :, None], other=0.0).to(tl.float32)                        # [B, N, K]
        prod = w[None, :, :, :] * xv[:, :, None, :]                                           # [B, N, A, K]
        acc += tl.sum(tl.sum(prod, axis=3), axis=1)
    tl.store(y_ptr + rb[:, None] * O + band * A + ra[None, :], acc.to(tl.bfloat16))


# Tuned on RTX 5060 Ti for 1x64 groups (LLaMA-2-7B shapes): one warp per band and a
# [B, N, A, K] register tile of ~1K elements is bandwidth-optimal for decoding.
_TUNED = {1: (16, 1), 2: (8, 1), 4: (4, 1), 8: (4, 1), 16: (1, 1)}


def bsr_gemv(vals, cols, ptr, x, O, A, K, block_n=None, num_warps=None):
    """x: [B, I] bf16 with small B. Returns y: [B, O] bf16."""
    B, I = x.shape
    Bp = max(1, triton.next_power_of_2(B))
    if Bp != B:
        x = torch.nn.functional.pad(x, (0, 0, 0, Bp - B))
    x = x.contiguous()
    y = torch.empty(Bp, O, dtype=torch.bfloat16, device=x.device)
    bn, nw = _TUNED.get(Bp, (1, 1))
    if A * K != 64:  # other block sizes: keep the register tile near 1K elements
        bn = max(1, triton.next_power_of_2(max(1, 1024 // (Bp * A * K))))
    block_n = block_n or bn
    num_warps = num_warps or nw
    _bsr_gemv_kernel[(O // A,)](vals, cols, ptr, x, y, I, O, BATCH=Bp, A=A, K=K, BLOCK_N=block_n, num_warps=num_warps)
    return y[:B]


@triton.jit
def _rows_gemv_kernel(vals_ptr, cols_ptr, row_ptr, order_ptr, x_ptr, y_ptr, I, O,
                      BATCH: tl.constexpr, R: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr,
                      USE_ORDER: tl.constexpr):
    """1 x K row-segment groups, R output rows per program processed in lockstep
    (rows of different lengths are masked), so each program keeps more loads in flight."""
    pid = tl.program_id(0)
    rs = pid * R + tl.arange(0, R)                       # [R] slot in (optionally length-sorted) row order
    rm = rs < O
    if USE_ORDER:
        rr = tl.load(order_ptr + rs, mask=rm, other=0)   # rows grouped by length -> balanced lockstep
    else:
        rr = rs
    start = tl.load(row_ptr + rr, mask=rm, other=0)
    end = tl.load(row_ptr + rr + 1, mask=rm, other=0)
    nmax = tl.max(end - start, axis=0)
    rk = tl.arange(0, K)
    rb = tl.arange(0, BATCH)
    acc = tl.zeros((BATCH, R), dtype=tl.float32)
    for s0 in range(0, nmax, BLOCK_N):
        rn = start[:, None] + s0 + tl.arange(0, BLOCK_N)[None, :]           # [R, N]
        mn = rn < end[:, None]
        c = tl.load(cols_ptr + rn, mask=mn, other=0)
        w = tl.load(vals_ptr + rn[:, :, None] * K + rk[None, None, :], mask=mn[:, :, None], other=0.0).to(tl.float32)  # [R,N,K]
        if BATCH == 1:
            xv = tl.load(x_ptr + c[:, :, None] * K + rk[None, None, :], mask=mn[:, :, None], other=0.0).to(tl.float32)
            acc += tl.reshape(tl.sum(tl.sum(w * xv, axis=2), axis=1), (1, R))
        else:
            xv = tl.load(x_ptr + rb[:, None, None, None] * I + c[None, :, :, None] * K + rk[None, None, None, :],
                         mask=mn[None, :, :, None], other=0.0).to(tl.float32)                  # [B,R,N,K]
            acc += tl.sum(tl.sum(w[None] * xv, axis=3), axis=2)
    tl.store(y_ptr + rb[:, None] * O + rr[None, :], acc.to(tl.bfloat16), mask=rm[None, :])


def row_order(ptr):
    """Rows sorted by number of kept groups (descending), so rows sharing a program have similar lengths."""
    return torch.argsort(ptr[1:] - ptr[:-1], descending=True, stable=True).to(torch.int32).contiguous()


def rows_gemv(vals, cols, ptr, x, O, K=64, R=4, block_n=4, num_warps=4, order=None):
    B, I = x.shape
    Bp = max(1, triton.next_power_of_2(B))
    if Bp != B:
        x = torch.nn.functional.pad(x, (0, 0, 0, Bp - B))
    x = x.contiguous()
    y = torch.empty(Bp, O, dtype=torch.bfloat16, device=x.device)
    _rows_gemv_kernel[(triton.cdiv(O, R),)](vals, cols, ptr, order if order is not None else ptr, x, y, I, O,
                                           BATCH=Bp, R=R, K=K, BLOCK_N=block_n, num_warps=num_warps,
                                           USE_ORDER=order is not None)
    return y[:B]


@triton.jit
def _bands_gemv_kernel(vals_ptr, cols_ptr, band_ptr, order_ptr, x_ptr, y_ptr, I, O, NB,
                       BATCH: tl.constexpr, R: tl.constexpr, A: tl.constexpr, K: tl.constexpr,
                       BLOCK_N: tl.constexpr, USE_ORDER: tl.constexpr):
    """A x K blocks (e.g. 8x8): R bands of A rows per program, processed in lockstep (optionally in
    length-sorted band order). Each weight block is loaded once and reused for every batch row; each
    x segment of K inputs is reused by the A rows of its block."""
    pid = tl.program_id(0)
    rs = pid * R + tl.arange(0, R)
    rm = rs < NB
    if USE_ORDER:
        rbnd = tl.load(order_ptr + rs, mask=rm, other=0)
    else:
        rbnd = rs
    start = tl.load(band_ptr + rbnd, mask=rm, other=0)
    end = tl.load(band_ptr + rbnd + 1, mask=rm, other=0)
    nmax = tl.max(end - start, axis=0)
    ra = tl.arange(0, A)
    rk = tl.arange(0, K)
    rb = tl.arange(0, BATCH)
    acc = tl.zeros((BATCH, R, A), dtype=tl.float32)
    for s0 in range(0, nmax, BLOCK_N):
        rn = start[:, None] + s0 + tl.arange(0, BLOCK_N)[None, :]                       # [R, N]
        mn = rn < end[:, None]
        c = tl.load(cols_ptr + rn, mask=mn, other=0)
        w = tl.load(vals_ptr + rn[:, :, None, None] * (A * K) + ra[None, None, :, None] * K + rk[None, None, None, :],
                    mask=mn[:, :, None, None], other=0.0).to(tl.float32)                 # [R, N, A, K]
        for b in tl.static_range(BATCH):
            xv = tl.load(x_ptr + b * I + c[:, :, None] * K + rk[None, None, :],
                         mask=mn[:, :, None], other=0.0).to(tl.float32)                  # [R, N, K]
            part = tl.sum(tl.sum(w * xv[:, :, None, :], axis=3), axis=1)                 # [R, A]
            acc += tl.where(rb[:, None, None] == b, part[None, :, :], 0.0)
    rows = rbnd[:, None] * A + ra[None, :]                                                # [R, A]
    tl.store(y_ptr + rb[:, None, None] * O + rows[None, :, :], acc.to(tl.bfloat16), mask=rm[None, :, None])


def bands_gemv(vals, cols, ptr, x, O, A, K, R=2, block_n=4, num_warps=1, order=None):
    B, I = x.shape
    Bp = max(1, triton.next_power_of_2(B))
    if Bp != B:
        x = torch.nn.functional.pad(x, (0, 0, 0, Bp - B))
    x = x.contiguous()
    NB = O // A
    y = torch.empty(Bp, O, dtype=torch.bfloat16, device=x.device)
    _bands_gemv_kernel[(triton.cdiv(NB, R),)](vals, cols, ptr, order if order is not None else ptr, x, y, I, O, NB,
                                             BATCH=Bp, R=R, A=A, K=K, BLOCK_N=block_n,
                                             USE_ORDER=order is not None, num_warps=num_warps)
    return y[:B]


# (R rows per program, BLOCK_N, num_warps, length-sorted rows) per padded batch,
# tuned with sweep_real.py on the real learned LLaMA-2-7B 1x64 masks.
_ROWS_TUNED = {
    "A100": {1: (1, 16, 1, False), 2: (1, 16, 1, False), 4: (1, 4, 1, True), 8: (1, 2, 1, True), 16: (2, 4, 2, True)},
    "default": {1: (2, 16, 1, False), 2: (8, 8, 1, True), 4: (1, 8, 1, False), 8: (1, 4, 1, True),
                16: (1, 2, 1, True)},  # RTX 5060 Ti
}
# 1x256 row segments (the paper's main configuration), tuned with sweep_real.py on the real 1x256 masks.
_ROWS_TUNED_256 = {
    "A100": {1: (1, 16, 4, False), 2: (1, 8, 4, False), 4: (1, 2, 1, True), 8: (1, 1, 2, True), 16: (2, 1, 2, True)},
    "default": {1: (1, 8, 4, False), 2: (1, 8, 4, False), 4: (1, 4, 2, False), 8: (1, 1, 1, True),
                16: (1, 2, 1, True)},  # RTX 5060 Ti
}
# Per group width K (1 x K row segments); widths without their own table fall back to the 1x64 one.
_ROWS_TUNED_BY_K = {64: _ROWS_TUNED, 256: _ROWS_TUNED_256}
# Optional extra tables from sweep_real.py runs: GSPARSE_TUNED=file.json with {"K": {"gpu substring": {"B": [R, BLOCK_N, warps, sorted]}}}
if os.environ.get("GSPARSE_TUNED") and os.path.exists(os.environ["GSPARSE_TUNED"]):
    for _k, _t in json.load(open(os.environ["GSPARSE_TUNED"])).items():
        _ROWS_TUNED_BY_K.setdefault(int(_k), {}).update(
            {g: {int(b): tuple(c) for b, c in tb.items()} for g, tb in _t.items()})
        _ROWS_TUNED_BY_K[int(_k)].setdefault("default", next(iter(_ROWS_TUNED_BY_K[int(_k)].values())))


# 8x8 blocks: batch 1 -> single-band kernel (BLOCK_N, warps); batch >= 2 -> multi-band kernel (R, BLOCK_N, warps, sorted).
# Tuned with sweep_8x8.py / sweep_8x8b.py on the real LLaMA-2-7B 8x8 masks (RTX 5060 Ti).
_BANDS_TUNED_8x8 = {2: (1, 16, 1, True), 4: (1, 16, 1, True), 8: (1, 16, 1, False), 16: (2, 16, 1, True)}
# A100-PCIE-40GB sweep (UConn job 33159559).
_BANDS_TUNED_8x8_A100 = {2: (1, 16, 1, True), 4: (1, 16, 1, False), 8: (1, 16, 1, True), 16: (2, 16, 1, True)}
_SINGLE_8x8 = {"A100": (32, 4), "default": (32, 1)}


def sparse_gemv(vals, cols, ptr, x, O, A, K, order=None, cfg=None):
    """Dispatch: 1 x K row segments -> multi-row kernel with per-GPU tuning; A x K blocks -> BSR kernels."""
    if A != 1:
        Bp = max(1, triton.next_power_of_2(x.shape[0]))
        if A == 8 and K == 8:
            a100 = "A100" in torch.cuda.get_device_name(x.device)
            if Bp == 1:
                bn, nw = _SINGLE_8x8["A100" if a100 else "default"]
                return bsr_gemv(vals, cols, ptr, x, O, A, K, block_n=bn, num_warps=nw)
            tab = _BANDS_TUNED_8x8_A100 if a100 else _BANDS_TUNED_8x8
            R, bn, nw, srt = tab.get(Bp, tab[16])
            return bands_gemv(vals, cols, ptr, x, O, A, K, R, bn, nw, order if srt else None)
        return bsr_gemv(vals, cols, ptr, x, O, A, K)
    if cfg is None:
        name = torch.cuda.get_device_name(x.device)
        tabs = _ROWS_TUNED_BY_K.get(K, _ROWS_TUNED)
        table = next((v for k, v in tabs.items() if k in name), tabs["default"])
        Bp = max(1, triton.next_power_of_2(x.shape[0]))
        cfg = table.get(Bp, table[16])
    R, bn, nw, use_sorted = cfg
    return rows_gemv(vals, cols, ptr, x, O, K, R, bn, nw, order=order if use_sorted else None)


@triton.jit
def _bsr_to_dense_kernel(vals_ptr, cols_ptr, band_ptr, w_ptr, I,
                         A: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr):
    """Write the kept blocks of one band into a zero-filled dense [O, I] buffer."""
    band = tl.program_id(0)
    start = tl.load(band_ptr + band)
    end = tl.load(band_ptr + band + 1)
    ra = tl.arange(0, A)
    rk = tl.arange(0, K)
    for s0 in range(start, end, BLOCK_N):
        rn = s0 + tl.arange(0, BLOCK_N)
        mn = rn < end
        c = tl.load(cols_ptr + rn, mask=mn, other=0)
        v = tl.load(vals_ptr + rn[:, None, None] * (A * K) + ra[None, :, None] * K + rk[None, None, :],
                    mask=mn[:, None, None], other=0.0)
        dst = w_ptr + (band * A + ra[None, :, None]) * I + c[:, None, None] * K + rk[None, None, :]
        tl.store(dst, v, mask=mn[:, None, None])


@triton.jit
def _unpack_rows_kernel(vals_ptr, row_ptr, bitmap_ptr, w_ptr, I, NBI,
                        NB: tl.constexpr, K: tl.constexpr):
    """1 x K groups: write one full dense row in a single pass (kept groups from vals, pruned groups as 0).
    The per-row byte bitmap gives each kept group's rank via a prefix sum, so no separate memset is needed
    and every dense element is written exactly once."""
    r = tl.program_id(0)
    start = tl.load(row_ptr + r)
    cb = tl.arange(0, NB)
    cm = cb < NBI
    b = tl.load(bitmap_ptr + r * NBI + cb, mask=cm, other=0).to(tl.int32)        # [NB]
    rank = tl.cumsum(b, axis=0) - 1
    rk = tl.arange(0, K)
    keep = (b == 1) & cm
    v = tl.load(vals_ptr + (start + rank)[:, None] * K + rk[None, :], mask=keep[:, None], other=0.0)  # [NB, K]
    tl.store(w_ptr + r * I + cb[:, None] * K + rk[None, :], v, mask=cm[:, None])


_DENSE_BUF = {}


def _dense_buf(n, dev):
    buf = _DENSE_BUF.get(dev)
    if buf is None or buf.numel() < n:
        buf = torch.empty(n, dtype=torch.bfloat16, device=dev)
        _DENSE_BUF[dev] = buf
    return buf


def bsr_to_dense(vals, cols, ptr, O, I, A, K, bitmap=None):
    """Dense [O, I] view of the packed weight in a buffer shared by all layers (one layer at a time).
    With a row bitmap (1 x K groups) this is a single kernel that writes every element once."""
    W = _dense_buf(O * I, vals.device)[: O * I].view(O, I)
    if bitmap is not None and A == 1:
        nbi = I // K
        _unpack_rows_kernel[(O,)](vals, ptr, bitmap, W, I, nbi, NB=triton.next_power_of_2(nbi), K=K, num_warps=8)
        return W
    W.zero_()
    _bsr_to_dense_kernel[(O // A,)](vals, cols, ptr, W, I, A=A, K=K, BLOCK_N=16, num_warps=4)
    return W


class GroupSparseLinear(torch.nn.Module):
    """Drop-in nn.Linear replacement (no bias) for decoding; falls back to a dense matmul for
    large token counts (prefill) by materialising the dense weight once per call."""

    def __init__(self, W, block_mask, A, K, prefill_threshold=16):
        super().__init__()
        self.out_features, self.in_features = W.shape
        self.A, self.K, self.prefill_threshold = A, K, prefill_threshold
        vals, cols, ptr = pack(W, block_mask, A, K)
        self.register_buffer("vals", vals)
        self.register_buffer("cols", cols)
        self.register_buffer("ptr", ptr)
        self.register_buffer("order", row_order(ptr))   # rows (1xK) or bands (AxK) sorted by kept groups
        # 1 byte per group (0.78% of the dense bf16 weight): lets prefill unpack in one pass
        self.register_buffer("bitmap", block_mask.to(torch.uint8).contiguous() if A == 1 else None)

    def dense_weight(self):
        O, I, A, K = self.out_features, self.in_features, self.A, self.K
        bands = torch.repeat_interleave(torch.arange(O // A, device=self.vals.device),
                                        (self.ptr[1:] - self.ptr[:-1]).long())
        W = torch.zeros(O // A, I // K, A, K, dtype=self.vals.dtype, device=self.vals.device)
        W[bands, self.cols.long()] = self.vals
        return W.permute(0, 2, 1, 3).reshape(O, I)

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if x2.shape[0] > self.prefill_threshold:
            with torch.profiler.record_function("gm_unpack"):
                W = bsr_to_dense(self.vals, self.cols, self.ptr, self.out_features, self.in_features, self.A, self.K,
                                 bitmap=self.bitmap)
            with torch.profiler.record_function("gm_gemm"):
                y = torch.nn.functional.linear(x2.to(torch.bfloat16), W)
        else:
            y = sparse_gemv(self.vals, self.cols, self.ptr, x2.to(torch.bfloat16), self.out_features, self.A, self.K,
                            order=self.order)
        return y.reshape(*shp[:-1], self.out_features)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.vals, self.cols, self.ptr, self.order, self.bitmap)
                   if t is not None)
