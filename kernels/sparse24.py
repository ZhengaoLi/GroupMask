"""2:4 semi-structured weights with a Triton decode GEMV, written the same way as the group-sparse kernel
(compressed storage, several output rows per program, per-GPU tuning) for an apples-to-apples comparison.

Format for W [O, I] with exactly 2 kept values in every 4 consecutive inputs:
  vals : bf16  [O, I/2]   kept values, in input order
  meta : uint8 [O, I/8]   2-bit in-group index per kept value; 4 bits per group of 4, two groups per byte
Memory = 0.5 * dense + 1 bit per weight = 56.25% of the dense bf16 weight (same as cuSPARSELt).
"""
import torch
import triton
import triton.language as tl


def pack24(W: torch.Tensor):
    """Magnitude 2:4 pruning of W (any values) and packing. Returns vals, meta, and the pruned dense W."""
    O, I = W.shape
    assert I % 8 == 0
    g = W.reshape(O, I // 4, 4)
    idx = g.abs().topk(2, dim=-1).indices.sort(dim=-1).values            # [O, I/4, 2], ascending
    vals = g.gather(-1, idx).reshape(O, I // 2).to(torch.bfloat16).contiguous()
    nib = (idx[..., 0] | (idx[..., 1] << 2)).to(torch.uint8)               # [O, I/4]
    meta = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()               # [O, I/8]
    Wp = torch.zeros_like(g).scatter_(-1, idx, g.gather(-1, idx)).reshape(O, I)
    return vals, meta, Wp


@triton.jit
def _gemv24_kernel(vals_ptr, meta_ptr, x_ptr, y_ptr, I, O,
                   BATCH: tl.constexpr, R: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    rr = pid * R + tl.arange(0, R)
    rm = rr < O
    rb = tl.arange(0, BATCH)
    jv = tl.arange(0, BLK // 2)                      # kept-value index inside the chunk
    g = jv // 2                                      # group (of 4 inputs) index inside the chunk
    s = jv % 2                                       # first / second kept value of the group
    acc = tl.zeros((BATCH, R), dtype=tl.float32)
    for c0 in range(0, I, BLK):
        v = tl.load(vals_ptr + rr[:, None] * (I // 2) + c0 // 2 + jv[None, :],
                    mask=rm[:, None], other=0.0).to(tl.float32)                       # [R, BLK/2]
        byte = tl.load(meta_ptr + rr[:, None] * (I // 8) + c0 // 8 + (g // 2)[None, :],
                       mask=rm[:, None], other=0).to(tl.int32)                        # [R, BLK/2]
        nib = (byte >> ((g % 2) * 4)[None, :]) & 15
        idx = (nib >> (s * 2)[None, :]) & 3
        col = c0 + g[None, :] * 4 + idx                                               # [R, BLK/2]
        if BATCH == 1:
            xv = tl.load(x_ptr + col, mask=rm[:, None], other=0.0).to(tl.float32)
            acc += tl.reshape(tl.sum(v * xv, axis=1), (1, R))
        else:
            xv = tl.load(x_ptr + rb[:, None, None] * I + col[None, :, :],
                         mask=rm[None, :, None], other=0.0).to(tl.float32)            # [B, R, BLK/2]
            acc += tl.sum(v[None] * xv, axis=2)
    tl.store(y_ptr + rb[:, None] * O + rr[None, :], acc.to(tl.bfloat16), mask=rm[None, :])


@triton.jit
def _gemv24_reg_kernel(vals_ptr, meta_ptr, x_ptr, y_ptr, I, O,
                       BATCH: tl.constexpr, R: tl.constexpr, BLK: tl.constexpr):
    """Same as _gemv24_kernel, but the input chunk x[:, c0:c0+BLK] is read once, contiguously, and the
    kept positions are selected in registers with tl.gather (no scattered global loads)."""
    pid = tl.program_id(0)
    rr = pid * R + tl.arange(0, R)
    rm = rr < O
    rb = tl.arange(0, BATCH)
    jv = tl.arange(0, BLK // 2)
    g = jv // 2
    s = jv % 2
    rc = tl.arange(0, BLK)
    acc = tl.zeros((BATCH, R), dtype=tl.float32)
    for c0 in range(0, I, BLK):
        v = tl.load(vals_ptr + rr[:, None] * (I // 2) + c0 // 2 + jv[None, :],
                    mask=rm[:, None], other=0.0).to(tl.float32)                        # [R, BLK/2]
        byte = tl.load(meta_ptr + rr[:, None] * (I // 8) + c0 // 8 + (g // 2)[None, :],
                       mask=rm[:, None], other=0).to(tl.int32)
        nib = (byte >> ((g % 2) * 4)[None, :]) & 15
        idx = (nib >> (s * 2)[None, :]) & 3
        lcol = g[None, :] * 4 + idx                                                     # [R, BLK/2] in-chunk
        for b in tl.static_range(BATCH):
            xc = tl.load(x_ptr + b * I + c0 + rc).to(tl.float32)                       # [BLK]
            xg = tl.gather(tl.broadcast_to(xc[None, :], (R, BLK)), lcol, axis=1)       # [R, BLK/2]
            part = tl.sum(v * xg, axis=1)                                               # [R]
            acc += tl.where(rb[:, None] == b, part[None, :], 0.0)
    tl.store(y_ptr + rb[:, None] * O + rr[None, :], acc.to(tl.bfloat16), mask=rm[None, :])


def gemv24_reg(vals, meta, x, O, cfg):
    B, I = x.shape
    Bp = max(1, triton.next_power_of_2(B))
    if Bp != B:
        x = torch.nn.functional.pad(x, (0, 0, 0, Bp - B))
    x = x.contiguous()
    R, blk, nw = cfg
    y = torch.empty(Bp, O, dtype=torch.bfloat16, device=x.device)
    _gemv24_reg_kernel[(triton.cdiv(O, R),)](vals, meta, x, y, I, O, BATCH=Bp, R=R, BLK=blk, num_warps=nw)
    return y[:B]


# {padded_batch: (R, BLK, warps)} from sweep24.py (global-gather kernel; the register-gather variant was slower).
_TUNED24 = {
    "A100": {1: (8, 256, 4), 2: (2, 256, 1), 4: (2, 256, 1), 8: (1, 256, 1), 16: (1, 256, 1)},
    "default": {1: (8, 256, 2), 2: (8, 256, 1), 4: (4, 256, 1), 8: (2, 256, 1), 16: (1, 256, 1)},  # RTX 5060 Ti
}
# Optional per-GPU table from a sweep24.py run: GSPARSE24_TUNED=file.json with {"gpu substring": {"B": [R, BLK, warps]}}
import json as _json, os as _os
if _os.environ.get("GSPARSE24_TUNED") and _os.path.exists(_os.environ["GSPARSE24_TUNED"]):
    for _gpu, _t in _json.load(open(_os.environ["GSPARSE24_TUNED"])).items():
        _TUNED24 = {_gpu: {int(b): tuple(c) for b, c in _t.items()}, **_TUNED24}


def gemv24(vals, meta, x, O, cfg=None):
    B, I = x.shape
    Bp = max(1, triton.next_power_of_2(B))
    if Bp != B:
        x = torch.nn.functional.pad(x, (0, 0, 0, Bp - B))
    x = x.contiguous()
    if cfg is None:
        name = torch.cuda.get_device_name(x.device)
        table = next((v for k, v in _TUNED24.items() if k in name), _TUNED24["default"])
        cfg = table.get(Bp, (2, 512, 4))
    R, blk, nw = cfg
    assert I % blk == 0
    y = torch.empty(Bp, O, dtype=torch.bfloat16, device=x.device)
    _gemv24_kernel[(triton.cdiv(O, R),)](vals, meta, x, y, I, O, BATCH=Bp, R=R, BLK=blk, num_warps=nw)
    return y[:B]


class Sparse24Linear(torch.nn.Module):
    """2:4 linear for decoding with the Triton GEMV; prefill unpacks to a shared dense buffer + cuBLAS."""

    def __init__(self, W, prefill_threshold=16):
        super().__init__()
        self.out_features, self.in_features = W.shape
        vals, meta, _ = pack24(W)
        self.register_buffer("vals", vals)
        self.register_buffer("meta", meta)
        self.prefill_threshold = prefill_threshold

    def dense_weight(self):
        O, I = self.out_features, self.in_features
        m = self.meta.to(torch.int32)
        nib = torch.stack([m & 15, m >> 4], -1).reshape(O, I // 4)
        idx = torch.stack([nib & 3, (nib >> 2) & 3], -1).long()                     # [O, I/4, 2]
        W = torch.zeros(O, I // 4, 4, dtype=self.vals.dtype, device=self.vals.device)
        W.scatter_(-1, idx, self.vals.reshape(O, I // 4, 2))
        return W.reshape(O, I)

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.bfloat16)
        if x2.shape[0] > self.prefill_threshold:
            y = torch.nn.functional.linear(x2, self.dense_weight())
        else:
            y = gemv24(self.vals, self.meta, x2, self.out_features)
        return y.reshape(*shp[:-1], self.out_features)
