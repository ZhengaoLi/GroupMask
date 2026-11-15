"""Full-request latency benchmark (prefill + decode of all generated tokens): dense vs GroupMask.

One request = prefill the prompt, then greedily decode --gens tokens; we time the whole thing as one
wall-clock span (CUDA synchronised at both ends), REPS times, and report the median [min-max].
Prefill only computes the last position's logits (logits_to_keep=1), as in serving; decode steps are
CUDA-graph replays.
python bench_request.py --mode {dense,group} --batch 1 --prompt 128 --gens 128,512 --out r.jsonl

Weights are random bf16 (values do not change speed or memory); group-sparse layers use the real
learned per-layer masks from extract_masks.py, so per-layer density matches the pruned model.
The model is built on the meta device and every linear is materialised directly in its final format,
so the group-sparse model never holds a dense copy (lets 13B fit on a 16 GB GPU).
Decoding uses a static KV cache and a captured CUDA graph per step (no Python/launch overhead), which
is how latency-oriented LLM serving runs.

"""
import argparse, json, time
import numpy as np
import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from gsparse import GroupSparseLinear
from sparse24 import Sparse24Linear

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["dense", "group", "2:4", "2:4-triton", "2:4-best", "struct50", "disp"])
ap.add_argument("--disp", default=None, help="DISP-LLM pruned checkpoint dir (mode disp): per-layer shapes are read "
                "from its safetensors headers; weights are random like the other rows")
ap.add_argument("--size", default="7b", choices=["7b", "13b"])
ap.add_argument("--masks", default=None)
ap.add_argument("--prompt", type=int, default=128)
ap.add_argument("--gens", default="128,512")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--graph", type=int, default=1)
ap.add_argument("--graph_prefill", type=int, default=0,
                help="1: also capture prefill in a CUDA graph (removes per-call host overhead for every method)")
ap.add_argument("--out", default=None)
ap.add_argument("--real", default=None,
                help="HF checkpoint dir: use its REAL weights (all params). group mode derives the mask from the "
                     "zero groups of the real weights; 2:4 modes require the real weights to already be 2:4")
ap.add_argument("--group", default="1x64", help="AxK group shape for --mode group --real")
ap.add_argument("--tokenizer", default=None, help="tokenizer dir (default: --real)")
ap.add_argument("--text", default=None, help="text file for a real prompt (with --real); default: built-in passage")
a = ap.parse_args()
torch.manual_seed(0)
dev = torch.device("cuda")
BF = torch.bfloat16

H, F, L, NH = {"7b": (4096, 11008, 32, 32), "13b": (5120, 13824, 40, 40)}[a.size]
cfg = LlamaConfig(hidden_size=H, intermediate_size=F, num_hidden_layers=L, num_attention_heads=NH,
                  num_key_value_heads=NH, vocab_size=32000, max_position_embeddings=4096, rms_norm_eps=1e-5,
                  torch_dtype=BF)
if a.mode == "struct50":
    # Ideal structured pruning at 50% of linear-layer parameters: half of the attention heads (Q/K/V/O, head_dim
    # kept) and half of the MLP channels are removed; the result is a smaller dense model run with cuBLAS. This is
    # an upper bound on the speed of structured pruning with the same parameter budget (also halves the KV cache).
    cfg = LlamaConfig(hidden_size=H, intermediate_size=F // 2, num_hidden_layers=L, num_attention_heads=NH // 2,
                      num_key_value_heads=NH // 2, head_dim=H // NH, vocab_size=32000, max_position_embeddings=4096,
                      rms_norm_eps=1e-5, torch_dtype=BF)
cfg._attn_implementation = "sdpa"
DSH = None
if a.mode == "disp":
    # Real DISP-LLM architecture (Gao et al., 2024): every layer keeps its own subset of hidden dimensions.
    # Shapes come from the pruned checkpoint; inference follows its modeling_llama_pruned.py: index_select of the
    # normalized hidden state before attention and MLP, pruned linears, and index_add back into the residual.
    import glob, struct as _st
    DSH = {}
    for f in sorted(glob.glob(os.path.join(a.disp, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = _st.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        DSH.update({k: tuple(v["shape"]) for k, v in hdr.items() if k != "__metadata__"})
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Linear24(torch.nn.Module):
    """2:4 weight via cuSPARSELt; pads the token dimension up to the minimum supported (16)."""
    def __init__(self, W):
        super().__init__()
        from torch.sparse import to_sparse_semi_structured
        O, I = W.shape
        g = W.reshape(-1, 4)
        idx = g.abs().topk(2, dim=1).indices
        W24 = torch.zeros_like(g).scatter_(1, idx, g.gather(1, idx)).reshape(O, I)
        self.w = to_sparse_semi_structured(W24)
        self.out_features = O
    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        n = x2.shape[0]
        pad = (16 - n) if n < 16 else (-n) % 8
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        return torch.nn.functional.linear(x2, self.w)[:n].reshape(*shp[:-1], self.out_features)


class Best24(torch.nn.Module):
    """Strongest 2:4 pipeline we can build: cuSPARSELt sparse tensor cores for prefill (many tokens) and the
    hand-tuned Triton 2:4 GEMV for decode (<= 16 tokens). Holds both formats, so its memory is reported but is
    not representative of a deployed 2:4 model (use the 2:4 / 2:4-triton rows for memory)."""
    def __init__(self, W, threshold=16):
        super().__init__()
        self.sp = Linear24(W)
        self.tr = Sparse24Linear(W)
        self.threshold = threshold
        self.out_features = W.shape[0]
    def forward(self, x):
        n = x.numel() // x.shape[-1]
        return self.sp(x) if n > self.threshold else self.tr(x)


class DenseLinear(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        self.weight = torch.nn.Parameter(W, requires_grad=False)
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class RealWeights:
    """Reads single tensors from a (possibly sharded) safetensors checkpoint without loading the whole model."""
    def __init__(self, d):
        import os
        from safetensors import safe_open
        self.d, self.h = d, {}
        idx = os.path.join(d, "model.safetensors.index.json")
        self.map = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
        self.open = lambda f: self.h.setdefault(f, safe_open(os.path.join(d, f), framework="pt", device="cpu"))
    def get(self, k):
        f = self.map[k] if self.map else "model.safetensors"
        return self.open(f).get_tensor(k)


with torch.device("meta"):
    model = LlamaForCausalLM(cfg)
RW = RealWeights(a.real) if a.real else None
check = {"layers": 0, "mismatch_elems": 0, "non24_groups": 0, "kept_groups": 0, "total_groups": 0}
masks = meta = None
if a.mode == "group" and a.masks:
    meta = json.load(open(a.masks.replace(".npz", ".json")))["layers"]
    masks = np.load(a.masks)

for name, mod in list(model.named_modules()):
    for t in TARGETS:
        child = getattr(mod, t, None)
        if child is None or not hasattr(child, "weight"):
            continue
        O, I = child.weight.shape if DSH is None else DSH[f"{name}.{t}.weight"]
        if RW is not None:
            W = RW.get(f"{name}.{t}.weight").to(device=dev, dtype=BF)
            assert W.shape == (O, I), (name, t, W.shape)
        else:
            W = torch.randn(O, I, dtype=BF, device=dev) * 0.02
        if RW is not None and a.mode.startswith("2:4"):
            # real weights must already be input-direction 2:4 (at most 2 nonzeros in every 4 consecutive inputs)
            check["non24_groups"] += int(((W.reshape(-1, 4) != 0).sum(1) > 2).sum())
        if a.mode in ("dense", "struct50", "disp"):
            new = DenseLinear(W)
        elif a.mode == "2:4":
            new = Linear24(W)
        elif a.mode == "2:4-triton":
            new = Sparse24Linear(W)
            if RW is not None:
                check["mismatch_elems"] += int((new.dense_weight() != W).sum())
        elif a.mode == "2:4-best":
            new = Best24(W)
            if RW is not None:
                check["mismatch_elems"] += int((new.tr.dense_weight() != W).sum())
        elif RW is not None:  # group mode on a real pruned checkpoint: kept groups = groups with any nonzero weight
            A, K = (int(v) for v in a.group.split("x"))
            bm = (W.reshape(O // A, A, I // K, K) != 0).any(3).any(1)
            new = GroupSparseLinear(W, bm, A, K)
            check["mismatch_elems"] += int((new.dense_weight() != W).sum())
            check["kept_groups"] += int(bm.sum()); check["total_groups"] += bm.numel()
        elif meta is not None:
            m = meta[f"{name}.{t}.weight"]
            A, K = m["group"]; g0, g1 = m["grid"]
            bits = np.unpackbits(masks[f"{name}.{t}.weight".replace(".", "__")])[: g0 * g1].reshape(g0, g1).astype(bool)
            new = GroupSparseLinear(W, torch.from_numpy(bits).to(dev), A, K)
        else:  # no mask file: random 1x64 masks at 50% (smoke test only)
            new = GroupSparseLinear(W, torch.rand(O, I // 64, device=dev) < 0.5, 1, 64)
        del W
        setattr(mod, t, new)
        check["layers"] += 1
# remaining params (embeddings, norms, lm_head) and buffers still on meta -> real tensors
# (module.to_empty would also wipe the already-built sparse buffers, so do it per tensor)
for mod in model.modules():
    for k, p in list(mod._parameters.items()):
        if p is not None and p.is_meta:
            mod._parameters[k] = torch.nn.Parameter(torch.empty(p.shape, dtype=BF, device=dev), requires_grad=False)
    for k, b in list(mod._buffers.items()):
        if b is not None and b.is_meta:
            mod._buffers[k] = torch.zeros(b.shape, dtype=b.dtype, device=dev)
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.is_floating_point() and p.device.type == "cuda" and not n.endswith(tuple(f"{t}.weight" for t in TARGETS)):
            if RW is not None:
                p.copy_(RW.get(n).to(p.dtype))
            elif "norm" in n:
                p.fill_(1.0)
            else:
                p.normal_(0, 0.02)
if RW is not None:
    print("REALCHECK " + json.dumps({"real": a.real, "mode": a.mode, **check}), flush=True)
    assert check["mismatch_elems"] == 0 and check["non24_groups"] == 0, "format is not lossless for these weights"
if DSH is not None:
    import types
    def _disp_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                      use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        h = torch.index_select(self.input_layernorm(hidden_states), -1, self.sel_a)
        h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids,
                              past_key_value=past_key_value, use_cache=use_cache, cache_position=cache_position,
                              position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual.index_add(-1, self.copy_a, h)
        residual = hidden_states
        h = torch.index_select(self.post_attention_layernorm(hidden_states), -1, self.sel_m)
        return residual.index_add(-1, self.copy_m, self.mlp(h))
    nparam = 0
    for i, layer in enumerate(model.model.layers):
        pre = f"model.layers.{i}."
        d_a = DSH[pre + "self_attn.q_proj.weight"][1]; d_o = DSH[pre + "self_attn.o_proj.weight"][0]
        d_m = DSH[pre + "mlp.gate_proj.weight"][1]; d_d = DSH[pre + "mlp.down_proj.weight"][0]
        pick = lambda d: torch.randperm(H, device=dev)[:d].sort().values
        layer.sel_a, layer.copy_a, layer.sel_m, layer.copy_m = pick(d_a), pick(d_o), pick(d_m), pick(d_d)
        layer.forward = types.MethodType(_disp_forward, layer)
        nparam += sum(int(np.prod(DSH[pre + k])) for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight",
                      "mlp.up_proj.weight", "mlp.down_proj.weight"))
    print("DISP linear params %d (%.4f of dense)" % (nparam, nparam / (L * (4 * H * H + 3 * H * F))), flush=True)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
model.model.rotary_emb = LlamaRotaryEmbedding(config=cfg, device=dev)
model = model.to(BF).eval()
torch.cuda.empty_cache()
torch.cuda.synchronize()
weight_mem = torch.cuda.memory_allocated() / 1e9

# static KV cache
from transformers import StaticCache
GENS = [int(g) for g in a.gens.split(',')]
maxlen = a.prompt + max(GENS) + 16
cache = None
for kw in ({"config": cfg, "max_batch_size": a.batch, "max_cache_len": maxlen, "device": dev, "dtype": BF},
           {"config": cfg, "max_cache_len": maxlen},
           {"config": cfg, "max_batch_size": a.batch, "max_cache_len": maxlen}):
    try:
        cache = StaticCache(**kw); break
    except TypeError:
        continue
assert cache is not None

if RW is not None:  # real prompt: natural text, tiled to the prompt length
    from transformers import AutoTokenizer
    tokz = AutoTokenizer.from_pretrained(a.tokenizer or a.real)
    txt = open(a.text).read() if a.text else (
        "The history of the city begins with a small settlement on the north bank of the river, where farmers "
        "traded grain and wool with merchants travelling between the coast and the mountains. Over the next two "
        "centuries the market grew into a town with a stone bridge, a cathedral and a university, and by the "
        "industrial era it had become one of the largest manufacturing centres in the region. ")
    t0 = tokz(txt, return_tensors="pt").input_ids[0]
    t0 = t0.repeat(a.prompt // len(t0) + 1)[: a.prompt]
    ids = t0.to(dev).unsqueeze(0).repeat(a.batch, 1)
else:
    ids = torch.randint(0, 32000, (a.batch, a.prompt), device=dev)
tok = torch.zeros(a.batch, 1, dtype=torch.long, device=dev)
pos = torch.zeros(1, dtype=torch.long, device=dev)


@torch.no_grad()
def prefill():
    cache.reset()
    out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(a.prompt, device=dev),
                logits_to_keep=1)
    tok.copy_(out.logits[:, -1:].argmax(-1))
    pos.fill_(a.prompt)


@torch.no_grad()
def step():
    out = model(tok, past_key_values=cache, use_cache=True, cache_position=pos)
    return out.logits


prefill()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        step()
torch.cuda.current_stream().wait_stream(s)
prefill()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_logits = step()


if a.graph_prefill:
    # Capture the whole prefill (cache reset + forward) in its own graph, so host-side work such as
    # cuSPARSELt plan setup is not paid on every request. Same treatment for every method.
    cpos_static = torch.arange(a.prompt, device=dev)
    def _prefill_body():
        cache.reset()
        return model(ids, past_key_values=cache, use_cache=True, cache_position=cpos_static, logits_to_keep=1).logits
    s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2), torch.no_grad():
        for _ in range(3):
            _prefill_body()
    torch.cuda.current_stream().wait_stream(s2)
    pgraph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(pgraph):
        static_plogits = _prefill_body()

    @torch.no_grad()
    def prefill():
        pgraph.replay()
        tok.copy_(static_plogits[:, -1:].argmax(-1))
        pos.fill_(a.prompt)


def decode(n):
    for _ in range(n):
        graph.replay()
        tok.copy_(static_logits[:, -1:].argmax(-1))
        pos.add_(1)


def request(n):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    prefill(); torch.cuda.synchronize(); t1 = time.perf_counter()
    decode(n); torch.cuda.synchronize(); t2 = time.perf_counter()
    return (t2 - t0) * 1e3, (t1 - t0) * 1e3, (t2 - t1) * 1e3


for _ in range(2):
    request(32)
torch.cuda.reset_peak_memory_stats()
for g in GENS:
    runs = sorted((request(g) for _ in range(a.reps)), key=lambda r: r[0])
    tot = [r[0] for r in runs]
    med = runs[len(runs) // 2]
    r = {"device": torch.cuda.get_device_name(), "size": a.size, "mode": a.mode, "batch": a.batch,
         "prompt": a.prompt, "gen": g, "reps": a.reps, "total_ms": round(med[0], 1),
         "total_min": round(tot[0], 1), "total_max": round(tot[-1], 1),
         "prefill_ms": round(med[1], 1), "decode_ms": round(med[2], 1),
         "tok_per_s": round(a.batch * g / med[0] * 1e3, 1),
         "decode_tok_per_s": round(a.batch * g / med[2] * 1e3, 1), "masks": a.masks,
         "graph": a.graph, "graph_prefill": a.graph_prefill, "torch": torch.__version__, "real": a.real,
         "weight_mem_GB": round(weight_mem, 2), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(json.dumps(r), flush=True)
    if a.out:
        open(a.out, "a").write(json.dumps(r) + "\n")

if RW is not None:  # sanity: greedy continuation of the real prompt through the same graph-captured decode path
    with torch.no_grad():
        prefill(); gen = [tok[0, 0].item()]
        for _ in range(23):
            graph.replay(); tok.copy_(static_logits[:, -1:].argmax(-1)); pos.add_(1); gen.append(tok[0, 0].item())
    print("SAMPLE " + json.dumps({"mode": a.mode, "batch": a.batch, "prompt": a.prompt,
                                  "continuation": tokz.decode(gen)}), flush=True)
