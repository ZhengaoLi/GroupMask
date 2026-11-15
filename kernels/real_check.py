"""Real-weight check on an exported GroupMask model (pruned groups are exactly zero).

1. Load the model in BF16 as a normal dense HF model (pruned weights are zeros) = reference.
2. Greedy-generate from a few prompts (static KV cache, eager), recording tokens and per-step logits;
   WikiText-2 test perplexity (non-overlapping 4096-token windows).
3. Convert every target linear to the packed 1x64 format (masks read from the zero pattern; mixed groups
   must be 0), freeing the dense weights.
4. Repeat 2 with the packed model (decode steps use the Triton sparse GEMV, prefill/PPL use unpack + cuBLAS),
   and compare tokens, logits and PPL; report decode speed (CUDA graph) for both.
"""
import argparse, json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from gsparse import GroupSparseLinear

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--ppl_windows", type=int, default=0, help="0 = full WikiText-2 test set")
ap.add_argument("--out", default="real_check.json")
ap.add_argument("--phase", required=True, choices=["dense", "packed"],
                help="dense: reference run, saves results; packed: loads on CPU, packs layer by layer onto the GPU, compares")
ap.add_argument("--ref", default="real_check_dense.pt")
a = ap.parse_args()
dev = torch.device("cuda")
BF = torch.bfloat16
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROMPTS = ["The capital of France is",
           "In machine learning, pruning is a technique that",
           "Once upon a time, in a small village by the sea,"]

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=BF, low_cpu_mem_usage=True).eval()
model.config._attn_implementation = "sdpa"
mixed = 0
if a.phase == "dense":
    model = model.to(dev)
else:
    # pack each target linear straight from CPU onto the GPU: the dense model never exists on the GPU
    with torch.no_grad():
        for name, mod in list(model.named_modules()):
            for t in TARGETS:
                lin = getattr(mod, t, None)
                if isinstance(lin, torch.nn.Linear):
                    W = lin.weight.data.to(dev)
                    O, I = W.shape
                    nz = (W != 0).reshape(O, I // 64, 64)
                    bm = nz.any(-1)
                    mixed += int((bm & ~nz.all(-1)).sum())     # groups that are only partly zero
                    setattr(mod, t, GroupSparseLinear(W, bm, 1, 64))
                    del lin, W, nz, bm
    model = model.to(dev)
    torch.cuda.empty_cache()


def mkcache(L):
    for kw in ({"config": model.config, "max_batch_size": 1, "max_cache_len": L, "device": dev, "dtype": BF},
               {"config": model.config, "max_cache_len": L}):
        try:
            return StaticCache(**kw)
        except TypeError:
            continue


@torch.no_grad()
def generate(prompt, n):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    cache = mkcache(ids.shape[1] + n + 16)
    out = model(ids, past_key_values=cache, use_cache=True)
    toks, logits = [], []
    nxt = out.logits[:, -1:].float()
    for _ in range(n):
        logits.append(nxt[0, -1].cpu())
        t = nxt.argmax(-1)
        toks.append(int(t))
        nxt = model(t, past_key_values=cache, use_cache=True).logits[:, -1:].float()
    return toks, torch.stack(logits)


@torch.no_grad()
def ppl(W=4096):
    from datasets import load_dataset
    torch.cuda.empty_cache()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    nw = ids.numel() // W if a.ppl_windows == 0 else min(a.ppl_windows, ids.numel() // W)
    nll, n = 0.0, 0
    for i in range(nw):
        x = ids[i * W:(i + 1) * W].to(dev)[None]
        lg = model(x, use_cache=False).logits.float()
        nll += torch.nn.functional.cross_entropy(lg[0, :-1], x[0, 1:], reduction="sum").item()
        n += W - 1
        del lg
    return math.exp(nll / n), n


@torch.no_grad()
def decode_speed(n=128, prompt_len=128):
    ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=dev)
    cache = mkcache(prompt_len + n + 16)
    t1 = torch.zeros(1, 1, dtype=torch.long, device=dev)
    def prefill():
        cache.reset()
        t1.copy_(model(ids, past_key_values=cache, use_cache=True).logits[:, -1:].argmax(-1))
    def step():
        return model(t1, past_key_values=cache, use_cache=True).logits
    prefill()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    prefill()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        lg = step()
    prefill(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        g.replay(); t1.copy_(lg[:, -1:].argmax(-1))
    torch.cuda.synchronize()
    return n / (time.time() - t0)


res = {"model": a.model, "phase": a.phase}
torch.cuda.synchronize()
res["weight_GB"] = round(torch.cuda.memory_allocated() / 1e9, 2)
if a.phase == "packed":
    res["mixed_groups"] = mixed
gens = [generate(p, a.gen) for p in PROMPTS]
res["ppl_2048"], res["ppl_tokens_2048"] = ppl(2048)
if a.phase == "packed":
    # dense @4096 does not fit in 16 GB next to the dense weights; packed does
    res["ppl_4096"], res["ppl_tokens_4096"] = ppl(4096)
res["decode_tok_s"] = round(decode_speed(), 1)
print(a.phase, "done", json.dumps(res), flush=True)
if a.phase == "dense":
    torch.save({"res": res, "gens": gens}, a.ref)
    raise SystemExit
D = torch.load(a.ref)
ref, sp, dres = D["gens"], gens, D["res"]
res["dense"] = dres
cmp = []
for p, (t0, l0), (t1, l1) in zip(PROMPTS, ref, sp):
    first_div = next((i for i, (x, y) in enumerate(zip(t0, t1)) if x != y), None)
    cmp.append({"prompt": p, "tokens_identical": t0 == t1, "first_divergence": first_div,
                "match_rate": sum(x == y for x, y in zip(t0, t1)) / len(t0),
                # logits compared only while both runs share the same context (up to and incl. first divergence)
                "max_abs_logit_diff": round(float((l0 - l1)[: (first_div + 1 if first_div is not None else len(t0))].abs().max()), 4),
                "max_rel_logit_diff": round(float((l0 - l1)[: (first_div + 1 if first_div is not None else len(t0))].abs().max() / l0.abs().max()), 6),
                "dense_text": tok.decode(t0), "packed_text": tok.decode(t1)})
res["generation"] = cmp
print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
