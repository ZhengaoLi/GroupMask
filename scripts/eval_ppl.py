"""Re-evaluate WikiText-2 perplexity of dense and exported GroupMask models under
several protocols, so that every number in the paper can be compared on one setting.

Protocols
  slide-B : identical to export_groupmask.evaluate() with block_size=B (first chunk scored with
            in-chunk context, later chunks score B new tokens with B tokens of left context)
  nonov-L : standard GPTQ/SparseGPT/Wanda protocol, non-overlapping chunks of L tokens,
            ppl = exp(mean over chunks of mean token CE)
Weights are loaded in fp32 and run under bf16 autocast, as export_groupmask.py does.
"""
import argparse, math, os, sys, time
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def wikitext2_ids(tok):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt")["input_ids"]
    if ids.shape[1] > 256 * 2048:  # same cap as export_groupmask.evaluate
        ids = ids[:, : 256 * 2048]
    return ids.cuda()


@torch.inference_mode()
def ppl_slide(model, ids, step, batch_limit=2000):
    seq_len = ids.shape[1]
    nll = torch.zeros((), device="cuda", dtype=torch.float64)
    toks = 0
    end = min(step, seq_len)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids[:, :end], use_cache=False).logits
    nll += F.cross_entropy(logits[0, :-1].float(), ids[0, 1:end], reduction="sum").double()
    toks += end - 1
    pos, batches = end, 1
    while pos < seq_len and batches < batch_limit:
        eval_len = min(step, seq_len - pos)
        ctx_start = max(0, pos - step)
        seg = ids[:, ctx_start : pos + eval_len]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(seg, use_cache=False).logits
        ctx = seg.shape[1] - eval_len
        nll += F.cross_entropy(logits[0, ctx - 1 : -1].float(), seg[0, ctx:], reduction="sum").double()
        toks += eval_len
        pos += eval_len
        batches += 1
    return math.exp((nll / toks).item()), toks


@torch.inference_mode()
def ppl_nonoverlap(model, ids, seqlen):
    n = ids.shape[1] // seqlen
    nlls = []
    for i in range(n):
        x = ids[:, i * seqlen : (i + 1) * seqlen]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, use_cache=False).logits
        loss = F.cross_entropy(logits[0, :-1].float(), x[0, 1:])
        nlls.append(loss.double() * seqlen)
    return math.exp((torch.stack(nlls).sum() / (n * seqlen)).item()), n * seqlen


def measured_sparsity(model):
    zeros = total = 0
    per = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in TARGETS:
            w = mod.weight
            z = (w == 0).sum().item()
            zeros += z
            total += w.numel()
            k = name.split(".")[-1]
            pz, pt = per.get(k, (0, 0))
            per[k] = (pz + z, pt + w.numel())
    return zeros / max(total, 1), {k: z / t for k, (z, t) in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="HF repo id or exported dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--slide", default="1024,2048,4096")
    ap.add_argument("--nonov", default="2048,4096")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=a.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=(torch.bfloat16 if a.dtype == "bf16" else torch.float32),
                                                 trust_remote_code=a.trust_remote_code).cuda().eval()
    ids = wikitext2_ids(tok)
    sp, per = measured_sparsity(model)
    print(f"RESULT|{a.name}|sparsity|{sp:.6f}|" + ",".join(f"{k}={v:.4f}" for k, v in sorted(per.items())), flush=True)
    for b in [int(x) for x in a.slide.split(",") if x]:
        p, n = ppl_slide(model, ids, b)
        print(f"RESULT|{a.name}|slide-{b}|{p:.4f}|tokens={n}", flush=True)
    for L in [int(x) for x in a.nonov.split(",") if x]:
        p, n = ppl_nonoverlap(model, ids, L)
        print(f"RESULT|{a.name}|nonov-{L}|{p:.4f}|tokens={n}", flush=True)
    print(f"DONE|{a.name}|{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
