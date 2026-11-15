"""OWL (Yin et al., ICML 2024) per-block keep ratios for a dense LLaMA model.

Follows the reference OWL implementation:
  * Wanda metric |W_ij| * ||X_j||_2 for every linear in a Transformer block,
  * block outlier ratio = fraction of metric entries > M * mean(metric) over the whole block,
  * ratios rescaled to [0, 2*lambda], shifted so the mean keep ratio equals 1 - sparsity.
Calibration: 128 C4-en sequences of 2048 tokens (OWL default).
Writes a JSON list of per-block keep ratios (index = block id).
"""
import argparse, gzip, json, os, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sparsity", type=float, default=0.5)
ap.add_argument("--M", type=float, default=5.0)
ap.add_argument("--lam", type=float, default=0.08)
ap.add_argument("--nsamples", type=int, default=128)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--c4", default=os.environ.get("C4_JSON", "c4-train.00000-of-01024.json.gz"))
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
layers = model.model.layers

# calibration: random C4 documents long enough for one window each (as in OWL/Wanda get_c4)
random.seed(0)
docs = [json.loads(l)["text"] for l in gzip.open(a.c4, "rt")]
samples = []
while len(samples) < a.nsamples:
    ids = tok(docs[random.randint(0, len(docs) - 1)], return_tensors="pt").input_ids
    if ids.shape[1] > a.seqlen:
        i = random.randint(0, ids.shape[1] - a.seqlen - 1)
        samples.append(ids[:, i:i + a.seqlen])

# accumulate per-input-channel squared activation norms for every linear in every block
sq = {}
hooks = []
for li, layer in enumerate(layers):
    for name, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            key = (li, name)
            sq[key] = torch.zeros(mod.in_features, dtype=torch.float64, device="cuda")
            def hook(m, inp, out, key=key):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
                sq[key] += (x * x).sum(0)
            hooks.append(mod.register_forward_hook(hook))
with torch.no_grad():
    for s in samples:
        model(s.cuda())
for h in hooks:
    h.remove()

ratios = []
for li, layer in enumerate(layers):
    mets = []
    for name, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            norm = torch.sqrt(sq[(li, name)] / a.nsamples).float()
            mets.append((mod.weight.data.float().abs() * norm.reshape(1, -1)).flatten().cpu())
    w = torch.cat(mets)
    ratios.append(float((w > w.mean() * a.M).float().mean()) * 100)
    del mets, w

r = np.array(ratios)
keep = (r - r.min()) / (r.max() - r.min()) * a.lam * 2
keep = keep - keep.mean() + (1 - a.sparsity)
print("outlier ratio (%):", np.round(r, 3).tolist())
print("keep ratio per block:", np.round(keep, 4).tolist(), "mean", keep.mean())
json.dump({"keep": keep.tolist(), "outlier_pct": r.tolist(), "M": a.M, "lam": a.lam,
           "sparsity": a.sparsity, "nsamples": a.nsamples, "seqlen": a.seqlen}, open(a.out, "w"), indent=1)
