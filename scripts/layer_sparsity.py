"""Per-layer, per-projection fraction of pruned weights of an exported model (zeros / numel), as JSON.
python layer_sparsity.py <out.json> <name>=<model dir> [...]"""
import json, os, re, sys
import torch
from safetensors import safe_open

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
out = {}
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1)
    idx = os.path.join(d, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    res = {}
    for f in files:
        with safe_open(os.path.join(d, f), framework="pt", device="cpu") as h:
            for k in h.keys():
                m = re.match(r"model\.layers\.(\d+)\..*\.(\w+_proj)\.weight$", k)
                if not m or m.group(2) not in TARGETS:
                    continue
                w = h.get_tensor(k)
                res.setdefault(int(m.group(1)), {})[m.group(2)] = 1.0 - torch.count_nonzero(w).item() / w.numel()
    L = max(res) + 1
    out[name] = {"layers": L, "proj": list(TARGETS), "pruned": [[res[l][t] for t in TARGETS] for l in range(L)]}
    print(name, L, "mean", sum(sum(r) for r in out[name]["pruned"]) / (7 * L), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
