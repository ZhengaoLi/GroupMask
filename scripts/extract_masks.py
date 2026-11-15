"""Extract group masks from an exported GroupMask model (pruned groups are exactly zero).

For every target linear (q/k/v/o/gate/up/down) find the group orientation for a group of
`g` contiguous weights (along the input dim = 1 x g row segments, or along the output dim =
g x 1 column segments, or a square s x s block), check that every group is all-zero or has
nonzeros, and save packed bit masks + metadata to an .npz.
Usage: python extract_masks.py <model_dir> <out.npz> <gh> <gw>   (gh x gw group, e.g. 1 64 or 8 8)
"""
import json, os, sys
import numpy as np
import torch
from safetensors import safe_open

model_dir, out, gh, gw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
targets = [k for k in idx if k.endswith(".weight") and any(p in k for p in
           ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))]
files = {}
meta, arrays = {}, {}
tot = kept = 0
for k in sorted(targets, key=lambda s: (int(s.split(".layers.")[1].split(".")[0]), s)):
    f = idx[k]
    if f not in files:
        files[f] = safe_open(os.path.join(model_dir, f), "pt")
    W = files[f].get_tensor(k)
    O, I = W.shape
    nz = W != 0
    best = None
    # candidate orientations: (rows per group, cols per group)
    for (a, b) in {(gh, gw), (gw, gh)}:
        if O % a or I % b:
            continue
        g = nz.reshape(O // a, a, I // b, b).permute(0, 2, 1, 3).reshape(O // a, I // b, a * b)
        anyg = g.any(-1)
        # a group is "clean" if it is all zero or (almost) all nonzero; count mixed groups
        frac = g.float().mean(-1)
        mixed = ((frac > 0) & (frac < 0.9)).sum().item()
        if best is None or mixed < best[0]:
            best = (mixed, a, b, anyg)
    mixed, a, b, anyg = best
    m = anyg.numpy()
    arrays[k] = np.packbits(m.reshape(-1))
    meta[k] = {"shape": [O, I], "group": [a, b], "grid": list(m.shape), "keep": float(m.mean()), "mixed_groups": int(mixed)}
    tot += O * I; kept += float(m.mean()) * O * I
    print(f"{k:55s} {O}x{I} group={a}x{b} keep={m.mean():.4f} mixed={mixed}", flush=True)
np.savez_compressed(out, **{k.replace(".", "__"): v for k, v in arrays.items()})
json.dump({"model_dir": model_dir, "global_keep": kept / tot, "layers": meta}, open(out.replace(".npz", ".json"), "w"), indent=1)
print("global keep:", kept / tot)
