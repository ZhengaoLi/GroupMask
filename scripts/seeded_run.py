"""Seed torch/numpy/random from $SEED, then run a training recipe unchanged.
The recipes accept --rand_seed but never call manual_seed, so repeated runs share one init."""
import os, random, runpy, sys
import numpy as np
import torch

seed = int(os.environ["SEED"])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
print(f"[seeded_run] SEED={seed}", flush=True)
recipe = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(recipe)))  # recipes import flashlm from the project root
sys.argv = [recipe] + sys.argv[2:]
runpy.run_path(recipe, run_name="__main__")
