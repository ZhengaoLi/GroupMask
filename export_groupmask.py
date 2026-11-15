import math
import sys
import time
import os
from pathlib import Path
from typing import Optional
import torch
import tqdm

from datasets import load_dataset
from datasets import IterableDataset
from flashlm.compression.semi_structure.semi_pruning_helper import model_replace_with_qk_share
from flashlm.data import load_hf_dataset_pile_dedup, dataloader_creator

from flashlm.compression.factorize.hypernetwork import hard_sample, hypernetwork  # type used elsewhere not needed here
from flashlm.compression.factorize.param_util import collect_info_reg, help_functions_hn, unwrap_model
from flashlm.compression.factorize.factorization_helper import model_replace, model_SVD_and_replace, get_approx_fisher, Repalce_Attention, hard_rank_pruning, model_SVD_and_replace_test, partial_defactorize_modules, round_mid_dim

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaConfig
from flashlm.models import FlashLlamaForCausalLM, FlashLlamaDecoderLayer, FlashLlamaTokenizer
from flashlm.utils import DistributedEnv
from torch import autocast
import torch.nn as nn
from typing import Tuple, Optional

def precache_masks(model, device="cuda:0"):
    device = torch.device(device)
    for m in model.modules():
        # Works for both factorize and semi_structure virtual ops
        if hasattr(m, "forward") and hasattr(m, "_cached_mask"):
            try:
                m._forward_pre_hooks.clear()
            except Exception:
                pass
            with torch.no_grad():
                _ = m.forward(device=device)
    torch.cuda.empty_cache()

@torch.no_grad()
def summarize_param(vectors, param_reg, tag="[GATE]", hard: bool = True):
    gate_total = 0
    gate_keep  = 0.0

    # ---------
    # param-level (weighted by matrix size) stats
    # ---------
    weighted_kept_params = 0.0
    weighted_total_params = float(param_reg.sum_ori_params)

    for i, v in enumerate(vectors):
        v = v.detach()

        # choose hard vs soft keep-count
        if hard:
            # hard keep count (0/1)
            v_keep = (v > 0.5).float()
        else:
            # soft keep expectation
            v_keep = v.float().clamp_(0.0, 1.0)

        # gate-level
        gate_total += v_keep.numel()
        gate_keep  += v_keep.sum().item()

        # structure gate fraction -> maps to parameter fraction
        denom = float(param_reg.in_group_list[i] * param_reg.out_group_list[i])
        if denom <= 0:
            continue

        keep_frac = v_keep.mean().item()  # == sum/numel
        # alternative (equivalent): keep_frac = v_keep.sum().item() / denom

        # this structure's parameter count
        struct_params = float(param_reg.in_dim_list[i] * param_reg.out_dim_list[i])

        weighted_kept_params += keep_frac * struct_params

    gate_keep_rate = gate_keep / max(1, gate_total)
    gate_spars     = 1.0 - gate_keep_rate

    param_keep_rate = weighted_kept_params / max(1e-12, weighted_total_params)
    param_spars     = 1.0 - param_keep_rate

    mode = "HARD(>0.5)" if hard else "SOFT(E[v])"
    print(f"{tag} mode={mode} | gate keep: {gate_keep_rate:.6f} -> gate spars: {gate_spars:.6f} | "
          f"param-weighted keep: {param_keep_rate:.6f} -> param spars: {param_spars:.6f}")

    return {
        "gate_keep": gate_keep_rate,
        "gate_spars": gate_spars,
        "param_keep": param_keep_rate,
        "param_spars": param_spars,
        "gate_n": gate_total,
        "param_total": weighted_total_params,
    }

def apply_hn_gates(model, hn, hn_helper, mode: str = "soft", device='cuda:0'):
    """Apply gates from hn to model.
    mode: "soft" to mirror training forward; "hard" to binarize.
    """
    dev = torch.device(device)
    with torch.no_grad():
        if mode == "soft":
            was_training = hn.training
            hn.train()  # ensure forward returns soft gates
            from torch import autocast as _autocast
            with _autocast(device_type='cuda', enabled=False):
                if hasattr(hn, "bi_GRU"):  # hypernetwork variant expects an input
                    inputs = hn.inputs.to(next(hn.parameters()).device)
                    vectors = hn(inputs)
                else:  # simplifed_gate takes no input
                    vectors = hn()
            if not was_training:
                hn.eval()
        else:  # hard mode
            hn.eval()
            if hasattr(hn, "hard_output"):
                vectors = hn.hard_output()
            else:
                # fallback: use current forward
                vectors = hn(hn.inputs.to(next(hn.parameters()).device)) if hasattr(hn, "inputs") else hn()
        hn_helper.set_gate_vectors(model, vectors)
        # clear cached masks and rebuild once to reflect new vectors
        for m in model.modules():
            if hasattr(m, "_cached_mask"):
                m._cached_mask = None
        precache_masks(model, device=dev)

@torch.no_grad()
def apply_hn_gates_new(model, hn, hn_helper, device="cuda:0", param_reg=None, is_simple=False):
    assert param_reg is not None
    dev = torch.device(device)

    hn.eval()
    with torch.autocast("cuda", enabled=False):

        if is_simple:
            soft_vectors = hn()
            if hasattr(hn, "hard_output"):
                hard_vectors = hn.hard_output()
            else:
                hard_vectors = [hard_sample(v) for v in soft_vectors]
        else:
            x = torch.zeros(1, device=next(hn.parameters()).device)
            hn.eval_return_soft = True
            soft_vectors = hn(x)

            hn.eval_return_soft = False
            if hasattr(hn, "hard_output"):
                hard_vectors = hn.hard_output()
            else:
                hard_vectors = hn(x)

        gate_vectors = [
            hard_vectors[i].float() * soft_vectors[i].float()
            for i in range(len(soft_vectors))
        ]

    # soft-vs-hard gate analysis.
    # GATE_HIST_JSON=<file>: dump the distribution of the noise-free soft gate values.
    # EXPORT_GATE_SOFT=1: export W * soft gate (no rounding) instead of the hard mask.
    # GATE_DUMP_LOGITS=<file.pt>: save the effective gate logits u (gate = sigmoid((u + G)/T)) for every group,
    # u = p + base for simplifed_gate, u = head output + base for hypernetwork; GATE_DUMP_ONLY=1 exits afterwards.
    if os.environ.get("GATE_DUMP_LOGITS"):
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            if is_simple:
                u = torch.cat([(q.detach().float() + hn.base).flatten() for q in hn.p_list]).cpu()
            else:
                os.environ["SEMI_RETURN_LOGITS"] = "1"
                hn.train()
                u = torch.cat([v.detach().float().flatten() for v in hn(torch.zeros(1, device=next(hn.parameters()).device))]).cpu()
                hn.eval()
                os.environ["SEMI_RETURN_LOGITS"] = "0"
        torch.save({"u": u, "T": float(hn.T), "base": float(hn.base)}, os.environ["GATE_DUMP_LOGITS"])
        print("[GATE-DUMP]", os.environ["GATE_DUMP_LOGITS"], u.numel(), "T", hn.T, "base", hn.base, flush=True)
        if os.environ.get("GATE_DUMP_ONLY") == "1":
            import sys as _sys
            _sys.exit(0)
    if os.environ.get("GATE_HIST_JSON") or os.environ.get("EXPORT_GATE_SOFT") == "1":
        with torch.autocast("cuda", enabled=False):
            if is_simple:  # simplifed_gate (hypernetwork2) in det-eval mode already returns sigmoid((p+base)/T)
                det_soft = [v.float() for v in hn()]
            else:          # hypernetwork: train-mode forward without Gumbel noise = sigmoid((logit+base)/T)
                os.environ["SEMI_TRAIN_DETERMINISTIC"] = "1"
                hn.train()
                det_soft = [v.float() for v in hn(torch.zeros(1, device=next(hn.parameters()).device))]
                hn.eval()
        allv = torch.cat([v.flatten() for v in det_soft])
        if os.environ.get("GATE_HIST_JSON"):
            import json as _json
            edges = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001]
            cnt = [((allv >= a) & (allv < b)).sum().item() for a, b in zip(edges[:-1], edges[1:])]
            info = {"n": allv.numel(), "edges": edges, "counts": cnt,
                    "frac_lt_0.01": (allv < 0.01).float().mean().item(), "frac_gt_0.99": (allv > 0.99).float().mean().item(),
                    "frac_0.1_0.9": ((allv > 0.1) & (allv < 0.9)).float().mean().item(),
                    "mean": allv.mean().item(), "hard_keep": (allv > 0.5).float().mean().item()}
            _json.dump(info, open(os.environ["GATE_HIST_JSON"], "w"), indent=1)
            print("[GATE-HIST]", _json.dumps({k: v for k, v in info.items() if k != "counts"}), flush=True)
        if os.environ.get("EXPORT_GATE_SOFT") == "1":
            gate_vectors = det_soft
            print("[EXPORT] using SOFT gates (no rounding); mean gate", allv.mean().item(), flush=True)

    summarize_param(hard_vectors, param_reg, tag="[EVAL-HARD]", hard=True)
    summarize_param(gate_vectors, param_reg, tag="[EVAL-HARDxSOFT]", hard=False)

    hn_helper.set_gate_vectors(model, gate_vectors)

    for m in model.modules():
        if hasattr(m, "_cached_mask"):
            m._cached_mask = None
    precache_masks(model, device=str(dev))



def load_eval_data(dataset_name: str) -> str:
    # this mimics gptq datautils
    if dataset_name == "wikitext":
        # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # testdata = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testdata = "\n\n".join(testdata["sentence"])
    elif dataset_name == 'pile':
        testdata = load_hf_dataset_pile_dedup('validation',2)
        print(testdata)
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testdata = " ".join(testdata[:1100]["text"])

    elif dataset_name == "quick":
        testdata = "Hello world. " * 2048
    else:
        raise ValueError("invalid dataset name (wikitext, ptb, c4, pile, quick are allowed)")
    return testdata

@torch.no_grad()
def save_hf_model(model: nn.Module, tokenizer, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    try:
        model_cpu = model.to('cpu')
    except Exception:
        model_cpu = model
    model_cpu.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] Saved HF model to: {save_dir}")

# ==== 1) Bake: multiply the masks/gates into the weights ====
@torch.no_grad()
def _maybe_to(t, ref):
    if t is None: 
        return None
    return t.to(dtype=ref.dtype, device=ref.device)

@torch.no_grad()
def _find_weight_and_mask(module: nn.Module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
        w = module.weight
        m = getattr(module, "_cached_mask", None)
        return w, m

    inner = getattr(module, "linear", None)
    if inner is not None and hasattr(inner, "weight"):
        w = inner.weight
        m = getattr(module, "_cached_mask", None)
        if m is None:
            m = getattr(inner, "_cached_mask", None)
        if m is None:
            vo = getattr(module, "virtual_operation", None)
            if vo is not None:
                m = getattr(vo, "_cached_mask", None)
        return w, m

    return None, None

# @torch.no_grad()
# def bake_masks_inplace(model: nn.Module, verbose: bool = True):
#     baked, skipped = 0, 0
#     for mod in model.modules():
#         w, mask = _find_weight_and_mask(mod)
#         if w is None or mask is None:
#             skipped += 1
#             continue

#         mask = mask.to(dtype=w.dtype, device=w.device)

#         if mask.shape != w.shape:
#             if mask.dim() == 1 and w.dim() == 2:
#                 if mask.numel() == w.shape[0]:
#                     mask = mask.view(-1, 1)
#                 elif mask.numel() == w.shape[1]:
#                     mask = mask.view(1, -1)

#         try:
#             w.data.mul_(mask)
#         except Exception as e:
#             if verbose:
#                 print(f"[bake] shape mismatch: {mod.__class__.__name__}, w={tuple(w.shape)}, mask={tuple(mask.shape)} ({e})")
#             skipped += 1
#             continue

#         scale = getattr(mod, "scale_weight", None)
#         inner = getattr(mod, "linear", None)
#         if scale is None and inner is not None:
#             scale = getattr(inner, "scale_weight", None)
#         if isinstance(scale, torch.Tensor):
#             scale = scale.to(dtype=w.dtype, device=w.device)
#             try:
#                 w.data.mul_(scale)
#             except Exception as e:
#                 if verbose:
#                     print(f"[bake] scale apply failed: {tuple(w.shape)} x {tuple(scale.shape)} ({e})")
#         baked += 1

#     if verbose:
#         print(f"[bake] baked={baked}, skipped={skipped}")

@torch.no_grad()
def defactorize_and_unwrap(model: nn.Module, verbose: bool = True):
    try:
        partial_defactorize_modules(model)
        if verbose: print("[unwrap] partial_defactorize_modules done")
    except Exception as e:
        if verbose: print(f"[unwrap] partial_defactorize_modules skipped: {e}")

    try:
        unwrap_model(model)
        if verbose: print("[unwrap] unwrap_model done")
    except Exception as e:
        if verbose: print(f"[unwrap] unwrap_model skipped: {e}")

@torch.no_grad()
def unwrap_wrapped_linears(model: nn.Module, verbose: bool = True):
    def _replace_in_parent(parent: nn.Module, child_name: str, new_child: nn.Module):
        setattr(parent, child_name, new_child)

    replaced = 0
    def _walk(parent: nn.Module):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            inner = getattr(child, "linear", None)
            if isinstance(inner, nn.Linear):
                _replace_in_parent(parent, name, inner)
                replaced += 1
                _walk(inner)
            else:
                _walk(child)
    _walk(model)
    if verbose:
        print(f"[unwrap] replaced wrapped linears: {replaced}")

@torch.no_grad()
def export_hf_readable(model: nn.Module, tokenizer, save_dir: str, device: str = "cuda"):
    model.eval()
    try:
        precache_masks(model, device=device)
        print("[export] precache_masks done")
    except Exception:
        pass

    bake_masks_inplace(model, verbose=True)
    defactorize_and_unwrap(model, verbose=True)

    model.to("cpu")
    if hasattr(model, "config"):
        model.config.model_type = "llama"
        model.config.architectures = ["LlamaForCausalLM"]

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    print(f"[export] saved HF-readable model to: {save_dir}")

@torch.no_grad()
def bake_masks_inplace(model: nn.Module, verbose: bool = True, debug: bool = True):
    baked, skipped = 0, 0
    total_params = 0
    zeroed_params = 0

    for name, mod in model.named_modules():
        w, mask = _find_weight_and_mask(mod)
        if w is None or mask is None:
            skipped += 1
            continue

        mask = mask.to(dtype=w.dtype, device=w.device)

        if mask.shape != w.shape:
            if mask.dim() == 1 and w.dim() == 2:
                if mask.numel() == w.shape[0]:
                    mask = mask.view(-1, 1)
                elif mask.numel() == w.shape[1]:
                    mask = mask.view(1, -1)

        if debug:
            print(f"\n[DEBUG] Layer: {name}")
            print(f"  weight shape: {tuple(w.shape)}")
            print(f"  mask shape  : {tuple(mask.shape)}")
            print(f"  mask min/max: {mask.min().item():.4f} / {mask.max().item():.4f}")
            print(f"  mask mean   : {mask.float().mean().item():.6f}")
            print(f"  weight norm(before): {w.norm().item():.6f}")

        before_zero = (w == 0).sum().item()
        total_params += w.numel()

        try:
            w.data.mul_(mask)
        except Exception as e:
            print(f"[bake] shape mismatch: {name}, {e}")
            skipped += 1
            continue

        after_zero = (w == 0).sum().item()
        zeroed_params += (after_zero - before_zero)

        if debug:
            print(f"  weight norm(after): {w.norm().item():.6f}")
            print(f"  zeros added      : {after_zero - before_zero}")

        baked += 1

    if verbose:
        print("\n==================== BAKE SUMMARY ====================")
        print(f"baked layers : {baked}")
        print(f"skipped      : {skipped}")
        print(f"total params : {total_params}")
        print(f"zeroed params: {zeroed_params}")
        print(f"sparsity     : {zeroed_params / max(total_params,1):.6f}")
        print("======================================================\n")

def evaluate(
    env: DistributedEnv,
    model, 
    tokenizer, 
    datasets="wikitext,ptb,c4", 
    block_size=2048, 
    hn_helper=None, 
    static_flag=False,
    batch_limit=200,
    max_eval_tokens=131072,
    ignored_token=-1,
):
    device_id = env.local_rank
    model.eval().cuda()
    print("[DBG] Enter evaluate()")
    total_toks = 0

    with torch.inference_mode(): 
        for dsname in datasets.split(","):
            t0 = time.time()
            print(f"[DBG] Begin load_eval_data('{dsname}') ...")
            test_string = load_eval_data(dsname)
            print(f"[DBG] Done load_eval_data('{dsname}') in {time.time()-t0:.2f}s")
            torch.backends.cuda.matmul.allow_tf32 = True

            enc = tokenizer(test_string, return_tensors="pt")
            input_ids = enc["input_ids"].cuda(non_blocking=True)
            if input_ids.shape[1] > 256 * 2048: 
                input_ids = input_ids[:, : 256 * 2048]

            seq_len = int(input_ids.shape[1])
            step = int(block_size)

            # warm_len = min(step, seq_len)
            warm_len = min(128, block_size, input_ids.shape[1]) 

            print("[DBG] Begin warmup forward in evaluate()")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(input_ids[:, :warm_len], use_cache=False)
            print("[DBG] Finished warmup forward in evaluate()")

            print(f"Dataset {dsname} encoded with shape: {input_ids.shape}")

            nlls = torch.zeros((), device="cuda", dtype=torch.float64)
            toks = 0
            batch_count = 0
            i = 0
            past = None
            attention_mask = (input_ids != ignored_token).long().to(device_id)  # ORIGINAL: used ignored_token (BOS before), not PAD
            end = min(i + step, seq_len)
            print("[DBG] Begin first main forward with cache=True")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids[:, i:end], use_cache=True, attention_mask=attention_mask,  past_key_values=None)
            print("[DBG] Finished first main forward")
            logits = out.logits
            past = out.past_key_values

            if logits.size(1) > 1:
                nlls = nlls + torch.nn.functional.cross_entropy(
                    logits[0, :-1], input_ids[0, i+1:end].to(torch.long), reduction="sum"
                ).to(torch.float64)
                toks += (end - i - 1)
            batch_count += 1
            i = end

            total_steps = min(batch_limit, (max(seq_len - end, 0) + step - 1) // step) + 1
            pbar = tqdm.tqdm(total=total_steps, disable=False)
            pbar.update(1) 

            # BOUNDED-MEMORY evaluation: overlapping window (stride) evaluation for numerically accurate perplexity under context-length cap.
            # Sliding context evaluation: each chunk has up to `step` tokens of left context.
            # We only score the *new* tokens in each chunk, so accuracy matches a context cap of `step`.
            pos = i
            while pos < seq_len and batch_count < batch_limit:
                # Evaluate up to `step` new tokens in this iteration
                eval_len = min(step, seq_len - pos)
                # Build segment with left context capped at `step`
                ctx_start = max(0, pos - step)
                seg_end = pos + eval_len
                segment = input_ids[:, ctx_start:seg_end]  # shape: [1, ctx + eval_len]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(segment, use_cache=False)
                logits = out.logits  # [1, ctx+eval_len, V]

                # We only compute loss for the last `eval_len` tokens (the new tokens),
                # which have at most `step` tokens of left context.
                if eval_len > 0:
                    ctx = segment.size(1) - eval_len
                    # logits indices [ctx-1 : -1] predict targets [ctx : ]
                    if logits.size(1) >= ctx + 1:  # guard for very small segments
                        nlls = nlls + torch.nn.functional.cross_entropy(
                            logits[0, ctx-1:-1], segment[0, ctx:].to(torch.long), reduction="sum"
                        ).to(torch.float64)
                        toks += eval_len

                batch_count += 1
                pbar.update(1)
                pos += eval_len

            pbar.close()

            ppl = math.exp((nlls / max(toks, 1)).item())
            print(f"Perplexity on {dsname}: {ppl:.2f}")
            total_toks += toks

            if hn_helper and not static_flag:
                print(hn_helper.get_router_logits())
                dy_head_list = hn_helper.get_dynamic_head_list()
                return dy_head_list


def main(
    hf_model: str = "meta-llama/Llama-2-7b-hf",
    hn_path: str = "outputs/hn-ckpt-final.pt",
    #semi_config: str = None,
    compressed_model: Optional[str] = None,
    dynamic_evaluate: bool = False,
    share_evaluate: bool = False,
    prune_evaluate: bool = False,
    raw_prune_evaluate: bool = False,
    group_moe_evaluate: bool = False,
    semi_evaluate: bool = False,
    semi_prune_evaluate: bool = False,
    semi_scratch_evaluate: bool = False,
    attn_experts: bool = False,
    simple_gate: bool = False,
    mlp_gate: bool = False,
    use_reinmax: bool = False,
    u_rank:float = 0.75,
    dataset:str ="wikitext,ptb",
    groups_in_dim:int = 1024,
    groups_out_dim:int = 1,
    hn_groups:int =1,
    block_size:int = 1024,
    T:float = 0.4,
    semi_params:bool = False,
    expand_rate:int = 1,
    semi_p:float=0.5,
    eval_loss:bool = False,
    mlp_only:bool =False,
    scale_weight:bool = False,
    slice_gpt:bool = False,
    wo_repeat:bool = False,
    adapter:bool = False,
    rank:int = 16,
    num_kv_heads:int = 32,
    constrained:str = 'none',
    dynamic_experts:int = 8,
    dynamic_non_uniform:bool = False,
    static_flag: bool = False,
    lam: float = 16.0,
    mask_size_per_group: int = 4,
    keep_k: int = 2,
    batch_limit: int = 200,
    compile_flag: bool = False,
    eval_baseline: bool = False,
    baseline_model_name: str = "meta-llama/Llama-2-7b-hf",
    baseline_eager: bool = True,
    #block_size:int = 1024,
    #"wikitext,ptb,c4"
    gate_mode: str = "soft",
    save_hf_dir: str = "./masked_llama_hf",
    share_qk: bool = False,
    is_simple_gate = False,
) -> None:
    env = DistributedEnv()
    print(f"[ENV] Torch {torch.__version__}, CUDA {torch.version.cuda}, Device {torch.cuda.get_device_name(0)} SM {torch.cuda.get_device_capability()}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    # Ensure pad token exists for consistent CE/attention masks
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if semi_evaluate: # groupsparsity pruning
        # evaluate baseline llama2 
        if eval_baseline:
                print("\n[BASELINE] Evaluating baseline LLaMA-2 model for comparison...")
                base_attn_impl = "eager" if baseline_eager else "flash_attention_2"
                baseline_model = AutoModelForCausalLM.from_pretrained(
                    baseline_model_name,
                    attn_implementation=base_attn_impl,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
                baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
                if baseline_tokenizer.pad_token_id is None:
                    baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
                baseline_model.eval().cuda()
                evaluate(env, baseline_model, baseline_tokenizer, datasets=dataset, block_size=block_size, batch_limit=batch_limit)

        device_id = env.local_rank

        if hf_model == "microsoft/phi-1_5" or hf_model == "microsoft/phi-2" or slice_gpt:
            model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        else:
            # model = FlashLlamaForCausalLM.from_pretrained(hf_model)
            model = AutoModelForCausalLM.from_pretrained(hf_model, 
                                                         torch_dtype=torch.float32,
                                                        #  trust_remote_code=True,
                                                        #  attn_implementation="flash_attention_2",
                                                        #  attn_implementation="eager",
                                                         )
        # from flashlm.compression.
        from flashlm.compression.semi_structure.hypernetwork import hypernetwork, simplifed_gate
        if os.environ.get("EXPORT_GATE_V2") == "1":  # export with the same gate class used in training
            from flashlm.compression.semi_structure.hypernetwork2 import simplifed_gate
        from flashlm.compression.semi_structure.semi_pruning_helper import collect_info_reg, help_functions_hn, model_replace
        group_info = {}
        group_info['groups_in_dim'] = groups_in_dim
        group_info['groups_out_dim'] = groups_out_dim
        # llama models
        if share_qk==False:
            model_replace(model, device_id, group_info=group_info, hf_model='llama')
        else:    
            model_replace_with_qk_share(model, device_id, group_info=group_info, hf_model='llama', model_dim=slice_gpt, share_qk=True)

        precache_masks(model, device='cuda:0')
        
        param_reg = collect_info_reg(model, p = 0.5, lam = 1.0)
        if simple_gate:
            hn = simplifed_gate(t_structures = param_reg.structures, num_groups=hn_groups, reinmax=use_reinmax)
        else:
            hn = hypernetwork(t_structures = param_reg.structures, num_groups=hn_groups, reinmax=use_reinmax, param_flag=semi_params)
            hn.T = T

        hn_helper = help_functions_hn(param_reg.structures)
        #hn_helper.set_mask_status(model, use_mask=False)
        hn_helper.set_mask_status(model, use_mask=True)
        hn_helper.set_scale_weight(model, scale_weight=scale_weight)
        hn_stat_dict = torch.load(hn_path, map_location='cpu')
        
        with torch.no_grad():
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in hn_stat_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            hn.load_state_dict(new_state_dict)
            hn.eval().cuda()
            # apply_hn_gates(model, hn, hn_helper, mode=gate_mode, device='cuda:0')
            apply_hn_gates_new(model, hn, hn_helper, device='cuda:0', param_reg=param_reg, is_simple=is_simple_gate)

        precache_masks(model, device='cuda:0')
        # save model
        # save_hf_model(model, tokenizer, save_hf_dir)
        # export_hf_readable(model, tokenizer, save_hf_dir, assume_masks_ready=True, device="cuda:0")
        print("\n[CHECK] Running pre-bake forward...")
        model.eval().cuda()
        hn_helper.set_mask_status(model, use_mask=True)
        precache_masks(model, device="cuda:0")

        dummy = torch.randint(0, tokenizer.vocab_size, (1, 128)).cuda()
        with torch.no_grad():
            logits_before = model(dummy, use_cache=False).logits

        print("[CHECK] Pre-bake forward done.")

        bake_masks_inplace(model, verbose=True)
        unwrap_wrapped_linears(model, verbose=True)
        print(model.model.layers[0].mlp.down_proj) 
        assert not any(".linear" in n for n, _ in model.named_modules()), "wrapped .linear submodules remain"
        defactorize_and_unwrap(model, verbose=True)
        export_hf_readable(model, tokenizer, save_hf_dir, device='cuda')
        # print(model.model.layers[0].mlp.down_proj)          
        # print(model.model.layers[0].self_attn.q_proj)   

        print("\n[CHECK] Running post-bake forward...")

        # Important: disable the mask mechanism to avoid applying it twice
        hn_helper.set_mask_status(model, use_mask=False)
        for m in model.modules():
            if hasattr(m, "_cached_mask"):
                m._cached_mask = None
                
        device = torch.device("cuda:0")
        model = model.to(device)
        dummy = dummy.to(device)
        with torch.no_grad():
            logits_after = model(dummy, use_cache=False).logits

        diff = (logits_before - logits_after).abs()
        print("\n==================== EQUIVALENCE CHECK ====================")
        print("max|diff| :", diff.max().item())
        print("mean|diff|:", diff.mean().item())
        print("============================================================\n")
        # evaluate PPL
        evaluate(env, model, tokenizer, datasets=dataset, block_size=block_size, batch_limit=batch_limit)


if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_float32_matmul_precision("high")
    CLI(main)