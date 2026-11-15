import os
import re
import glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
import time
import datetime
from functools import partial
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer as HFLlamaDecoderLayer
from datasets import IterableDataset
from flashlm.models import FlashLlamaForCausalLM, FlashLlamaDecoderLayer, FlashLlamaTokenizer
from flashlm.utils import DistributedEnv, softmax_fp32, log_softmax_fp32
from flashlm.data import distributed_mixed_datasets, dataloader_creator, load_hf_dataset_pile_dedup, load_hf_dataset_minipile, load_hf_dataset_wiki, load_hf_dataset_alpaca, load_hf_dataset_wizardlMv2, load_hf_dataset_mixed, load_hf_dataset_new_mixed, load_hf_dataset_orca_dpo
from flashlm.compression.weightsharing import is_tie_word_embeddings

from flashlm.compression.factorize.param_util import unwrap_model
import bitsandbytes as bnb

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    always_wrap_policy,
    enable_wrap,
    wrap,
)

from flashlm.compression.semi_structure.semi_pruning_helper3 import collect_info_reg, help_functions_hn, model_replace, model_replace_with_qk_share, help_functions_share, collect_info_share
from flashlm.compression.semi_structure.hypernetwork2 import hypernetwork, simplifed_gate, hard_sample
# TORCH_SHARING_STRATEGY=file_descriptor avoids the torch_shm_manager ("Shared memory manager connection has timed out")
if os.environ.get("TORCH_SHARING_STRATEGY"):
    import torch.multiprocessing as _tmp_mp
    _tmp_mp.set_sharing_strategy(os.environ["TORCH_SHARING_STRATEGY"])

def _find_param_aliases(model):
    seen = {}
    dups = []
    for name, p in model.named_parameters():
        pid = id(p)
        if pid in seen:
            dups.append((seen[pid], name, tuple(p.shape)))
        else:
            seen[pid] = name
    return dups

# === Parameter mapping snapshot/diff utilities ===
def _snapshot_param_meta(model):
    meta = {}
    for name, p in model.named_parameters():
        meta[name] = {"id": id(p), "shape": tuple(p.shape)}
    return meta

def _diff_param_meta(model, baseline_meta):
    changed = []
    for name, p in model.named_parameters():
        if name not in baseline_meta:
            changed.append((name, "NEW_PARAM", tuple(p.shape)))
            continue
        m = baseline_meta[name]
        if id(p) != m["id"]:
            changed.append((name, "REBOUND", tuple(p.shape), m["shape"]))
        elif tuple(p.shape) != m["shape"]:
            changed.append((name, "RESHAPED", tuple(p.shape), m["shape"]))
    return changed
def _flatten_to_1d(x, device=None):
    """Flatten Tensor / (nested) list/tuple/dict of Tensors into one 1D Tensor on device."""
    if torch.is_tensor(x):
        t = x
        if device is not None and t.device != device:
            t = t.to(device)
        return t.reshape(-1)

    if isinstance(x, dict):
        parts = [_flatten_to_1d(v, device=device) for v in x.values()]
        parts = [p for p in parts if p is not None and p.numel() > 0]
        return torch.cat(parts) if parts else None

    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            p = _flatten_to_1d(v, device=device)
            if p is not None and p.numel() > 0:
                parts.append(p)
        return torch.cat(parts) if parts else None

    return None

def kl_div_loss_with_ignore_index(predictions, targets, labels, ignore_index=-100):
    """
    Compute KL divergence loss with an option to ignore specific indices.
    
    Parameters:
    - predictions: Tensor of model outputs (logits) with shape (batch_size, num_classes).
    - targets: Tensor of target distributions (probabilities) with shape (batch_size, num_classes).
    - ignore_index: Index to ignore in the loss calculation, default is -100.
    
    Returns:
    - loss: KL divergence loss with ignored indices.
    """
    # Compute the log probabilities of the predictions
    log_predictions = torch.nn.functional.log_softmax(predictions, dim=-1)
    targets = torch.nn.functional.softmax(targets, dim=-1)
    # Mask the targets and predictions based on ignore_index
    mask = (labels != ignore_index).float().view(-1).to(predictions.get_device())
    # masked_log_predictions = log_predictions * mask
    # masked_targets = targets * mask

    # Compute KL divergence loss
    # loss = torch.nn.functional.kl_div(masked_log_predictions, masked_targets, reduction='batchmean')
    
    kl_div_per_position = torch.nn.functional.kl_div(log_predictions, targets, reduction='none')
    masked_loss = kl_div_per_position * mask.unsqueeze(-1)
    
    # Compute the mean loss across non-ignored positions
    loss = masked_loss.sum()/mask.sum()
    return loss

def kl_div_loss_with_ignore_index_softmax(predictions, targets, labels, ignore_index=-100):
    """
    Compute KL divergence loss with an option to ignore specific indices.
    
    Parameters:
    - predictions: Tensor of model outputs (logits) with shape (batch_size, num_classes).
    - targets: Tensor of target distributions (probabilities) with shape (batch_size, num_classes).
    - ignore_index: Index to ignore in the loss calculation, default is -100.
    
    Returns:
    - loss: KL divergence loss with ignored indices.
    """
    # Compute the log probabilities of the predictions
    #log_predictions = torch.nn.functional.log_softmax(predictions, dim=-1, dtype=torch.float32)
    #targets = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32).detach()
    # Mask the targets and predictions based on ignore_index
    mask = (labels != ignore_index).to(predictions.get_device())
    #.float().view(-1).to(predictions.get_device())
    # masked_log_predictions = log_predictions * mask
    # masked_targets = targets * mask

    # Compute KL divergence loss
    # loss = torch.nn.functional.kl_div(masked_log_predictions, masked_targets, reduction='batchmean')
    mask_flat = mask.view(-1)

    valid_log_probs = predictions[mask_flat]
    valid_target_probs = targets[mask_flat]

    loss = F.kl_div(
    #    F.log_softmax(valid_log_probs, dim=-1, dtype=torch.float32),
        log_softmax_fp32(valid_log_probs, dim=-1,),
        softmax_fp32(valid_target_probs, dim=-1,).detach(),
        #F.softmax(valid_target_probs, dim=-1, dtype=torch.float32).detach(),
        reduction="batchmean",
)
    
    # kl_div_per_position = torch.nn.functional.kl_div(log_predictions, targets, reduction='none')
    # masked_loss = kl_div_per_position * mask.unsqueeze(-1)
    
    # # Compute the mean loss across non-ignored positions
    # loss = masked_loss.sum()/mask.sum()
    return loss

import torch.nn.functional as F
def ForwardKLLoss(student_logits, teacher_logits, labels, ignore_index = -100) -> torch.Tensor:
    # Implementation from https://github.com/jongwooko/distillm
    # Computes the softmax of the teacher logits
    teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    # Computes the student log softmax probabilities
    student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    # Computes the forward KL divergence
    prod_probs = teacher_prob * student_logprob
    # Compute the sum
    x = torch.sum(prod_probs, dim=-1).view(-1)
    # We don't want to include the ignore labels in the average
    mask = (labels != ignore_index).int()
    # Loss is averaged over non-ignored targets
    return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

def TopKForwardKDLoss(student_logits, teacher_logits, labels, k=64, T=1.0, ignore_index=-100):
    """
    Top-K forward KD: only distill teacher's top-k tokens to remove tail noise.
    student_logits/teacher_logits: [B, S, V]
    labels: [B, S], ignore_index = -100
    """
    # flatten valid positions
    mask = (labels != ignore_index).view(-1)
    s = student_logits.view(-1, student_logits.size(-1))[mask]
    t = teacher_logits.view(-1, teacher_logits.size(-1))[mask]

    # temperature
    if T != 1.0:
        s = s / T
        t = t / T

    # top-k on teacher (use fp32 for stability)
    topv, topi = torch.topk(t.float(), k, dim=-1)          # [N, k]
    s_top = torch.gather(s.float(), dim=-1, index=topi)    # [N, k]

    # teacher prob over top-k only
    t_prob = torch.softmax(topv, dim=-1).detach()          # [N, k]
    s_logp = torch.log_softmax(s_top, dim=-1)              # [N, k]

    loss = -(t_prob * s_logp).sum(dim=-1).mean()           # scalar
    # standard KD scaling
    return (T * T) * loss

def round_to_block_size(current_rank, block_size=32):

    round_rank = max(block_size, (current_rank // block_size) * block_size)

    return round_rank

def pre_forward_warmup(model, device_id, seq_len):
    for module in model.modules():
        if hasattr(module, "virtual_operation") and hasattr(module.virtual_operation, "generate_pv"):
            module.virtual_operation.generate_pv(seed=0, p=0.5)
            _ = module.virtual_operation.forward(device=torch.device(f"cuda:{device_id}"))
    model.eval()
    device = torch.device(f"cuda:{device_id}")
    print("device:", device)
    dummy_input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    model.train()


def _parse_iter_from_name(path: str) -> int:
    m = re.search(r"hn-ckpt-iter-(\d+)-", os.path.basename(path))
    return int(m.group(1)) if m else -1

def _find_latest_hn_ckpt(out_dir: str) -> str | None:
    # Prefer explicit iter checkpoints
    iter_ckpts = sorted(glob.glob(os.path.join(out_dir, "hn-ckpt-iter-*-*.pt")), key=_parse_iter_from_name)
    if iter_ckpts:
        return iter_ckpts[-1]
    # Fallbacks
    finals = sorted(glob.glob(os.path.join(out_dir, "hn-ckpt-final-*.pt")))
    if finals:
        return finals[-1]
    any_ckpts = sorted(glob.glob(os.path.join(out_dir, "hn-ckpt-*.pt")))
    if any_ckpts:
        return any_ckpts[-1]
    return None

def main(
    out_dir: str = None,
    exp_name: str = 'semi',
    start_iter: int = 0, 
    rand_seed: int = None,
    learning_rate: float = None,
    batch_size: int = 1,
    hf_model: str = 'meta-llama/Llama-2-7b-hf',
    non_hf_tokenizer_path: str = None,
    dataset_list: list = ['wiki'],
    num_workers: int = 1,
    dataset_seed: int = 42,
    hn_block_size = 2048,
    lam: float = 16.0,
    hn_path:str = None,
    use_bf16: bool = False,
    compile_flag: bool = True,
    hn_lr: float = 1e-3,
    adam_8bit:bool = False,
    total_n_step: int = 100000,
    min_hn_lr: float = 1e-3,
    kd_loss: bool = True,
    mix_loss: bool = False,
    save_interval: int = 10000,  
    use_fsdp: bool = False,  
    slice_gpt: bool = False,
    use_minipile: bool = False,
    groups_in_dim: int = 1024,
    groups_out_dim: int = 1,
    simple_gate: bool = False,
    hn_groups: int = 16,
    gamma: float = 0.01,
    scale_weight: bool = False,
    soft_rank: bool = False,
    use_reinmax: bool = False,
    T: float = 0.4,
    p: float = 0.5,
    hard_flag: bool = False,
    semi_params: bool = False,
    use_ddp:bool = False,
    model_size:str = '7B',
    resume: bool = False,
    resume_dir: str | None = None,
    hidden_kd = False,
    share_qk: bool = False,
    uniform_sparsity: bool = False,
):
    env = DistributedEnv()
    # === Hyperparameters Logging Start ===
    if env.global_rank == 0:
        params = locals().copy()
        exclude = {'env', 'dateTimeObj', 'timestamp', 'tic', 'device_id'}
        print("\n" + "="*35 + " CONFIGURATION " + "="*35)
        print(f"{'Hyperparameter':<25} | {'Value'}")
        print("-" * 71)
        
        for k in sorted(params.keys()):
            if k not in exclude and not k.startswith('__'):
                val = params[k]
                val_str = str(val)
                if len(val_str) > 40:
                    val_str = val_str[:37] + "..."
                print(f"{k:<25} | {val_str}")
        print("="*71 + "\n")

    dist.init_process_group(
        "nccl",
        rank=env.global_rank,
        world_size=env.world_size,
        timeout=datetime.timedelta(seconds=3600*5),
    )
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # data_type = torch.float32
    print(data_type)
    # we disable fsdp for world_size=1 to avoid writeback mismatches due to param rebind/reshape
    if use_fsdp and dist.get_world_size() == 1:
        use_fsdp = False
        env.print_master("[FSDP] Disabled for world_size=1 to avoid writeback mismatches (no speed/memory benefit on 1 GPU).")
        
    # Decide out_dir: when resuming, prefer user-provided resume_dir/out_dir
    if resume:
        if resume_dir:
            out_dir = resume_dir
        elif out_dir is None:
            raise ValueError("--resume is True but neither --resume_dir nor --out_dir was provided.")
        # else: use the provided out_dir as resume target
    if out_dir is None:
        dateTimeObj = datetime.datetime.now()
        timestamp = dateTimeObj.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(os.environ.get('GROUPMASK_OUTPUT_ROOT', 'outputs'), exp_name, timestamp)
    if rand_seed is None:
        rand_seed = start_iter
        
    if learning_rate is None:
        llama_learning_rate_per_sample = 0.0003 / (4*1024*1024)
        learning_rate = min(llama_learning_rate_per_sample * batch_size * 4096 * env.world_size, 0.0003)
    
    if env.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    
    # GPU preparation
    device_id = env.local_rank
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    
    # prepare tokenizer
    # hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    # tokenizer = hf_tokenizer
    if non_hf_tokenizer_path:
        env.print_master('Using non_hf_tokenizer ...')
        tokenizer = FlashLlamaTokenizer(non_hf_tokenizer_path, output_type='list')
    # ignored_token = tokenizer.bos_token_id
    # print("###### ignored_token:", ignored_token)
    # IGNORE_INDEX = -100
    # PAD_ID = tokenizer.pad_token_id
    # ignored_token = IGNORE_INDEX
    
    if hf_model == "meta-llama/Llama-2-7b-hf" or hf_model == "meta-llama/Llama-2-13b-hf" or hf_model == "meta-llama/Meta-Llama-3-8B" or hf_model == "Qwen/Qwen3-8B" or hf_model == "Qwen/Qwen3-14B":
        model = AutoModelForCausalLM.from_pretrained(hf_model,
                                                    trust_remote_code=True,
                                                    # device_map="auto", 
                                                    # attn_implementation="flash_attention_2",
                                                    # torch_dtype=torch.float32,
                                                    torch_dtype=torch.bfloat16
                                                )
        model = model.to(device_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        ignored_token = tokenizer.pad_token_id
    else:
        model = FlashLlamaForCausalLM.from_pretrained(
            hf_model,
            # torch_dtype=torch.float32,
            torch_dtype=torch.bfloat16
        )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        env.print_master("[MEM] Enabled gradient checkpointing.")
    config = model.config
    print(model)
    
    # env.print_master(config)
    # env.print_master(model)

    # dataset
    tic = time.time()
    from flashlm.data.huggingface_dataset import load_hf_dataset_wiki
    if dataset_list == ['wiki']:
        train_dataset = load_hf_dataset_wiki(split='train', n_shards=env.world_size * num_workers, seed=dataset_seed)
    # use alpaca for training
    elif dataset_list == ['alpaca']:
        train_dataset = load_hf_dataset_alpaca(split='train', n_shards=env.world_size * num_workers, seed=dataset_seed)
    # C4-en calibration (one local shard), to match ProxSparse's calibration data
    elif dataset_list == ['c4']:
        from datasets import load_dataset as _load_dataset
        train_dataset = _load_dataset('json', data_files=os.environ.get('C4_JSON', 'c4-train.00000-of-01024.json.gz'),
                                      split='train').select_columns(['text']).to_iterable_dataset(num_shards=env.world_size * num_workers)
    train_dataloader_hn = dataloader_creator(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size, 
        block_size=hn_block_size,
        num_workers=num_workers,
        cycling=True,
        rank=env.global_rank,
        world_size=env.world_size,
        ignored_token=ignored_token,
    )

    toc = time.time() - tic
    env.print(f"Initializing training dataset (wiki) - done. Time elapsed (s): {toc:.2f}")
    # model replace
    group_info = {}
    group_info['groups_in_dim'] = groups_in_dim
    group_info['groups_out_dim'] = groups_out_dim
    if hf_model in ("meta-llama/Llama-2-7b-hf", "Qwen/Qwen2.5-7B", "meta-llama/Llama-2-13b-hf", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "meta-llama/Meta-Llama-3-8B") or slice_gpt:
        if share_qk==False:
            model_replace(model, device_id, group_info=group_info, hf_model='llama', model_dim=slice_gpt)
        else:
            model_replace_with_qk_share(model, device_id, group_info=group_info, hf_model='llama', model_dim=slice_gpt, share_qk=True)
        pre_forward_warmup(model, device_id, hn_block_size)
    # else:
    #     model_replace(model, group_info=group_info)
    dup_refs = _find_param_aliases(model)
    if dup_refs:
        env.print_master("[WARN] Found duplicated Parameter registrations (aliases):")
        for a, b, shp in dup_refs[:20]:
            env.print_master(f"    {a}  <==>  {b}  shape={shp}")
    else:
        env.print_master("[OK] No duplicated Parameter registrations detected.")
    # hypernetwork loading
    if share_qk:
        param_reg = collect_info_share(model, p=p, lam=lam)
        hn_helper = help_functions_share(param_reg.structures, gamma=gamma)
    else:
        param_reg = collect_info_reg(model, p=p, lam=lam, uniform=uniform_sparsity)
        hn_helper = help_functions_hn(param_reg.structures, gamma=gamma)
    # param_reg = collect_info_reg(model, p=p, lam=lam)
    # hn_helper = help_functions_hn(param_reg.structures, gamma=gamma)
    
    if simple_gate:
        hn = simplifed_gate(t_structures=param_reg.structures, num_groups=hn_groups, reinmax=use_reinmax)
    else:
        hn = hypernetwork(t_structures=param_reg.structures, num_groups=hn_groups, reinmax=use_reinmax, hard_flag=hard_flag, param_flag=semi_params)
        hn.T = T
        # data_type = torch.float16
        # hn = hn.to(data_type)

    hn_helper.set_mask_status(model, use_mask=True)
    hn_helper.set_scale_weight(model, scale_weight=scale_weight)
    if soft_rank:
        hn_helper.init_rank_reg(model)
    hn.to(device_id)
    if dist.is_initialized() and dist.get_world_size() > 1:
        hn = DDP(
            hn,
            device_ids=[device_id],
            output_device=device_id,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    model.to(device_id)
    if use_bf16:
        model = model.to(data_type).to(device_id)
        if use_fsdp:
            wrapped = 0
            for i, block in enumerate(model.model.layers):
                if isinstance(block, HFLlamaDecoderLayer):
                    model.model.layers[i] = FSDP(block, use_orig_params=True)
                    wrapped += 1
            env.print_master(f"[FSDP] Wrapped {wrapped} decoder layers only (no root wrap; embed_tokens & lm_head left outside).")
    else:
        model = model.to(device_id)
        if use_fsdp:
            wrapped = 0
            for i, block in enumerate(model.model.layers):
                if isinstance(block, HFLlamaDecoderLayer):
                    model.model.layers[i] = FSDP(block, use_orig_params=True)
                    wrapped += 1
            env.print_master(f"[FSDP] Wrapped {wrapped} decoder layers only (no root wrap; embed_tokens & lm_head left outside).")

    # Snapshot parameter identities and shapes after wrapping (baseline expected by FSDP)
    if use_fsdp:
        model.__baseline_param_meta__ = _snapshot_param_meta(model)
    else:
        model.__baseline_param_meta__ = None

    if compile_flag and not use_fsdp and not use_ddp:
        model = torch.compile(model)
    if use_ddp:
        model = DDP(model)
    
    # Make sure start_iter is in scope, may be overwritten by resume
    start_iter = start_iter  # from CLI default; may be overwritten by resume block

    # Resume from latest checkpoint if requested
    if resume:
        latest = _find_latest_hn_ckpt(out_dir)
        if latest is None:
            env.print_master(f"[RESUME] No checkpoint found in {out_dir}; starting fresh.")
        else:
            env.print_master(f"[RESUME] Loading HN checkpoint: {latest}")
            ckpt = torch.load(latest, map_location='cpu')
            # Strip possible Distributed prefixes
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in ckpt.items():
                new_state[k.replace('module.', '')] = v
            missing, unexpected = hn.load_state_dict(new_state, strict=False)
            if missing:
                env.print_master(f"[RESUME] Missing keys: {list(missing)[:5]} ...")
            if unexpected:
                env.print_master(f"[RESUME] Unexpected keys: {list(unexpected)[:5]} ...")
            # Infer start_iter
            start_iter = _parse_iter_from_name(latest)
            if start_iter < 0:
                # fallback: if it's a 'final' ckpt, keep start_iter as provided by CLI (default 0)
                start_iter = 0
            env.print_master(f"[RESUME] start_iter set to {start_iter}")

    print("################# training started")
    # train
    tic = time.time()
    train_hn(
        env,
        model,
        hn=hn,
        train_hn_data=train_dataloader_hn,
        hn_helper=hn_helper,
        param_reg=param_reg,
        # ignored_token=ignored_token,
        start_iter=start_iter,
        max_iter=total_n_step,
        bf_16=use_bf16,
        out_dir=out_dir,
        p=p,
        model_size=model_size,
        hn_block_size=hn_block_size,
        hn_lr=hn_lr,
        min_hn_lr=min_hn_lr,
        semi_params=semi_params,
        soft_rank=soft_rank,
        use_fsdp=use_fsdp,
        load_balance=False,
        dynamic_transit=1.0,
        save_interval=save_interval,
        adam_8bit=adam_8bit,
        kd_loss=kd_loss,
        mix_loss=mix_loss,
        hidden_kd = hidden_kd,
        ignored_token=tokenizer.pad_token_id,
        pad_id=tokenizer.pad_token_id,
        # pad_id=PAD_ID,
        simple_gate=simple_gate,
    )
    toc = time.time() - tic
    print("################# training finished")
    env.print_master(f"Total training time: {toc:.2f}")
    
def train_hn(
    env: DistributedEnv,
    model: torch.nn.Module,
    hn: torch.nn.Module or torch.nn.ModuleList,
    train_hn_data: IterableDataset,
    hn_helper,
    param_reg,
    experts_list = None,
    start_iter=0,
    ignored_token=-1,
    log_interval=1,
    max_iter=None,
    bf_16=True,
    use_fsdp=True,
    out_dir=None,
    p=None,
    model_size:str = '7B',
    hn_block_size=2048,
    hn_lr=1e-3,
    min_hn_lr=1e-3,
    use_sch=False,
    semi_params=False,
    soft_rank=False,
    load_balance = False,
    dynamic_transit = 1.0,
    save_interval=5000,
    scheduler_start_iter=9000,
    kd_loss = True, 
    mix_loss = False,
    adam_8bit=True,
    hidden_kd = False,
    pad_id=None,
    # pad_id=None,
    simple_gate = False,
) -> None:
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_id = env.local_rank

    iter_num = start_iter
    if use_fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
    else:
        scaler = torch.amp.GradScaler(device="cuda")
        
    if use_sch:
        if hn_lr == min_hn_lr:
            min_hn_lr = 0.1*hn_lr
    if adam_8bit:
        optimizer = bnb.optim.AdamW8bit([{'params':hn.parameters(), 'initial_lr':hn_lr}], lr=hn_lr, weight_decay=0.05,betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.AdamW([{'params':hn.parameters(), 'initial_lr':hn_lr}], lr=hn_lr, weight_decay=0.05,betas=(0.9, 0.999))
    if hidden_kd:
        from flashlm.compression.semi_structure import hidden_state_kd_loss
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter-scheduler_start_iter, eta_min=min_hn_lr, last_epoch=iter_num-1)
    
    tic = time.time()

    with torch.no_grad():
        pesudo_x = torch.randn(1).to(device_id)
        if simple_gate:
            _ = hn()
        else:
            _ = hn(pesudo_x)
        
    if env.world_size == 1:
        if env.global_rank == 0:
            state_dict_hn = hn.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt.pt")
            torch.save(state_dict_hn, hn_path)

    else:
        if hasattr(hn, "module"):
            state_dict_hn = hn.module.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt.pt")
            torch.save(state_dict_hn, hn_path)
        else:
            if env.world_size == 1:
                # save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
                state_dict_hn = hn.module.state_dict()
            else:
                # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                state_dict_hn = hn.state_dict()

            # with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):

                # state_dict_hn = hn._orig_mod.state_dict()
                # state_dict_hn = hn.state_dict()
            if env.global_rank == 0:
                env.print_master(f"Saving checkpoint to {out_dir}")
                hn_path = os.path.join(out_dir, f"hn-ckpt.pt")
                torch.save(state_dict_hn, hn_path)
                

        
    print("################# successful")
    torch.cuda.empty_cache()
    
    # Freeze parameters of the main model
    for params in model.parameters():
        params.requires_grad = False
    # Enable training only for the hypernetwork
    for params in hn.parameters():
        params.requires_grad = True
    hn.train()
    hn_moe_ddp_flag = False
    # if isinstance(hn, torch.nn.parallel.DistributedDataParallel):
    #     if hasattr(hn.module,'model_list') and isinstance(hn.module.model_list, torch.nn.ModuleList):
    #         hn_moe_ddp_flag=True
    # elif isinstance(hn, FSDP):
    #     if hasattr(hn._fsdp_wrapped_module,'model_list'):
    #         hn_moe_ddp_flag=True
    # else:
    #     if isinstance(hn.model_list, torch.nn.ModuleList):
    #         hn_moe_ddp_flag=True
    
    env.print_master(hn_moe_ddp_flag)
    
    print("################# skip MOE")
    # env.print_master(hn)
    
    # train
    for batch in train_hn_data:
        if iter_num >= max_iter:
            break
        # with torch.no_grad():
        #     input_ids, targets = batch['input_ids'].to(device_id), batch['labels'].to(device_id)
        #     input_ids = input_ids[:,:hn_block_size]
        #     targets = targets[:,:hn_block_size]

        #     if iter_num < 5 and env.global_rank == 0:
        #         # pad_id = tokenizer.pad_token_id
        #         env.print_master(f"[DBG] pad_id={pad_id} ignored_token={ignored_token}")
        #         env.print_master(f"[DBG] input has pad: {bool((input_ids==pad_id).any().item())} | ratio: {(input_ids==pad_id).float().mean().item():.6f}")
        #         env.print_master(f"[DBG] labels has -100: {bool((targets==-100).any().item())} | ratio: {(targets==-100).float().mean().item():.6f}")
        #         env.print_master(f"[DBG] labels has pad_id: {bool((targets==pad_id).any().item())} | ratio: {(targets==pad_id).float().mean().item():.6f}")

        #     if iter_num == 0 and env.global_rank == 0:
        #         env.print_master(f"[DBG] input_ids[0,:10]={input_ids[0,:10].tolist()}")
        #         env.print_master(f"[DBG] labels   [0,:10]={targets  [0,:10].tolist()}")

        #     # attention_mask = (input_ids != ignored_token).long().to(device_id)  # ORIGINAL: used ignored_token (BOS before), not PAD
        #     attention_mask = (input_ids != pad_id).long()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device_id)
            targets   = batch["labels"].to(device_id)

            input_ids = input_ids[:, :hn_block_size]
            targets   = targets[:, :hn_block_size]

            labels = targets.clone()
            if pad_id is not None:
                labels[labels == pad_id] = -100
            labels[:, -1] = -100

            attention_mask = (input_ids != pad_id).long()


            # attention_mask = (input_ids != pad_id)

            # if 'PAD_ID' in globals() and PAD_ID is not None:
            #     attention_mask = (input_ids != PAD_ID).long().to(device_id)      # NEW: mask out only PAD positions
            # else:
            #     attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device_id)  # NEW: no PAD token -> all attend
            


            # print(input_ids.size())
            # print(targets.size())
            
        #print(input_ids.size())
        # (To be cleaned)
        # if bf_16:
        #     vectors = hn()
        #     hn_helper.set_gate_vectors(unwrap_model(model),vectors)
        #     with autocast(device_type='cuda',dtype=torch.bfloat16):
        #         logits = model(input_ids)
        #         loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)    
        # else:

        # torch.cuda.empty_cache()
        
        with autocast(device_type='cuda', dtype=data_type):
            # if semi_params:
            #     if hasattr(hn, 'module'):
            #         scales,biases = hn.module.param_forward()
            #     else:
            #         scales,biases = hn.param_forward()
            #     hn_helper.set_params_vectors(unwrap_model(model),scales,biases)
            if kd_loss or mix_loss:
                # with torch.no_grad():
                with torch.inference_mode():
                    hn_helper.set_mask_status(unwrap_model(model), False)
                    teacher_output = model(input_ids, attention_mask=attention_mask, output_hidden_states=hidden_kd)
                    # print("####### teacher output is: ", teacher_output)
                    if hasattr(teacher_output, 'logits'):
                        teacher_logits = teacher_output.logits
                    else:
                        teacher_logits = teacher_output
                    hn_helper.set_mask_status(unwrap_model(model), True)    

            if hn_moe_ddp_flag:
                pesudo_x = torch.randn(1).to(device_id)
                #        _ = hn(pesudo_x)
                # vectors, pair_loss, hard_c_out, hard_out = hn(pesudo_x)
                from torch import autocast as _autocast
                with _autocast(device_type='cuda', enabled=False):
                    vectors, pair_loss, hard_c_out, hard_out = hn(pesudo_x)
                hn_helper.set_mask_vectors(unwrap_model(model),vectors)
            else:
                pesudo_x = torch.zeros(1, device=device_id)
                from torch import autocast as _autocast
                hn_obj = hn.module if hasattr(hn, "module") else hn
                hn_obj.train()
                os.environ["SEMI_TRAIN_DETERMINISTIC"] = "1"
                os.environ["SEMI_FORCE_HARD"] = "1"

                with _autocast(device_type="cuda", enabled=False):
                    if isinstance(hn_obj, simplifed_gate):
                        vectors = hn_obj()
                    else:
                        vectors = hn_obj(pesudo_x)

                    hard_vectors = [hard_sample(v) for v in vectors]

                hard_vectors = vectors
                hn_helper.set_gate_vectors(unwrap_model(model), hard_vectors)

            # ---- GATE CONSISTENCY TEST (every 50 steps) ----
            if (iter_num % 50 == 0) and (env.global_rank == 0):
                with torch.no_grad():
                    v_flat = _flatten_to_1d(vectors, device=torch.device(f"cuda:{device_id}"))
                    h_flat = _flatten_to_1d(hard_vectors, device=torch.device(f"cuda:{device_id}"))

                    if v_flat is None or v_flat.numel() == 0:
                        env.print_master(f"[GATE] iter={iter_num} vectors is empty/unflattenable (type={type(vectors)}).")
                    else:
                        env.print_master(
                            f"[GATE] iter={iter_num} soft_mean={v_flat.float().mean().item():.6f} "
                            f"soft_std={v_flat.float().std().item():.6f} n={v_flat.numel()}"
                        )

                    if h_flat is None or h_flat.numel() == 0:
                        env.print_master(f"[GATE] iter={iter_num} hard_vectors is empty/unflattenable (type={type(hard_vectors)}).")
                    else:
                        env.print_master(
                            f"[GATE] iter={iter_num} hard_mean={h_flat.float().mean().item():.6f} "
                            f"hard_std={h_flat.float().std().item():.6f} n={h_flat.numel()}"
                        )

                    # Determinism test with fixed pseudo_x
                    pseudo_x_fixed = torch.zeros_like(pesudo_x)

                    from torch import autocast as _autocast

                    def _hard_out(hn_obj):
                        if hasattr(hn_obj, "hard_output"):
                            return hn_obj.hard_output()
                        if hasattr(hn_obj, "module") and hasattr(hn_obj.module, "hard_output"):
                            return hn_obj.module.hard_output()
                        return None

                    with _autocast(device_type="cuda", enabled=False):
                        if isinstance(hn_obj, simplifed_gate):
                            v1 = hn_obj()
                        else:
                            v1 = hn_obj(pseudo_x_fixed)
                        h1 = _hard_out(hn)

                    with _autocast(device_type="cuda", enabled=False):
                        if isinstance(hn_obj, simplifed_gate):
                            v2 = hn_obj()
                        else:
                            v2 = hn_obj(pseudo_x_fixed)
                        h2 = _hard_out(hn)

                    v1f = _flatten_to_1d(v1, device=torch.device(f"cuda:{device_id}"))
                    v2f = _flatten_to_1d(v2, device=torch.device(f"cuda:{device_id}"))
                    if v1f is not None and v2f is not None and v1f.numel() == v2f.numel():
                        vdiff = (v1f - v2f).abs()
                        env.print_master(
                            f"[GATE] iter={iter_num} soft_repeat | max_abs_diff={vdiff.max().item():.6e} "
                            f"mean_abs_diff={vdiff.mean().item():.6e}"
                        )
                    else:
                        env.print_master(f"[GATE] iter={iter_num} soft_repeat | shape mismatch or empty.")

                    h1f = _flatten_to_1d(h1, device=torch.device(f"cuda:{device_id}"))
                    h2f = _flatten_to_1d(h2, device=torch.device(f"cuda:{device_id}"))
                    if h1f is not None and h2f is not None and h1f.numel() == h2f.numel():
                        hdiff = (h1f.float() - h2f.float()).abs()
                        env.print_master(
                            f"[GATE] iter={iter_num} hard_repeat | max_abs_diff={hdiff.max().item():.6e} "
                            f"mean_abs_diff={hdiff.mean().item():.6e}"
                        )
                    else:
                        env.print_master(f"[GATE] iter={iter_num} hard_repeat | shape mismatch or empty.")
                    
                    
                    reg_soft = param_reg(vectors)
                    reg_hard = param_reg(hard_vectors)
                    env.print_master(
                        f"[REG] iter={iter_num} "
                        f"reg_soft={reg_soft.item():.6f} "
                        f"reg_hard={reg_hard.item():.6f}"
                    )
            # ---- END GATE CONSISTENCY TEST ----
            # Guard against Parameter rebind/reshape that breaks FSDP writeback mapping
            if use_fsdp and hasattr(model, "__baseline_param_meta__") and model.__baseline_param_meta__ is not None:
                _changed = _diff_param_meta(model, model.__baseline_param_meta__)
                if _changed:
                    env.print_master("[FSDP-MAP] Detected parameter mapping changes since wrap (first 8):")
                    for item in _changed[:8]:
                        if len(item) == 3:
                            nm, kind, shp = item
                            env.print_master(f"    {kind}: {nm}  current={shp}")
                        else:
                            nm, kind, cur, base = item
                            env.print_master(f"    {kind}: {nm}  current={cur}  baseline={base}")
                    raise RuntimeError("Parameter mapping changed after FSDP wrap; see [FSDP-MAP] log above for offenders.")

            # print(f"Input IDs Shape: {input_ids.shape}, Dtype: {input_ids.dtype}")
            # print(f"Attention Mask Shape: {attention_mask.shape}, Dtype: {attention_mask.dtype}")
            # # print("################# before model forward")
            # with torch.no_grad():
                    # hn_helper.set_mask_status(unwrap_model(model), False)
            model_output = model(input_ids, attention_mask=attention_mask, output_hidden_states=hidden_kd)
            # print("####### model output is: ", model_output)
            # print("################# after model forward")

            if hasattr(model_output, 'logits'):
                logits = model_output.logits
                #loss = model_output.loss
            else:
                logits = model_output
            #logits = model(input_ids)
            if kd_loss:
                labels = targets.clone()
                labels[:, -1] = -100
                #loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                # print("####### student logits is: ", logits.view(-1, logits.size(-1)))
                # print("####### teacher logits is: ", teacher_logits.view(-1, teacher_logits.size(-1)))
                
                # loss = 4 * kl_div_loss_with_ignore_index(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                # loss = 4 * kl_div_loss_with_ignore_index_softmax(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                # loss = 4 * kl_div_loss_with_ignore_index_softmax(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), labels.view(-1), ignore_index=-100)

                
                # take a try on 16 times
                # loss = 8 * kl_div_loss_with_ignore_index_softmax(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), ignore_index=-100)
                
                # loss = 1 * ForwardKLLoss(logits.view(-1, logits.size(-1)), 
                #                             teacher_logits.view(-1, teacher_logits.size(-1)), 
                #                             targets.view(-1))
                
                # top-k loss
                # T_kd = 1.0
                # k_top = 64
                # loss_kd = TopKForwardKDLoss(
                #     student_logits=logits,
                #     teacher_logits=teacher_logits,
                #     labels=labels,
                #     k=k_top,
                #     T=T_kd,
                #     ignore_index=-100,
                # )
                # loss = loss_kd

                loss = 1 * ForwardKLLoss(
                    logits.view(-1, logits.size(-1)),
                    teacher_logits.view(-1, teacher_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

                # loss = 16 * torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1), torch.nn.functional.softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1))
                # loss = 8 * torch.nn.KLDivLoss(reduction='batchmean',log_target=True)(torch.nn.functional.log_softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1), torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1))

                if hidden_kd:
                    # # print("student hidden states", model_output.hidden_states, "teacher hidden states", teacher_output.hidden_states)
                    # #  += 0.002 * Hidden State KD 
                    # hidden_kd_loss = hidden_state_kd_loss(model_output.hidden_states, teacher_output.hidden_states)
                    # loss += 2e-3 * hidden_kd_loss
                    # # loss += 1e-2 * hidden_kd_loss

                    L = model.config.num_hidden_layers
                    
                    # pick = [L//4, L//2, 3*L//4]
                    pick = [L//2]

                    hidden_kd_loss = 0.0
                    for i in pick:
                        h_t = teacher_output.hidden_states[i + 1]
                        h_s = model_output.hidden_states[i + 1]
                        hidden_kd_loss += torch.nn.functional.mse_loss(h_s, h_t)

                    hidden_kd_loss /= len(pick)
                    loss = loss + 1e-3 * hidden_kd_loss
            elif mix_loss:
                #kd_loss_value = 16 * torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1), torch.nn.functional.softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1))
                # kd_loss_value = loss = 16 * kl_div_loss_with_ignore_index(logits.view(-1, logits.size(-1)), teacher_logits.view(-1, teacher_logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
                IGNORE_TAIL = 100

                labels = targets.clone()
                labels[:, -IGNORE_TAIL:] = -100

                kd_loss_value = ForwardKLLoss(
                    logits.view(-1, logits.size(-1)),
                    teacher_logits.view(-1, teacher_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                lm_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                alpha = 0.8
                loss = alpha * kd_loss_value + (1.0 - alpha) * lm_loss
            else:
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)

            if not hn_moe_ddp_flag:
                # hard_vectors already computed above
                pass
            # print(loss)
            # for h in hard_vectors:
            #     print(h.requires_grad, h.grad_fn)

            reg_loss = param_reg(hard_vectors) # from vector to hard vector
            # reg_loss = param_reg(vectors)  # soft and forward use the same mask

            
            # reg_loss = param_reg(vectors)
            if hasattr(param_reg, 'constant_p'):
                # reg_c_loss = param_reg.constant_reg_loss(hard_c_out)
                reg_c_loss =  torch.scalar_tensor(0).to(reg_loss.get_device()).float()
                width_loss =  torch.scalar_tensor(0).to(reg_loss.get_device()).float()
            if hasattr(hn_helper, 'attn_max_reg_loss'):
                if hn_helper.attn_max_reg_loss:
                    width_loss = hn_helper.get_attn_reg_loss(unwrap_model(model), iter_num=0, targets=param_reg.width_piror, num_heads=param_reg.num_heads)
                # width_loss = hn_helper.get_self_entropy_loss(unwrap_model(model))
            # alignment_loss = param_reg.extra_alignment(vectors, hard_out)
            # print(reg_loss)
            
            loss = loss + reg_loss
            if soft_rank:
                soft_rank_loss = hn_helper.rank_reg(unwrap_model(model))
                loss = loss + soft_rank_loss
                #soft_rank_loss
            # if hn_moe_ddp_flag:
            #     loss = loss + width_loss + pair_loss + load_balance_loss + reg_c_loss

        
        if torch.isnan(loss):
            # The data may be noisy. Ignore it when loss is nan.
            env.print_master(f"!!! nan loss detected !!!")
            loss.fill_(0)

        # if bf_16:
        toc = time.time() - tic
        # print(str(toc*1000) +' ms')
        # print(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
            # model.clip_grad_norm_(grad_clip)
        # for name, param in hn.named_parameters():
        #     if param.grad is None:
        #         print(name)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # else:
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        if use_sch and iter_num > scheduler_start_iter:
            scheduler.step()
        # scheduler.step()

        toc = time.time() - tic
        tic = time.time()
        if iter_num % log_interval == 0:
            if use_sch:
                if soft_rank:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {loss.item():.4f}, reg_loss {reg_loss.item():.4f}, soft_rank {soft_rank_loss.item():.4f}, lr: {scheduler.get_last_lr()}, time: {toc*1000:.2f}msS")
                elif hn_moe_ddp_flag:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {(loss-reg_loss-pair_loss-width_loss-load_balance_loss-reg_c_loss).item():.4f}, reg_loss {reg_loss.item():.4f}, pair_loss {pair_loss.item():.4f}, width_loss {width_loss.item():.4f}, reg_c_loss {reg_c_loss.item():.4f}, balance_loss {load_balance_loss.item():.4f}, lr: {scheduler.get_last_lr()},  time: {toc*1000:.2f}msS")
                else:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {loss.item():.4f}, reg_loss {reg_loss.item():.4f}, lr: {scheduler.get_last_lr()}, time: {toc*1000:.2f}msS")
            else:
                if hn_moe_ddp_flag:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {(loss-reg_loss-pair_loss-width_loss-load_balance_loss-reg_c_loss).item():.4f}, reg_loss {reg_loss.item():.4f}, pair_loss {pair_loss.item():.4f}, width_loss {width_loss.item():.4f}, reg_c_loss {reg_c_loss.item():.4f}, balance_loss {load_balance_loss.item():.4f}, time: {toc*1000:.2f}msS")
                elif hidden_kd:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {loss.item():.4f}, reg_loss {reg_loss.item():.4f}, hidden_kd: {hidden_kd_loss.item():.4f}, time: {toc*1000:.2f}msS")
                else:
                    env.print_master(f"iter {iter_num}/{max_iter}: loss {loss.item():.4f}, reg_loss {reg_loss.item():.4f}, time: {toc*1000:.2f}msS")
                    
        iter_num += 1
        # print("################################### training successful")
        
        if iter_num % save_interval == 0:
            if env.world_size == 1:
                if env.global_rank == 0:
                    state_dict_hn = hn.state_dict()
                    env.print_master(f"Saving checkpoint to {out_dir}")
                    hn_path = os.path.join(out_dir, f"hn-ckpt-iter-{iter_num:06d}.pt")
                    torch.save(state_dict_hn, hn_path)

            else:
                if hasattr(hn, "module"):
                    state_dict_hn = hn.module.state_dict()
                    env.print_master(f"Saving checkpoint to {out_dir}")
                    hn_path = os.path.join(out_dir, f"hn-ckpt-iter-{iter_num:06d}.pt")
                    torch.save(state_dict_hn, hn_path)

                else:
                    if env.world_size == 1:
                        save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
                    else:
                        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                    with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
                        # state_dict_hn = hn._orig_mod.state_dict()
                        state_dict_hn = hn.state_dict()
                        if env.global_rank == 0:
                            env.print_master(f"Saving hn checkpoint to {out_dir}")
                            hn_path = os.path.join(out_dir, f"hn-ckpt-iter-{iter_num:06d}.pt")
                            torch.save(state_dict_hn, hn_path)
            # torch.cuda.empty_cache()
            
    if env.world_size == 1:
        if env.global_rank == 0:
            state_dict_hn = hn.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt-final.pt")
            torch.save(state_dict_hn, hn_path)
    else:
        if hasattr(hn, "module"):
            state_dict_hn = hn.module.state_dict()
            env.print_master(f"Saving checkpoint to {out_dir}")
            hn_path = os.path.join(out_dir, f"hn-ckpt-final.pt")
            torch.save(state_dict_hn, hn_path)

        else:
            if env.world_size == 1:
                save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
            else:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            with FSDP.state_dict_type(hn, StateDictType.FULL_STATE_DICT, save_policy):
                # state_dict_hn = hn._orig_mod.state_dict()
                state_dict_hn = hn.state_dict()
                if env.global_rank == 0:
                    env.print_master(f"Saving hn checkpoint to {out_dir}")
                    hn_path = os.path.join(out_dir, f"hn-ckpt-final.pt")
                    torch.save(state_dict_hn, hn_path)
    
    
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from jsonargparse import CLI
    CLI(main)