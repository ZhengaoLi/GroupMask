import torch
import torch.nn as nn
import torch.nn.functional as F
# from misc_functions import custom_grad_weight
import numpy as np
import math
import os

# flashlm/compression/semi_structure/hypernetwork.py
def safe_repeat_interleave(tensor, repeat_factor, dim=0, max_chunk_size=8192):
    """Memory-safe repeat_interleave along a given dim."""
    if dim < 0:
        dim += tensor.dim()
    try:
        return tensor.repeat_interleave(repeat_factor, dim=dim).contiguous()
    except Exception:
        pass
    size_along = tensor.size(dim)
    chunk = max(1, min(size_along, max_chunk_size))
    chunks = torch.split(tensor, chunk, dim=dim)
    return torch.cat([c.repeat_interleave(repeat_factor, dim=dim) for c in chunks], dim=dim)

# def safe_repeat_interleave(tensor, repeat_factor, max_chunk_size=8192):
#     # Fast path when the tensor is already small
#     if tensor.numel() <= max_chunk_size:
#         return tensor.repeat_interleave(repeat_factor).contiguous()
#     tensor = tensor.contiguous()
#     chunks = torch.split(tensor, max_chunk_size)
#     return torch.cat([c.repeat_interleave(repeat_factor) for c in chunks], dim=0)

def sample_gumbel(shape,eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U+eps)+eps)

def hard_sample(out):
    binary_out = torch.round(out)
    binary_out = (binary_out - out).detach() + out
    return binary_out

def gumbel_sigmoid_sample(logits, T, offset=0):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.get_device() == -1:
        logits = logits.cpu()
        gumbel_sample = gumbel_sample.cpu()
    else:
        gumbel_sample = gumbel_sample.to(logits.get_device())

    y = logits + gumbel_sample + offset

    return F.sigmoid(y/T)

def reinmax_simple(logits: torch.Tensor, T: float, offset=0, hard_sample=False):
    logits = logits + offset
    pi_0 = torch.sigmoid(logits)
    b = torch.bernoulli(pi_0)
    pi_1 = (b + torch.sigmoid((logits)/T))/2
    # print(pi_1)
    pi_1 = torch.sigmoid((torch.log(pi_1)-logits).detach() + logits)
    # print(pi_0)
    #pi_2 = 2*pi_1 - 0.5*pi_0
    pi_2 = (2*pi_1 - 0.5*pi_0) 
    if not hard_sample:
        return pi_2 + 0.5*pi_0.detach()
    else:
        return pi_2 - pi_2.detach() + b, pi_2 + 0.5*pi_0.detach()

class BottleneckLinear(nn.Module):
    def __init__(self, in_dim : int, out_dim : int, groups : int, bias=False):
        super(BottleneckLinear, self).__init__()
        self.in_dim = in_dim
        #self.weight = nn.Parameter(torch.empty(groups, in_dim//groups, out_dim//groups))
        self.groups = groups
        self.linear_A = nn.Linear(in_dim, in_dim//self.groups)
        self.linear_B = nn.Linear(in_dim//self.groups, out_dim)
        self.ln = nn.LayerNorm([in_dim//self.groups])

    def forward(self, x):
        out = self.linear_A(x)
        out = self.ln(out)
        out = F.gelu(out)
        out = self.linear_B(out)
        return out

class GroupLinear(nn.Module):
    def __init__(self, in_dim : int, out_dim : int, groups : int, bias=False):
        super(GroupLinear, self).__init__()
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.empty(groups, in_dim//groups, out_dim//groups))
        self.groups = groups

        self.bias = None
        if bias:
            raise NotImplementedError
        # nn.init.uniform_(self.group_weight, a=-1/math.sqrt(in_dim*groups), b=1/math.sqrt(in_dim*groups))
        self.reset_parameters()
        
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for i in range(self.groups):
            torch.nn.init.kaiming_uniform_(self.weight[i,:], a=math.sqrt(5))

    def forward(self, x):
        BN,D = x.size()
        x = x.reshape(-1, self.groups, self.in_dim//self.groups)
        x = torch.matmul(x.transpose(0,1), self.weight).transpose(0,1)
        
        return x.reshape(BN,-1)

class virtual_operation(nn.Module):
    def __init__(self, dim,ex_dict={},):
        super().__init__()
        self.dim = dim
        self.pruning_vector = torch.ones(dim)
        self.ex_dict = ex_dict
        # self.generation = generation
        # self.seed = seed
        self.mask = None    
        self.sv = None
        self.scale = None
        self.bias = None
        self.rank = None
        #self.bias = None
    # def forward(self, input):
    #     assert len(input.size())==3
    #     # if len(input.size())==3:
    #     p_v = self.pruning_vector[:,None,None]

    #     if input.get_device() == -1:
    #         p_v = p_v.cpu()
    #     else:
    #         p_v = p_v.to(input.get_device())
    #     input = p_v.expand_as(input) * input

    #     return input
        self.expand_rate = 1
        self.wo_repeat = False
        
        self._debug_printed = False
        self._cached_mask: Optional[torch.Tensor] = None
        
    def clear_cache(self):
        self._cached_mask = None
        
    def rank_reg(self):
        hard_pv = torch.round(self.pruning_vector)
        target_vector = torch.zeros(self.dim)
        target_vector[self.sv[:int(hard_pv.sum())]] = 1
        device = self.pruning_vector.get_device()
        if device is not None:
            target_vector = target_vector.to(device)
        elif device == -1:
            target_vector = target_vector.cpu()
        
        source_vector = (hard_pv - self.pruning_vector).detach() + self.pruning_vector
        return torch.nn.functional.mse_loss(source_vector, target_vector)
    
    def generate_pv_repeat(self, seed=0, p=0.5):
        expand_rate = self.expand_rate
        import numpy as np
        if expand_rate > 1:
            assert expand_rate % 2 == 0
        mask = []
        if expand_rate == 1:
            num_repeat = 1
        else:    
            num_repeat = expand_rate//2

        for i in range(num_repeat):
            np.random.seed(seed+i*1000)
            pruning_vector = np.random.binomial(1, p, size=self.dim)
            pruning_vector = torch.from_numpy(pruning_vector)
            self.pruning_vector.copy_(pruning_vector)
            with torch.no_grad():
                mask.append(self.forward().to(torch.uint8))
        if len(mask) == 1:
            self.mask = mask[0]
        else:
            self.mask = mask
    def generate_pv_norepeat(self, seed=0, p=0.5):
        expand_rate = self.expand_rate
        if expand_rate == 1:
            self.generate_pv_repeat(seed=seed, p=p)
        else:
            import numpy as np
            np.random.seed(seed)
            permutated_vector = np.random.permutation(self.dim)
            per_gap = int(self.dim/expand_rate)
            index_list = []
            for i in range(self.expand_rate-1):
                index_list.append(torch.from_numpy(permutated_vector[i*per_gap:(i+1)*per_gap]).long())
            index_list.append(torch.from_numpy(permutated_vector[(self.expand_rate-1)*per_gap:]).long())
            mask = []
            for i in range(self.expand_rate):
                pruning_vector = torch.zeros(self.dim)
                pruning_vector[index_list[i]] = 1
                self.pruning_vector.copy_(pruning_vector)
                with torch.no_grad():
                    mask.append(self.forward().to(torch.uint8))
            self.mask = mask

    def generate_pv(self, seed=0, p=0.5):
        if self.wo_repeat:
            self.generate_pv_norepeat(seed=seed, p=p)
        else:
            self.generate_pv_repeat(seed=seed, p=p)
                #self.mask = self.forward().to(torch.uint8)
        # [MEM] Do not eagerly materialize a permanent full-weight-sized GPU
        # mask here. self.mask (used by SemiSparseLinear at actual forward
        # time) is already set above; caching another full copy in
        # self._cached_mask pins ~1 extra byte/param on every GPU rank for
        # the life of the process, before FSDP ever gets a chance to shard
        # anything. forward() below recomputes on demand instead.
    
    def forward_bias(self, device=None):
        if device is not None and self.bias is not None:
            self.bias = self.bias.to(device)
        b = self.bias
        if self.ex_dict['groups_in_dim'] == 1 or self.ex_dict['groups_out_dim'] ==1:
            b = b.repeat_interleave(self.ex_dict['groups_in_dim']*self.ex_dict['groups_out_dim'])
            bias = b.view(self.ex_dict['out_dim'], self.ex_dict['in_dim'])
        else:
            #p_v = p_v.unsqueeze(-1)
            b = b.view(self.ex_dict['groups_out'], self.ex_dict['groups_in'])
            b = b.repeat_interleave(self.ex_dict['groups_out_dim'], dim=0)
            b = b.repeat_interleave(self.ex_dict['groups_in_dim'], dim=1)
            # p_v = p_v.expand(-1, self.ex_dict['groups_in_dim']*self.ex_dict['groups_out_dim'])
            # p_v = p_v.view(-1,  self.ex_dict['groups_in_dim'], self.ex_dict['groups_out_dim'])
            # p_v = torch.cat(tuple(p_v), dim=1)
            # mask = torch.cat(torch.chunk(p_v, self.ex_dict['groups_out'], dim=1), dim=0)
            bias = b
        return bias

    def forward(self,device=None):
        # print(f"\n[DEBUG - virtual_operation FORWARD CALL] Called {self.__class__.__name__}")
        # traceback.print_stack(limit=10)

        # if not self._debug_printed:
        #     print(f"[DEBUG] Memory allocated: {torch.cuda.memory_allocated()/1024/1024:.2f} MiB")
        #     print(f"[DEBUG] Memory reserved:  {torch.cuda.memory_reserved()/1024/1024:.2f} MiB")
        #     self._debug_printed = True
            
        p_v = self.pruning_vector
        if device is not None:
            p_v = p_v.to(device)
            if self.scale is not None:
                self.scale = self.scale.to(device)
        else:
            p_v = p_v.cpu()
            if self.scale is not None:
                self.scale = self.scale.cpu()

        if self.scale is not None:
            p_v = p_v*self.scale

        max_chunk_size = 8192
        if self.ex_dict['groups_in_dim'] == 1 or self.ex_dict['groups_out_dim'] ==1:
            # p_v = p_v.repeat_interleave(self.ex_dict['groups_in_dim']*self.ex_dict['groups_out_dim'])
            p_v = safe_repeat_interleave(p_v, self.ex_dict['groups_in_dim'] * self.ex_dict['groups_out_dim'], dim=0)
            mask = p_v.view(self.ex_dict['out_dim'], self.ex_dict['in_dim'])
        else:
            #p_v = p_v.unsqueeze(-1)
            p_v = p_v.view(self.ex_dict['groups_out'], self.ex_dict['groups_in'])
            # p_v = p_v.repeat_interleave(self.ex_dict['groups_out_dim'], dim=0)
            # p_v = p_v.repeat_interleave(self.ex_dict['groups_in_dim'], dim=1)
            
            # p_v = safe_repeat_interleave(p_v, self.ex_dict['groups_out_dim'], max_chunk_size)
            # p_v = safe_repeat_interleave(p_v, self.ex_dict['groups_in_dim'], max_chunk_size)
            
            p_v = safe_repeat_interleave(p_v, self.ex_dict['groups_out_dim'], dim=0)
            p_v = safe_repeat_interleave(p_v, self.ex_dict['groups_in_dim'], dim=1)
            
            # p_v = p_v.expand(-1, self.ex_dict['groups_in_dim']*self.ex_dict['groups_out_dim'])
            # p_v = p_v.view(-1,  self.ex_dict['groups_in_dim'], self.ex_dict['groups_out_dim'])
            # p_v = torch.cat(tuple(p_v), dim=1)
            # mask = torch.cat(torch.chunk(p_v, self.ex_dict['groups_out'], dim=1), dim=0)
            mask = p_v

        return mask

    def get_parameters(self):
        return self.ex_dict['in_dim'] * self.ex_dict['out_dim']

    def set_vector_value(self, value):
        self.pruning_vector = torch.ones(self.dim)
        assert value.size() == self.pruning_vector.size()
        if value is not None:
            self.pruning_vector = value.squeeze()
        else:
            self.pruning_vector = value
    def set_scale_bias(self, scale, bias=None):
        self.scale = scale.squeeze()
        if bias is not None:
            self.bias = bias.squeeze()

class simplifed_gate(nn.Module):
    def __init__(self, t_structures, num_groups=1, reinmax=False):
        super(simplifed_gate, self).__init__()
        self.T = 0.4
        self.base = float(os.environ.get('SIMPLE_GATE_BASE', '3.0'))
        self.det_eval = os.environ.get('SIMPLE_GATE_DET_EVAL', '0') == '1'
        self.t_sp = t_structures

        # optional init shift for the w/o-HN ablation (SIMPLE_GATE_INIT_SHIFT, default 0 = original)
        _shift = float(os.environ.get('SIMPLE_GATE_INIT_SHIFT', '0'))
        self.p_list = nn.ParameterList([nn.Parameter(torch.randn(t_structures[i]) + _shift) for i in range(len(t_structures))])
        self.groups = num_groups
        
        if reinmax:
            self.approxiate_fucntion = reinmax_simple
            self.T = 1
            self.reinmax = True
        else:
            self.approxiate_fucntion = gumbel_sigmoid_sample
            self.reinmax = False
    def _det_scores(self):
        # SIMPLE_GATE_TOPK=<keep ratio> -> deterministic hard mask that keeps the top-k learned logits
        # globally (all groups have the same size, so count == params); kept gates are exactly 1 (no soft scaling).
        k = os.environ.get("SIMPLE_GATE_TOPK")
        if k:
            allp = torch.cat([q.detach().flatten().float() for q in self.p_list])
            n_keep = int(round(float(k) * allp.numel()))
            thr = torch.topk(allp, n_keep).values.min()
            return [(self.p_list[i].detach().float() >= thr).float().squeeze() for i in range(len(self.t_sp))]
        return [torch.sigmoid((self.p_list[i] + self.base) / self.T).squeeze() for i in range(len(self.t_sp))]

    def forward(self,):
        if (not self.training) and self.det_eval:
            return self._det_scores()
        # SIMPLE_GATE_FOLLOW_SEMI=1 makes the w/o-HN gate follow the SAME training protocol as the
        # hypernetwork (the recipe sets SEMI_TRAIN_DETERMINISTIC=1 / SEMI_FORCE_HARD=1 before every gate call):
        # deterministic sigmoid((p+base)/T), then hard 0/1 with a straight-through estimator. Default: unchanged.
        if self.training and os.environ.get("SIMPLE_GATE_FOLLOW_SEMI", "0") == "1":
            scores = [torch.sigmoid((self.p_list[i] + self.base) / self.T).squeeze() for i in range(len(self.t_sp))]
            if os.environ.get("SEMI_TRAIN_DETERMINISTIC", "0") != "1":
                scores = [self.approxiate_fucntion(self.p_list[i], offset=self.base, T=self.T).squeeze() for i in range(len(self.t_sp))]
            if os.environ.get("SEMI_FORCE_HARD", "0") != "0":
                scores = [hard_sample(s_) for s_ in scores]
            return scores
        if self.reinmax:
            tp_out = [self.approxiate_fucntion(self.p_list[i], offset=self.base, T=self.T, hard_sample=True)[0].squeeze() for i in
                    range(len(self.t_sp))]
        else:
            tp_out = [self.approxiate_fucntion(self.p_list[i], offset=self.base, T=self.T).squeeze() for i in
                    range(len(self.t_sp))]
        if not self.training:
            if not self.reinmax:
                tp_out = [hard_sample(tp_out[i]) for i in range(len(self.t_sp))]
        
        return tp_out
    
    def hard_output(self):
        if (not self.training) and self.det_eval:
            return [hard_sample(s_) for s_ in self._det_scores()]
        if self.reinmax:
            tp_out = [self.approxiate_fucntion(self.p_list[i], offset=self.base, T=self.T, hard_sample=True)[0].squeeze() for i in
                    range(len(self.t_sp))]
        else:
            tp_out = [self.approxiate_fucntion(self.p_list[i], offset=self.base, T=self.T).squeeze() for i in
                    range(len(self.t_sp))]
            tp_out = [hard_sample(tp_out[i]) for i in range(len(self.t_sp))]

        return tp_out

class hypernetwork(nn.Module):
    def __init__(self, t_structures, num_groups=1, reinmax=False, hard_flag=False, param_flag=False):
        super(hypernetwork, self).__init__()
        self.T = 0.4
        self.base = 3.0
        # self.group_size = group_size

        # assert type(self.group_size) == int
        # assert self.group_size >= 1

        self.t_sp = t_structures
        # self.h0 = torch.zeros(2,1,64)
        self.register_buffer("h0", torch.zeros(2, 1, 64), persistent=False)

        self.bi_GRU = nn.GRU(32, 64, bidirectional=True)
        #self.inputs = nn.Parameter(torch.Tensor(len(t_structures),1,32))
        #nn.init.normal_(self.inputs)
        #self.inputs.requires_grad=False
        inputs = torch.normal(mean=0.0, std=0.02, size=(len(t_structures),1,32))
        self.register_buffer("inputs", inputs)

        self.param_flag = param_flag
        if param_flag:
        # #     self.bias_list = nn.ParameterList([nn.Parameter(torch.zeros(t_structures[i])) for i in range(len(t_structures))])
            self.scale_list = nn.ParameterList([nn.Parameter(self.base*torch.ones(t_structures[i])) for i in range(len(t_structures))])
        else:
            self.scale_list = None

        if num_groups == 1:
            self.linear_list_tp = [nn.Linear(128, int(self.t_sp[i]), bias=False) for i in range(len(self.t_sp))]
        else:
            #self.linear_list_tp = [GroupLinear(128, int(self.t_sp[i]), num_groups)  for i in range(len(self.t_sp))]
            self.linear_list_tp = [BottleneckLinear(128, int(self.t_sp[i]), num_groups) for i in range(len(self.t_sp))]
        self.groups = num_groups

        self.linear_list_tp = nn.ModuleList(self.linear_list_tp)

        # self.ln_in = nn.LayerNorm([128])
        self.ln_tp = nn.LayerNorm([128])
        self.hard_flag = hard_flag
        # self.head_size = self.t_sp[0].size(0)
        if reinmax:
            self.approxiate_fucntion = reinmax_simple
            self.T = 1
            self.reinmax = True
        else:
            self.approxiate_fucntion = gumbel_sigmoid_sample
            self.reinmax = False
    
    def param_forward(self,):
        return self.scale_list, self.bias_list

    def forward(self, x):
        if self.ln_tp.weight.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.ln_tp.weight.get_device())
        # print(self.inputs.get_device())
        # print(self.h0.get_device())
        # if inputs is not None:
            
        # # New
        # # dtype = self.inputs.dtype
        # # self.h0 = self.h0.to(self.inputs.device).to(dtype)
        # self.h0 = self.h0.to(torch.float16)
        # self.inputs = self.inputs.to(torch.float16)
        # print(f"################## inputs dtype: {self.inputs.dtype}, device: {self.inputs.device}, is_contiguous: {self.inputs.is_contiguous()}")
        # print(f"################## h0 dtype: {self.h0.dtype}, device: {self.h0.device}, is_contiguous: {self.h0.is_contiguous()}")
        h0 = self.h0.to(self.inputs.dtype) 
        outputs, hn = self.bi_GRU(self.inputs, self.h0)
        # else:
        #     outputs, hn = self.bi_GRU(inputs, self.h0)

        tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]
        tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]
        
        if not self.training:
            if self.reinmax:
                if self.param_flag:
                    tp_out_approx = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T, hard_sample=True) for i in
                    range(len(self.linear_list_tp))]
                    soft_tp_out = [tp_out_approx[i][1] for i in range(len(self.linear_list_tp))]
                    #tp_out = [tp_out_approx[i][0].squeeze()*tp_out_approx[i][1].squeeze() for i in range(len(self.linear_list_tp))]
                    tp_out = [tp_out_approx[i][0].squeeze()*F.sigmoid(self.scale_list[i]).squeeze() for i in range(len(self.linear_list_tp))]
                else:
                    pre_tp_out = tp_out
                    tp_out = [self.approxiate_fucntion(pre_tp_out[i], offset=self.base, T=self.T, hard_sample=True)[0].squeeze() for i in
                        range(len(self.linear_list_tp))]
                    soft_tp_out = [self.approxiate_fucntion(pre_tp_out[i], offset=self.base, T=self.T, hard_sample=True)[1].squeeze() for i in
                        range(len(self.linear_list_tp))]
            else:
                # Deterministic gating during evaluation by default (no Gumbel noise).
                # Set SEMI_DETERMINISTIC=0 to revert to stochastic gumbel-sigmoid at eval.
                deterministic = os.environ.get("SEMI_DETERMINISTIC", "1") != "0"
                if deterministic:
                    pre_tp_out = tp_out  # scores before nonlinearity
                    tp_scores = [torch.sigmoid((pre_tp_out[i] + self.base) / self.T).squeeze() for i in range(len(self.linear_list_tp))]
                else:
                    tp_scores = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T).squeeze() for i in range(len(self.linear_list_tp))]
                soft_tp_out = tp_scores
                tp_out = [hard_sample(tp_scores[i]) for i in range(len(self.linear_list_tp))]
            for i in range(len(tp_out)):
                if tp_out[i].sum()==0:
                    max_ind = soft_tp_out[i].argmax()
                    tp_out[i][max_ind] = 1

        else:
            # if self.reinmax:
            #     tp_out = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T, hard_sample=True)[0].squeeze() for i in
            #         range(len(self.linear_list_tp))]
            # else:
            # tp_out = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T).squeeze() for i in
            #         range(len(self.linear_list_tp))]
            if self.reinmax:
                if self.param_flag:
                    tp_out_approx = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T, hard_sample=True) for i in
                    range(len(self.linear_list_tp))]
                    #tp_out = [tp_out_approx[i][0].squeeze()*tp_out_approx[i][1].squeeze() for i in range(len(self.linear_list_tp))]
                    tp_out = [tp_out_approx[i][0].squeeze()*F.sigmoid(self.scale_list[i]).squeeze() for i in range(len(self.linear_list_tp))]
                else:
                    tp_out = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T, hard_sample=True)[0].squeeze() for i in
                    range(len(self.linear_list_tp))]
            else:
                # # (disabled)
                # tp_out = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T).squeeze() for i in
                #     range(len(self.linear_list_tp))]
                # if self.hard_flag:
                #     tp_out = [hard_sample(tp_out[i]) for i in range(len(self.linear_list_tp))]
                ## deterministic hard gate
                det_train = os.environ.get("SEMI_TRAIN_DETERMINISTIC", "0") != "0"
                force_hard = os.environ.get("SEMI_FORCE_HARD", "0") != "0"

                if det_train:
                    # deterministic sigmoid, NO gumbel noise (same formula as your eval deterministic path)
                    tp_scores = [torch.sigmoid((tp_out[i] + self.base) / self.T).squeeze()
                                for i in range(len(self.linear_list_tp))]
                else:
                    # stochastic (gumbel) as before
                    tp_scores = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T).squeeze()
                                for i in range(len(self.linear_list_tp))]

                if self.hard_flag or force_hard:
                    tp_scores = [hard_sample(tp_scores[i]) for i in range(len(self.linear_list_tp))]

                tp_out = tp_scores
        # print("tpout0:", tp_out[0].mean())
        return tp_out

    def hard_output(self):
        if self.ln_tp.weight.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.ln_tp.weight.get_device())
        
        dtype = self.inputs.dtype
        self.h0 = self.h0.to(self.inputs.device).to(dtype)

        outputs, hn = self.bi_GRU(self.inputs, self.h0)

        tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]

        tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]

        if self.reinmax:
            tp_out = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T, hard_sample=True)[0].squeeze() for i in range(len(self.linear_list_tp))]
        else:
            deterministic = os.environ.get("SEMI_DETERMINISTIC", "1") != "0"
            if deterministic:
                tp_scores = [torch.sigmoid((tp_out[i] + self.base) / self.T).squeeze() for i in range(len(self.linear_list_tp))]
            else:
                tp_scores = [self.approxiate_fucntion(tp_out[i], offset=self.base, T=self.T).squeeze() for i in range(len(self.linear_list_tp))]
            tp_out = [hard_sample(tp_scores[i]) for i in range(len(self.linear_list_tp))]

        return tp_out