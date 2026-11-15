import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

import numpy as np
import torch.nn as nn
import os
from tqdm.auto import tqdm
import time
from torch.cuda.amp import GradScaler 
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from .hypernetwork import virtual_operation
from torch.nn import Module

def clear_cache_hook(module: Module, *args):
    from .hypernetwork import virtual_operation
    # Do not clear cache in eval mode; only clear during training to avoid recomputation on every forward
    if not module.training:
        return
    if isinstance(module, virtual_operation):
        module.clear_cache()

def register_virtual_cache_hooks(model: Module):
    for m in model.modules():
        from .hypernetwork import virtual_operation
        if isinstance(m, virtual_operation):
            m.register_forward_pre_hook(clear_cache_hook)

def log_inv_function(sum_params, sum_ori_params, p):

    param_ratio = sum_params / (sum_ori_params)

    if param_ratio>p:
        clampled_p_ratio = torch.clamp(param_ratio, min=p)
        loss = torch.log(clampled_p_ratio/p)
    else:
        clampled_p_ratio = torch.clamp(param_ratio, max=p)
        loss = torch.log(p/clampled_p_ratio)
    return loss

class collect_info_reg(nn.Module):
    def __init__(self, model, p=None, lam=4.0):
        super(collect_info_reg, self).__init__()
        self.sum_ori_params = 0
        self.p = p
        self.in_dim_list = []
        self.out_dim_list = []
        self.in_group_list = []
        self.out_group_list = []
        self.structures = []
        self.lam = lam
        self.expand_rate = 1
        self.mlp_only = False
        #self.rescale_factor = 1
        basic_flag = False
        # list.insert(0, "The")
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            # print(type(m))
            # if isinstance(m, virtual_share_operation):
            if type(m).__name__ == 'virtual_operation':
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['in_dim'])
                self.out_dim_list.append(m.ex_dict['out_dim'])
                self.in_group_list.append(m.ex_dict['groups_in'])
                self.out_group_list.append(m.ex_dict['groups_out'])
                self.structures.append(m.dim)
        print("number of oringal parameters: %.3f" % (self.sum_ori_params / 10 ** 6))

    def count_current_params(self, vectors):
        with torch.no_grad():
            sum_params = 0
            model_dim = vectors[0].sum().item()
            for i in range(len(self.structures)-1):
                groups_rate = model_dim/(self.in_group_list[i]*self.out_group_list[i])
                current_params = groups_rate*(self.in_dim_list[i]*self.out_dim_list[i])
                sum_params += current_params
        print("current parameters: %.3f" % (sum_params / 10 ** 6))
        return sum_params

    def forward(self, vectors):
        sum_params = 0
        for i in range(len(self.structures)):
            model_dim = vectors[i].sum()
            groups_rate = model_dim/(self.in_group_list[i]*self.out_group_list[i])
            current_params = groups_rate*(self.in_dim_list[i]*self.out_dim_list[i])
            sum_params += current_params
        param_ratio = sum_params / (self.sum_ori_params)
        if param_ratio>self.p:
            clampled_p_ratio = torch.clamp(param_ratio, min=self.p)
            loss = torch.log(clampled_p_ratio/self.p)
        else:
            clampled_p_ratio = torch.clamp(param_ratio, max=self.p)

            loss = torch.log(self.p/clampled_p_ratio)
        
        # print("current parameters: %.3f" % (sum_params / 10 ** 6))
        # loss = custom_grad_weight.apply(loss, self.grad_w)

        return self.lam * loss


class collect_info_share(nn.Module):
    """Share-mode info collector.

    Mirrors your collect_info_reg() logic but *deduplicates* Q/K pairs so that
    structures/vectors are counted once per pair.
    """
    def __init__(self, model: nn.Module, p: Optional[float] = None, lam: float = 4.0):
        super().__init__()
        self.p = p
        self.lam = lam
        self.sum_ori_params = 0
        self.structures: List[int] = []
        self.in_dim_list: List[int] = []
        self.out_dim_list: List[int] = []
        self.in_group_list: List[int] = []
        self.out_group_list: List[int] = []

        seen_pairs: set = set()
        modules = list(model.modules())
        for m in modules:
            # recognize virtual_operation attached to SemiSparseShareLinear or SemiSparseLinear
            if type(m).__name__ == 'virtual_operation':  # same check as your code
                # Try to reach its owner flags (walk up: parent module sets attrs onto the virtual op in our replace)
                # We store the pairing id *on the virtual operation* to simplify discovery here.
                pair_id = getattr(m, 'share_pair_id', None)
                if pair_id is not None:
                    if pair_id in seen_pairs:
                        continue  # already counted this Q/K pair
                    seen_pairs.add(pair_id)

                # accumulate once (either non-shared or first time of a shared pair)
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['in_dim'])
                self.out_dim_list.append(m.ex_dict['out_dim'])
                self.in_group_list.append(m.ex_dict['groups_in'])
                self.out_group_list.append(m.ex_dict['groups_out'])
                self.structures.append(m.dim)

        print("[share] number of original parameters: %.3f" % (self.sum_ori_params / 1e6))

    def count_current_params(self, vectors: List[torch.Tensor]):
        with torch.no_grad():
            sum_params = 0.0
            model_dim = vectors[0].sum().item()
            for i in range(len(self.structures) - 1):
                groups_rate = model_dim / (self.in_group_list[i] * self.out_group_list[i])
                current_params = groups_rate * (self.in_dim_list[i] * self.out_dim_list[i])
                sum_params += current_params
        print("[share] current parameters: %.3f" % (sum_params / 1e6))
        return sum_params

    def forward(self, vectors: List[torch.Tensor]):
        sum_params = 0.0
        for i in range(len(self.structures)):
            model_dim = vectors[i].sum()
            groups_rate = model_dim / (self.in_group_list[i] * self.out_group_list[i])
            current_params = groups_rate * (self.in_dim_list[i] * self.out_dim_list[i])
            sum_params += current_params
        param_ratio = sum_params / (self.sum_ori_params)
        if self.p is None:
            # if no p target, return zero to be no-op
            return 0.0 * param_ratio
        if param_ratio > self.p:
            clamped = torch.clamp(param_ratio, min=self.p)
            loss = torch.log(clamped / self.p)
        else:
            clamped = torch.clamp(param_ratio, max=self.p)
            loss = torch.log(self.p / clamped)
        return self.lam * loss


# class SemiSparseMLP(torch.nn.Module):

class AdapterforExpand(torch.nn.Module):
    def __init__(self, in_dim : int, out_dim: int, r:int=32, expand_rate:int=4, lora_alpha:float=1, setting='kron'):
        super(AdapterforExpand, self).__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(r).float())
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.setting = setting
        #assert expand_rate == 4
        if self.setting == 'kron':
            r_out = r_in = r
            #self.lora_A = nn.Parameter(torch.randn(r, in_dim) * std_dev)
            #self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
            while out_dim % r_out != 0:
                r_out = r_out/2
            while in_dim % r_in != 0:
                r_in = r_in/2
            r_out = int(r_out)
            r_in = int(r_in)
            self.lora_A = nn.Parameter(torch.randn(r_out, r_in) * std_dev)
            self.lora_B = nn.Parameter(torch.zeros(out_dim // r_out, in_dim // r_in))
        elif self.setting == 'lora':
            self.lora_A = nn.Parameter(torch.randn(r, r) * std_dev)
            self.lora_B = nn.Parameter(torch.zeros(out_dim // r, in_dim // r))
        self.scaling = lora_alpha / r
        
        #self.lora_M = nn.Parameter(torch.ones(1, int(out_dim*expand_rate)))
    
    def forward(self, x, mask, expand_dim):
        #return torch.mm(self.lora_B, self.lora_A)*self.scaling
        #return torch.kron(self.lora_A, self.lora_B)*self.scaling
        if self.setting == 'kron':
            weight = torch.kron(self.lora_A, self.lora_B)*self.scaling
        elif self.setting == 'lora':
            weight = torch.mm(self.lora_B, self.lora_A)*self.scaling

        masked_weight = mask*weight
        ex_masked_weight = (1-mask)*weight
        
        cat_weight = torch.cat((masked_weight, ex_masked_weight), dim=expand_dim)
        
        return torch.nn.functional.linear(x, cat_weight)

class AdapterforExpandMean(torch.nn.Module):
    def __init__(self, out_dim: int, expand_rate:int=4, expand_dim=0):
        super(AdapterforExpandMean, self).__init__()
        if expand_dim == 0:
            self.lora_M = nn.Parameter(torch.zeros(int(out_dim*expand_rate)))
        else:
            self.lora_M = nn.Parameter(torch.zeros(int(out_dim)))
    def forward(self, input):
        output = self.lora_M[None,None,:]*(input/(input.norm(p=2, dim=1, keepdim=True) + 1e-9))
        return output

class SemiSparseLinear(torch.nn.Module):
    # def __init__(self, weights, rank, bias=None):
    def __init__(self, in_dim : int, out_dim : int, groups_in_dim : int, groups_out_dim : int, expand_rate:int=1,expand_dim:int=0, bias=False, wo_repeat:bool=False, adapter:bool=False):
        super(SemiSparseLinear, self).__init__()
        #self.linear = nn.Linear(in_dim, out_dim, bias=False)
        if in_dim % groups_in_dim !=0 or out_dim % groups_out_dim != 0 or in_dim < groups_in_dim or out_dim < groups_out_dim:
            temp = groups_out_dim
            groups_out_dim = groups_in_dim
            groups_in_dim = temp           
        # print(groups_in_dim)
        # print(groups_out_dim)
        self.groups_in = int(in_dim/groups_in_dim)
        self.groups_out = int(out_dim/groups_out_dim)
        self.groups_in_dim = groups_in_dim
        self.groups_out_dim = groups_out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        # y_block = y.view(-1, groups_in_dim, groups_out_dim)
        # y_row = torch.cat(tuple(y_block), dim=1)
        # torch.cat(torch.chunk(y_row, self.groups_in, dim=1), dim=0)
        ex_dict = {}
        ex_dict['in_dim'] = in_dim
        ex_dict['out_dim'] = out_dim
        ex_dict['groups_in'] = self.groups_in
        ex_dict['groups_out'] = self.groups_out
        ex_dict['groups_in_dim'] = groups_in_dim
        ex_dict['groups_out_dim'] = groups_out_dim

        self.expand_rate = expand_rate
        self.wo_repeat = wo_repeat
        # print('in/g_in', in_dim/groups_in_dim)
        # print('out/g_out', out_dim/groups_out_dim)
        # print('g_in', groups_in_dim)
        # print('g_out', groups_out_dim)
        # print(self.groups_in)
        # print(self.groups_out)
        # print(int(self.groups_in*self.groups_out))
        self.virtual_operation = virtual_operation(dim = int(self.groups_in*self.groups_out), ex_dict=ex_dict)
        self.virtual_operation.expand_rate = expand_rate
        self.virtual_operation.wo_repeat = wo_repeat

        self.gate_flag = False
        self.mask_flag = True
        self.ex_dict = ex_dict
        
        self.expand_dim = expand_dim
        self.scale_weight = False
        
        self.adapter = adapter
        # if self.adapter:
        #     rank = self.virtual_operation.rank
        #     self.adapter_modules = nn.ModuleList([AdapterforExpand(in_dim, out_dim, rank, expand_rate) for i in range(expand_rate//2)])
        #     self.adapter_mean = AdapterforExpandMean(out_dim, expand_rate, expand_dim)
        #self.weight = self.linear.weight
        #self.mask = None
    def init_adpaters(self):
        if self.expand_rate > 1:
            rank = self.virtual_operation.rank
            self.adapter_modules = nn.ModuleList([AdapterforExpand(self.ex_dict['in_dim'], self.ex_dict['out_dim'], rank, self.expand_rate) for i in range(self.expand_rate//2)])
            #self.adapter_mean = AdapterforExpandMean(self.ex_dict['out_dim'], self.expand_rate, self.expand_dim)

    def sort_weight(self):
        weight_clone = self.linear.weight.data.clone()
        w_flat_abs = weight_clone.abs().flatten()
        assert self.ex_dict['groups_out_dim'] == 1
        w_sum = w_flat_abs.reshape(w_flat_abs.numel() // self.ex_dict['groups_in_dim'] , self.ex_dict['groups_in_dim'] ).sum(dim = 1)
        sorted_indices = torch.argsort(w_sum.squeeze(), descending=True)
        self.virtual_operation.sv = sorted_indices

    # @torch._dynamo.disable
    def forward_gate(self, input):
        w_clone = self.linear.weight
        
        dtype = w_clone.dtype
        device = w_clone.device
        
        # _ = self.virtual_operation(input)
        # mask = self.virtual_operation.mask.to(device).to(dtype)
        mask = self.virtual_operation(input)

        if mask is None:
            mask = self.virtual_operation(input)
        # reshape 1-dim mask
        if mask.dim() == 1:
            mask = mask.view_as(w_clone)
        
        # multiple weights
        masked_weight = mask * w_clone
        # print("mask mean:", mask.mean())
        
        # print("########## masked weight: ", masked_weight)
        # scale
        # if self.scale_weight:
        #     scale = w_clone.norm()/masked_weight.norm().detach()
        #     masked_weight = scale*masked_weight

        bias = self.linear.bias
        out = nn.functional.linear(input, masked_weight, bias=bias)

        return out

        # if self.linear.bias is not None:
        #     return nn.functional.linear(input, masked_weight.to(dtype), bias=self.linear.bias.to(dtype))        
        # else:
        #     return nn.functional.linear(input, masked_weight.to(dtype))

    def forward_mask_norepeat(self, input):
        # print("################### Running forward_mask_norepeat")
        if self.expand_rate == 1:
            return self.forward_mask(input)
        else:
            dtype = input.dtype

            w_clone = self.linear.weight
            if type(self.virtual_operation.mask) == list:
                mask = [single_mask.to(input.get_device()) for single_mask in self.virtual_operation.mask]
            else: 
                mask = self.virtual_operation.mask.to(input.get_device())
            
            num_repeat = self.expand_rate
            weight_list = []
            for i in range(num_repeat):
                current_mask = mask[i].to(w_clone.dtype)
                masked_weight = current_mask*w_clone
                weight_list.append(masked_weight)

            if self.expand_dim == 0:
                masked_weight = torch.cat(weight_list, dim=0)
            elif self.expand_dim == 1:
                masked_weight = torch.cat(weight_list, dim=1)
        output = nn.functional.linear(input, masked_weight, bias=self.linear.bias.to(dtype)) if self.linear.bias is not None else nn.functional.linear(input, masked_weight)

        del masked_weight

        return output
        # if self.linear.bias is not None:
        #     return nn.functional.linear(input, masked_weight, bias=self.linear.bias.to(dtype))        
        # else:
        #     return nn.functional.linear(input, masked_weight)

    def forward_mask(self, input):
        # print("################### Running forward_mask")
        #self.weight = self.linear.weight
        w_clone = self.linear.weight
        dtype = input.dtype
        
        # if type(self.virtual_operation.mask) == list: # outdated
        if isinstance(self.virtual_operation.mask, list):
            mask = [single_mask.to(input.get_device()).to(dtype) for single_mask in self.virtual_operation.mask]
        else: 
            mask = self.virtual_operation.mask.to(input.get_device()).to(dtype)

        if not isinstance(mask, list) and mask.dim() == 1:
            mask = mask.view_as(w_clone)

        if self.expand_rate > 1:
            assert self.expand_rate % 2 == 0
            num_repeat = self.expand_rate//2
            weight_list = []
            for i in range(num_repeat):
                current_mask = mask[i].to(dtype)
                # if self.adapter:
                #     #print( self.adapter_modules[i].forward().size())
                #     current_w = w_clone + self.adapter_modules[i].forward()
                # else:
                current_w = w_clone.to(dtype)
                current_mask = current_mask.to(dtype)
                masked_weight = current_mask*current_w
                weight_list.append(masked_weight)

                ex_masked_weight = (1-current_mask.to(dtype))*current_w
                weight_list.append(ex_masked_weight)

            if self.expand_dim == 0:
                
                #masked_weight = torch.cat((masked_weight, ex_masked_weight), dim=0)
                masked_weight = torch.cat(weight_list, dim=0)
            elif self.expand_dim == 1:
                #masked_weight = torch.cat((masked_weight, ex_masked_weight), dim=1)
                masked_weight = torch.cat(weight_list, dim=1)
            #masked_weight = mask.to(w_clone.dtype)*w_clone
        else:
            current_mask = mask.to(dtype)
            masked_weight = mask*w_clone.to(dtype)
        del current_mask
        
        if self.adapter and self.expand_rate >1:
            for i in range(num_repeat):
                
                if self.expand_dim == 1:
                    in_size = 2*self.adapter_modules[i].in_dim
                    if i==0:
                        adapter_outputs = self.adapter_modules[i](input[:,:,i*in_size:(i+1)*in_size], mask[i], self.expand_dim)
                    else:
                        adapter_outputs += self.adapter_modules[i](input[:,:,i*in_size:(i+1)*in_size], mask[i], self.expand_dim)
                if self.expand_dim == 0:
                    if i==0: adapter_outputs = []
                    adapter_outputs.append(self.adapter_modules[i](input, mask[i], self.expand_dim))
            
            if self.expand_dim == 0:
                adapter_outputs = torch.cat(adapter_outputs, dim = -1)
                
        if self.linear.bias is not None:
            output = nn.functional.linear(input, masked_weight, bias=self.linear.bias.to(dtype))        
        else:
            output = nn.functional.linear(input, masked_weight)
        
        if self.adapter and self.expand_rate >1:
            output = output + adapter_outputs
        # if self.adapter:
        #     return output + self.adapter_mean(output)
        # else:
        return output

    def forward(self, input):
        #masked_weight = self.linear.weight*
        # print("######################## semisparselinear forward")
        dtype = input.dtype
        output = None
        
        if not self.mask_flag:
            dtype = input.dtype
            if self.linear.bias is not None:
                output = nn.functional.linear(input, self.linear.weight, bias=self.linear.bias.to(dtype))
                del input
                return output      
            else:
                output = nn.functional.linear(input, self.linear.weight)
                del input
                return output
        else:
            output = self.forward_gate(input)
            return output
        # else:
        #     if self.gate_flag:
        #         # w_clone = self.linear.weight.data
        #         output = self.forward_gate(input)
        #         del input
        #         return output
        #     else:
        #         if self.wo_repeat:
        #             return self.forward_mask_norepeat(input)
        #         else:
        #             return self.forward_mask(input)
            #     w_clone = self.linear.weight
            # if self.virtual_operation.mask is not None:
            #     mask = self.virtual_operation.mask.to(input.get_device())
            # else:
            #     mask = self.virtual_operation(input.get_device())

            # dtype = input.dtype
            # masked_weight = mask*w_clone
            # if self.scale_weight:
            #     scale = w_clone.norm()/masked_weight.norm().detach()
            #     masked_weight = scale*masked_weight

            # # if self.virtual_operation.bias is not None:
            # #     bias = self.virtual_operation.forward_bias(input.get_device())
            # #     masked_weight = masked_weight + mask*bias
            # if self.expand_rate > 1:
            #     assert self.expand_rate % 2 == 0
            #     num_repeat = expand_rate//2
            #     for in range(num_repeat):


            #     ex_masked_weight = (1-mask)*w_clone
            #     if self.expand_dim == 0:
            #         masked_weight = torch.cat((masked_weight, ex_masked_weight), dim=0)
            #     elif self.expand_dim == 1:
            #         masked_weight = torch.cat((masked_weight, ex_masked_weight), dim=1)
            # if self.linear.bias is not None:
            #     return nn.functional.linear(input, masked_weight, bias=self.linear.bias.to(dtype))        
            # else:
            #     return nn.functional.linear(input, masked_weight)

class SemiSparseShareLinear(SemiSparseLinear): 
    """Share-aware version of SemiSparseLinear.
    """
    def __init__(self, *args, share_pair_key: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # mark this module as share-capable
        self.is_qk_share: bool = share_pair_key is not None
        # key like 'layers.12.self_attn' used to pair q/k in the same block
        self.share_pair_key: Optional[str] = share_pair_key
        # optional numeric id (filled in by model_replace when pairing discovered)
        self.share_pair_id: Optional[int] = None

def group_parameters(model):
    # attn_params_names = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
    #         'self_attn.o_proj']
    mlp_params_names = ['mlp.gate_proj', 'mlp.up_proj','mlp.down_proj']
    other_group, mlp_group = [], []
    for n, p in model.named_parameters():
        for pattern in mlp_params_names:
            # print("################### iii = ", n)
            if pattern in n:
                mlp_group.append(p)
    for p in model.parameters():
        if p not in set(mlp_group):
            # print("################### iiii = ", n)
            other_group.append(p)

    return mlp_group, other_group


def model_replace(model, device_id, seperate_att=True, hf_model='llama', group_info={}, expand_rate=1, mlp_only=False, model_dim=False, wo_repeat=False, adapter=False):
    torch.cuda.empty_cache()
    print(model_dim)
    if hf_model == 'llama' or hf_model == 'qwen':
        # print("################### model replacing llama")
        factorization_param_names_for_matching=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj','mlp.up_proj']
        expand_layers_up = ['mlp.gate_proj', 'mlp.up_proj']
        expand_layers_down = ['mlp.down_proj']
        if mlp_only:
            factorization_param_names_for_matching = expand_layers_down + expand_layers_up
    elif hf_model == 'phi-1_5':
        factorization_param_names_for_matching=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
            'self_attn.dense', 'mlp.fc1', 'mlp.fc2']
        expand_layers_up = ['mlp.fc1']
        expand_layers_down = ['mlp.fc2']

        #model_dim_layers = []        
        if model_dim:
            model_dim_layers = ['self_attn.dense']
            mlp_layers = ['mlp.fc1', 'mlp.fc2']
            #model_dim_layers = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','mlp.fc1']

        if mlp_only:
            factorization_param_names_for_matching = expand_layers_down + expand_layers_up
    else:
        if seperate_att:
            factorization_param_names_for_matching=['attn.c_attn.query','attn.c_attn.key','attn.c_attn.value', 
            'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj', 'lm_head']
        else:
            factorization_param_names_for_matching=['attn.c_attn',
            'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj','lm_head' ]
    
    # model = self.model
    groups_in_dim = group_info['groups_in_dim']
    groups_out_dim = group_info['groups_out_dim']

    state_dict = model.state_dict()

    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
                # print(n)
                fc_weights = state_dict[n]
                module_name = n.replace('.weight', '')
                bias_name = n.replace('.weight', '.bias')
                if bias_name in state_dict:
                    bias = model.state_dict()[bias_name]
                    bias_flag = True
                else:
                    bias = None
                    bias_flag = False
                
                in_dim = fc_weights.size(1)
                out_dim = fc_weights.size(0)

                with torch.no_grad():
                    device = torch.device(f"cuda:{device_id}")
                    if expand_rate >1:
                        if pattern in expand_layers_up:
                            semi_sparse_module = SemiSparseLinear(in_dim, out_dim, groups_in_dim, groups_out_dim, \
                                expand_rate, expand_dim=0, bias=bias_flag, wo_repeat=wo_repeat, adapter=adapter).to(device)
                        elif pattern in expand_layers_down:
                            semi_sparse_module = SemiSparseLinear(in_dim, out_dim, groups_in_dim, groups_out_dim, \
                                expand_rate, expand_dim=1, bias=bias_flag, wo_repeat=wo_repeat, adapter=adapter).to(device)
                        else:
                            semi_sparse_module = SemiSparseLinear(in_dim, out_dim, groups_in_dim, groups_out_dim, \
                                bias=bias_flag).to(device)
                    else:
                        if model_dim:
                            if pattern in model_dim_layers:
                                # if pattern is in ['mlp.c_fc2']:
                                #     semi_sparse_module = SemiSparseLinear(in_dim, out_dim, groups_out_dim, int(groups_in_dim*4), bias=bias_flag).cpu()
                                # else:
                                semi_sparse_module = SemiSparseLinear(in_dim, out_dim, groups_out_dim, groups_in_dim, \
                                    bias=bias_flag).to(device)
                            elif pattern in mlp_layers:
                                semi_sparse_module = SemiSparseLinear(in_dim, out_dim, int(4*groups_in_dim), groups_out_dim, \
                                    bias=bias_flag).to(device)
                            else:
                                semi_sparse_module = SemiSparseLinear(in_dim, out_dim, groups_in_dim, groups_out_dim, \
                                    bias=bias_flag).to(device)
                        else:
                            semi_sparse_module = SemiSparseLinear(in_dim, out_dim, groups_in_dim, groups_out_dim, \
                                bias=bias_flag).to(device)
                    if bias is not None:
                        semi_sparse_module.linear.bias.copy_(bias.to(device))
                    semi_sparse_module.linear.weight.copy_(fc_weights.to(device))
                    # factorized_module = WSVD(fc_weights, preserve_ratio, bias, Q=None, transpose=transpose,
                    #                             gate_flag=True, init=True
                    #                             ).cpu()
                deepsetattr(model, module_name, semi_sparse_module)
                del fc_weights, bias, semi_sparse_module
                torch.cuda.empty_cache()
                # each pattern should only be matched once
                break
    # print("################## replace successful!")
    register_virtual_cache_hooks(model)
    # qk_share_by_seed(model, base_seed=42, p=None) 
    return model
   
def _compute_share_pair_key(module_name: str) -> Optional[str]:
    """Return a canonical key for pairing q_proj/k_proj of the same attention block.
    Example:
      'model.layers.12.self_attn.q_proj' -> 'model.layers.12.self_attn'
      'model.layers.12.self_attn.k_proj' -> 'model.layers.12.self_attn'
    """
    if '.self_attn.q_proj' in module_name:
        return module_name.split('.self_attn.q_proj')[0] + '.self_attn'
    if '.self_attn.k_proj' in module_name:
        return module_name.split('.self_attn.k_proj')[0] + '.self_attn'
    return None


def _attach_shared_virtual_op(module: nn.Module, shared_vo: 'virtual_operation', pair_id: int):  # type: ignore[name-defined]
    """Replace module.virtual_operation with the shared instance and stamp metadata."""
    # overwrite the virtual op reference
    module.virtual_operation = shared_vo
    # propagate ex_dict / expand flags from the module (the VO carries ex_dict already; ensure consistency if needed)
    # mark on both module and virtual op for discovery later
    setattr(module, 'is_qk_share', True)
    setattr(module, 'share_pair_id', pair_id)
    setattr(shared_vo, 'share_pair_id', pair_id)


def model_replace_with_qk_share(
    model: nn.Module,
    device_id: int,
    group_info: Dict[str, int],
    expand_rate: int = 1,
    mlp_only: bool = False,
    model_dim: bool = False,
    wo_repeat: bool = False,
    adapter: bool = False,
    hf_model: str = 'llama',
    seperate_att: bool = True,
    share_qk: bool = False,
):
    """A thin wrapper showing how to splice the sharing into your model_replace.

    If you prefer, fold this logic directly into your existing model_replace.
    """
    torch.cuda.empty_cache()
    if hf_model == 'llama' or hf_model == 'qwen':
        factorization_param_names_for_matching = [
            'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
            'mlp.gate_proj','mlp.down_proj','mlp.up_proj'
        ]
        expand_layers_up = ['mlp.gate_proj','mlp.up_proj']
        expand_layers_down = ['mlp.down_proj']
        if mlp_only:
            factorization_param_names_for_matching = expand_layers_down + expand_layers_up
    else:
        # keep your other branches as-is
        raise NotImplementedError('Only llama shown here; mirror your original code for others.')

    groups_in_dim = group_info['groups_in_dim']
    groups_out_dim = group_info['groups_out_dim']

    state_dict = model.state_dict()

    # Registry: map share_pair_key -> shared virtual_operation
    shared_registry: Dict[str, 'virtual_operation'] = {}  # type: ignore[name-defined]
    pair_counter: int = 0

    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
                fc_weights = state_dict[n]
                module_name = n.replace('.weight', '')
                bias_name = n.replace('.weight', '.bias')
                bias_flag = bias_name in state_dict
                bias = model.state_dict()[bias_name] if bias_flag else None

                in_dim = fc_weights.size(1)
                out_dim = fc_weights.size(0)
                device = torch.device(f"cuda:{device_id}")

                # choose class depending on share flag and whether this is q/k
                is_q_or_k = ('self_attn.q_proj' in module_name) or ('self_attn.k_proj' in module_name)
                share_pair_key = _compute_share_pair_key(module_name) if (share_qk and is_q_or_k) else None
                use_share_class = (share_qk and is_q_or_k)

                # build the module
                if expand_rate > 1:
                    if pattern in expand_layers_up:
                        cls = SemiSparseShareLinear if use_share_class else SemiSparseLinear  # type: ignore[name-defined]
                        if use_share_class:
                            semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                    expand_rate, expand_dim=0, bias=bias_flag,
                                    wo_repeat=wo_repeat, adapter=adapter,
                                    share_pair_key=share_pair_key).to(device)
                        else:
                            semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                    expand_rate, expand_dim=0, bias=bias_flag,
                                    wo_repeat=wo_repeat, adapter=adapter).to(device)

                    elif pattern in expand_layers_down:
                        cls = SemiSparseShareLinear if use_share_class else SemiSparseLinear  # type: ignore[name-defined]
                        if use_share_class:
                            semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                    expand_rate, expand_dim=1, bias=bias_flag,
                                    wo_repeat=wo_repeat, adapter=adapter,
                                    share_pair_key=share_pair_key).to(device)
                        else:
                            semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                    expand_rate, expand_dim=1, bias=bias_flag,
                                    wo_repeat=wo_repeat, adapter=adapter).to(device)
                    else:
                        cls = SemiSparseShareLinear if use_share_class else SemiSparseLinear  # type: ignore[name-defined]
                        if use_share_class:
                            semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                    bias=bias_flag,
                                    share_pair_key=share_pair_key).to(device)
                        else:
                            semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                    bias=bias_flag).to(device)
                else:
                    cls = SemiSparseShareLinear if use_share_class else SemiSparseLinear  # type: ignore[name-defined]
                    if use_share_class:
                        semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                bias=bias_flag,
                                share_pair_key=share_pair_key).to(device)
                    else:
                        semi = cls(in_dim, out_dim, groups_in_dim, groups_out_dim,
                                bias=bias_flag).to(device)

                with torch.no_grad():
                    semi.linear.to(device)
                    semi.linear.weight.copy_(
                        fc_weights.detach().to(semi.linear.weight.device, dtype=semi.linear.weight.dtype)
                    )

                    if bias is not None:
                        semi.linear.bias.copy_(
                            bias.detach().to(semi.linear.bias.device, dtype=semi.linear.bias.dtype)
                        )

                # If this is a shareable q/k, wire the SAME virtual_operation
                if share_qk and is_q_or_k and share_pair_key is not None:
                    if share_pair_key in shared_registry:
                        # attach existing shared virtual op to this module
                        shared_vo = shared_registry[share_pair_key]
                        _attach_shared_virtual_op(semi, shared_vo, pair_id=getattr(shared_vo, 'share_pair_id', -1))
                    else:
                        # first occurrence: register this module's virtual op and stamp a new pair id
                        shared_registry[share_pair_key] = semi.virtual_operation
                        setattr(semi.virtual_operation, 'share_pair_id', pair_counter)
                        setattr(semi, 'share_pair_id', pair_counter)
                        pair_counter += 1

                # install into model
                deepsetattr(model, module_name, semi)  # use your existing helper
                del fc_weights, bias, semi
                torch.cuda.empty_cache()
                break  # only match each weight once

    register_virtual_cache_hooks(model)  # keep your hook registration
    return model

class help_functions_hn(nn.Module):
    def __init__(self, structures, gamma=0.1):
        super().__init__()
        self.structures = structures    
        self.gamma = gamma
    def init_rank_reg(self, model):
        modules = list(model.modules())
        ind = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if type(m).__name__ == 'SemiSparseLinear':
                m.sort_weight()
                ind+=1
    
    def rank_reg(self, model):
        modules = list(model.modules())
        ind = 0
        rank_reg = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if type(m).__name__ == 'virtual_operation':
                rank_reg += m.rank_reg()

        return self.gamma*rank_reg
    
    def set_num_rank(self, model, rank=16):
        modules = list(model.modules())
        #rank_reg = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_operation':
                m.rank = rank
                
    
    def init_adpaters(self, model):
        modules = list(model.modules())

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'SemiSparseLinear':
                m.init_adpaters()

    def print_info(self,vectors):
        print(self.structures)
        config = []
        for i in range(len(vectors)):
            config.append(vectors[i].sum().item())

        print(config)

    def set_gate_vectors(self, model, vectors):
        modules = list(model.modules())
        ind = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if type(m).__name__ == 'virtual_operation':
                m.set_vector_value(vectors[ind])
                ind+=1
    def set_params_vectors(self, model, scales, biases=None):
        modules = list(model.modules())
        ind = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if type(m).__name__ == 'virtual_operation':
                m.set_scale_bias(scales[ind], biases[ind])
                ind+=1

    def set_scale_weight(self, model, scale_weight=False):
        
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if hasattr(m, 'scale_weight'):
                m.scale_weight = scale_weight

    def set_gate_status(self, model, use_gate=False):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if hasattr(m, 'gate_flag'):
                # if use_gate:
                #     m.weight = m.linear.weight
                m.gate_flag = use_gate

    def set_mask_status(self, model, use_mask=False):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if hasattr(m, 'mask_flag'):
                m.mask_flag = use_mask
    
    def generate_random_mask(self, model, p=0.5):
        modules = list(model.modules())
        ind = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_operation':
                m.generate_pv(seed=ind, p=p)
                ind+=1
                

class help_functions_share(nn.Module):
    def __init__(self, structures: List[int], gamma: float = 0.1):
        super().__init__()
        self.structures = structures
        self.gamma = gamma

    def _iter_unique_virtual_ops(self, model: nn.Module):
        """Yield each virtual_operation once (dedup Q/K pairs).
        We rely on a numeric share_pair_id placed on virtual_operation.
        """
        seen: set = set()
        for m in model.modules():
            if type(m).__name__ == 'virtual_operation':
                pid = getattr(m, 'share_pair_id', None)
                if pid is not None:
                    if pid in seen:
                        continue
                    seen.add(pid)
                yield m

    def init_rank_reg(self, model: nn.Module):
        pass

    def rank_reg(self, model: nn.Module):
        reg = 0.0
        for m in self._iter_unique_virtual_ops(model):
            reg = reg + m.rank_reg()
        return self.gamma * reg

    def set_num_rank(self, model: nn.Module, rank: int = 16):
        for m in self._iter_unique_virtual_ops(model):
            m.rank = rank

    def print_info(self, vectors: List[torch.Tensor]):
        print(self.structures)
        config = [v.sum().item() for v in vectors]
        print(config)

    def set_gate_vectors(self, model: nn.Module, vectors: List[torch.Tensor]):
        # Assign one vector per unique virtual op (i.e., per Q/K pair)
        idx = 0
        for m in self._iter_unique_virtual_ops(model):
            m.set_vector_value(vectors[idx])
            idx += 1

    def set_params_vectors(self, model: nn.Module, scales: List[torch.Tensor], biases: Optional[List[torch.Tensor]] = None):
        idx = 0
        for m in self._iter_unique_virtual_ops(model):
            b = biases[idx] if biases is not None else None
            m.set_scale_bias(scales[idx], b)
            idx += 1

    def set_scale_weight(self, model: nn.Module, scale_weight: bool = False):
        for mod in model.modules():
            if hasattr(mod, 'scale_weight'):
                mod.scale_weight = scale_weight

    def set_gate_status(self, model: nn.Module, use_gate: bool = False):
        for mod in model.modules():
            if hasattr(mod, 'gate_flag'):
                mod.gate_flag = use_gate

    def set_mask_status(self, model: nn.Module, use_mask: bool = False):
        for mod in model.modules():
            if hasattr(mod, 'mask_flag'):
                mod.mask_flag = use_mask

    def generate_random_mask(self, model: nn.Module, p: float = 0.5):
        idx = 0
        for m in self._iter_unique_virtual_ops(model):
            m.generate_pv(seed=idx, p=p)
            idx += 1


def deepsetattr(obj, attr, value):
    """Set object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_setattr(a, 'b.c', 2)
    >>> a.b.c
    2
    """
    if '.' not in attr:
        setattr(obj, attr, value)
    else:
        L = attr.split('.')
        deepsetattr(getattr(obj, L[0]), '.'.join(L[1:]), value)
        
def hidden_state_kd_loss(all_hidden_states_student, all_hidden_states_teacher, layer_weights=None):
    """
    Compute MSE loss between all corresponding hidden states of student and teacher.

    Args:
        all_hidden_states_student: tuple of tensors, each (batch, seq_len, dim_s)
        all_hidden_states_teacher: tuple of tensors, each (batch, seq_len, dim_t)
        layer_weights: optional list/tuple of layer weights, default = uniform

    Returns:
        Scalar tensor: averaged MSE loss across all matched layers
    """
    assert len(all_hidden_states_student) == len(all_hidden_states_teacher), \
        "Student and teacher must have same number of hidden states"
    
    L = len(all_hidden_states_student)
    if layer_weights is None:
        layer_weights = [1.0] * L

    total_loss = 0.0
    norm = 0.0
    for i, (hs_s, hs_t) in enumerate(zip(all_hidden_states_student, all_hidden_states_teacher)):
        w = layer_weights[i]
        
        # Compute MSE loss between corresponding hidden states
        loss_i = (hs_s - hs_t).pow(2).mean(-1).sum()
        total_loss += w * loss_i
        norm += w

    return total_loss / (norm + 1e-8)