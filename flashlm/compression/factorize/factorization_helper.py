import torch
from tqdm.auto import tqdm
import time
from torch.cuda.amp import GradScaler 
from .hypernetwork import virtual_operation
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from hn_modules.param_util import unwrap_model
# from hn_modules.param_util import unwrap_model
import numpy as np
from torch.cuda.amp import autocast
import torch.nn as nn
import os
import math

# from lit_llama.quantization import ColBlockQuantizedLinear, QEmbedding


class Head_SVD(torch.nn.Module):
    def __init__(self, in_size, out_size, rank, discrete_ae=False):
        super().__init__()

        self.rank = rank
        self.in_dim = in_size
        self.out_dim = out_size
        self.discrete_ae = discrete_ae

        self.A = torch.nn.Linear(self.in_dim, self.rank, bias=False,)
        self.B = torch.nn.Linear(self.rank, self.out_dim, bias=False,)

        if self.discrete_ae:
            self.encoder = torch.nn.Linear(self.in_dim, 128)
            self.decoder = torch.nn.Linear(128, self.rank)
    

    def forward(self, inp):
        x = self.A(inp)
        if self.discrete_ae:
            ae_x = self.encoder(inp)
            ae_x = torch.sign(torch.nn.Hardtanh()(ae_x/0.4)) - torch.nn.Hardtanh()(ae_x.detach()/0.4) + torch.nn.Hardtanh()(ae_x/0.4)
            ae_x = self.decoder(ae_x)
            x = x + ae_x
        x = self.B(x)
        return x


class WSVD(torch.nn.Module):
    def __init__(self, weights, rank, bias=None, Q=None, use_cuda=True, transpose=False, group=None, gate_flag=False, init=False):
        # use_cuda=True:  conversion-time=20s, mrpc-F1=88.20
        # use_cuda=False: conversion-time=200s, mrpc-F1=88.85
        super().__init__()
        assert weights.dim() == 2, "The weights must be a 2-D matrix," + "W = {}".format(weights.shape)
        out_dim, in_dim = weights.size()

        if rank > 0 and rank <= 1:
            self.rank = int(rank * min(in_dim, out_dim))
        else:
            assert rank <= min(in_dim, out_dim), "--factorization_rank is too large"
            self.rank = rank

        W = weights.data.t()  # default is in_dim major
        if use_cuda:  # use GPU to speedup SVD
            W = W.cuda()

        if init:
            self.A = torch.nn.Linear(in_dim, self.rank, bias=False, device=weights.device)
            self.B = torch.nn.Linear(self.rank, out_dim, bias=(bias is not None), device=weights.device)
            actual_rank = self.rank
        else:

            if not transpose:
                # in_dim major (default)
                US, V = weighted_truncated_svd(W, self.rank, Q, group)
                if use_cuda:  # use GPU to speedup SVD
                    US.to(weights.device)
                    V.to(weights.device)

                actual_rank = US.shape[1]
                self.A = torch.nn.Linear(in_dim, actual_rank, bias=False, device=weights.device)
                self.A.weight.data.copy_(US.t())

                self.B = torch.nn.Linear(actual_rank, out_dim, bias=(bias is not None), device=weights.device)
                self.B.weight.data.copy_(V.t())
                if bias is not None:
                    self.B.bias.data.copy_(bias)
            else:
                # out_dim major
                W = W.t()
                U, SV = weighted_truncated_svd(W, self.rank, Q, group)
                if use_cuda:  # use GPU to speedup SVD
                    U.to(weights.device)
                    SV.to(weights.device)

                actual_rank = SV.shape[0]
                self.A = torch.nn.Linear(in_dim, actual_rank, bias=False, device=weights.device)
                self.A.weight.data.copy_(SV)

                self.B = torch.nn.Linear(actual_rank, out_dim, bias=(bias is not None), device=weights.device)
                self.B.weight.data.copy_(U)
                if bias is not None:
                    self.B.bias.data.copy_(bias)

        if gate_flag:
            self.gate = virtual_operation(dim=actual_rank)
            self.ori_in_dim = in_dim
            self.ori_out_dim = out_dim
            self.mid_dim = actual_rank
        self.fg_flag = False
        self.gate_flag = gate_flag
    
    def round_to_block_size(self, block_size=32):
        actual_rank, in_dim = self.A.weight.size()
        out_dim, actual_rank = self.B.weight.size()

        round_actual_rank = max(block_size, (actual_rank // block_size) * block_size)

        self.A = torch.nn.Linear(in_dim, round_actual_rank, bias=False, device=self.A.weight.device)
        self.B = torch.nn.Linear(round_actual_rank, out_dim, bias=(self.B.bias is not None), device=self.A.weight.device)

    def set_importnace_score(self, importance):
        self.importance = None
        self.importance = importance

    def rank_pruning(self, current_rank, current_vector=None):
        out_dim_A, in_dim_A = self.A.weight.size()
        out_dim_B, in_dim_B = self.B.weight.size()

        if not isinstance(current_rank, int):
            current_rank = int(current_rank.item())

        pruned_A = torch.nn.Linear(in_dim_A, current_rank, bias=False, device=self.A.weight.device)
        pruned_B = torch.nn.Linear(current_rank, out_dim_B, bias=(self.B.bias is not None), device=self.A.weight.device)
        if current_vector is not None:
            idx = np.squeeze(np.argwhere(np.asarray(current_vector.cpu().numpy())))

        pruned_A.weight.data = self.A.weight.data[:current_rank,:].clone()
        pruned_B.weight.data = self.B.weight.data[:, :current_rank].clone()

        # pruned_A.weight.data = self.A.weight.data[idx,:].clone()
        # pruned_B.weight.data = self.B.weight.data[:, idx].clone()

        if self.B.bias is not None:
            pruned_B.bias.data.copy_(self.B.bias)
        self.A = pruned_A
        self.B = pruned_B

        self.gate = virtual_operation(dim=current_rank)
        self.gate_flag = False

        self.fg_flag = False

    def forward(self, x):
        x = self.A(x)
        if self.gate_flag:
            x = self.gate(x)
        x = self.B(x)
        return x

    def get_structure(self):
        return self.gate.dim

    def get_parameters(self):
        ori_params = self.ori_in_dim * self.ori_out_dim
        svd_prams = self.ori_in_dim * self.mid_dim + self.ori_out_dim * self.mid_dim

        return ori_params, svd_prams

    def set_vector_value(self, value):
        self.gate.set_vector(value)

class Seperate_QKV(torch.nn.Module):
    def __init__(self, weights, bias=None):
        # use_cuda=True:  conversion-time=20s, mrpc-F1=88.20
        # use_cuda=False: conversion-time=200s, mrpc-F1=88.85
        super().__init__()
        assert weights.dim() == 2, "The weights must be a 2-D matrix," + "W = {}".format(weights.shape)
        out_dim, in_dim = weights.size()
        assert out_dim == in_dim*3

        self.query = nn.Linear(in_dim, in_dim, bias=(bias is not None), device=weights.device )
        self.key = nn.Linear(in_dim, in_dim, bias=(bias is not None),  device=weights.device)
        self.value = nn.Linear(in_dim, in_dim, bias=(bias is not None),  device=weights.device)

        query_weight = weights[:in_dim,:]
        key_weight = weights[in_dim:2*in_dim,:]
        value_weight = weights[2*in_dim:,:]

        if bias is not None:
            query_bias = bias[:in_dim]
            key_bias = bias[in_dim:2*in_dim]
            value_bias = bias[2*in_dim]
        self.query.weight.data.copy_(query_weight)
        self.key.weight.data.copy_(key_weight)
        self.value.weight.data.copy_(value_weight)
        if bias is not None:
            self.query.bias.data.copy_(query_bias)
            self.key.bias.data.copy_(key_bias)
            self.value.bias.data.copy_(value_bias)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        output = torch.cat((query,key,value),dim=-1)
        return output
def Repalce_Attention(model):
    # factorization_param_names_for_matching=['query_key_value']
    # "attn.c_attn",
    # "attn.c_proj",
    # "mlp.c_fc1",
    # "mlp.c_fc2",
    # "mlp.c_proj",
    factorization_param_names_for_matching = ['attn.c_attn']
    state_dict = model.state_dict()
    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
                print(n)


                fc_weights = state_dict[n]
                module_name = n.replace('.weight', '')
                bias_name = n.replace('.weight', '.bias')
                if bias_name in state_dict:
                    bias = model.state_dict()[bias_name]
                else:
                    bias = None
                with torch.no_grad():
                    factorized_module = Seperate_QKV(fc_weights, bias)
                deepsetattr(model, module_name, factorized_module)

# (To be cleaned)
# def Repalce_Linear_with_Quantized_Linear(model, bits=4, group_size=-1,df_flag=False, emb=False):
#     if not df_flag:
#         factorization_param_names_for_matching = ['attn.c_attn.query.A','attn.c_attn.key.A','attn.c_attn.value.A', 
#             'attn.c_proj.A', 'mlp.c_fc1.A', 'mlp.c_fc2.A','mlp.c_proj.A', 'attn.c_attn.query.B','attn.c_attn.key.B','attn.c_attn.value.B', 
#             'attn.c_proj.B', 'mlp.c_fc1.B', 'mlp.c_fc2.B','mlp.c_proj.B', 'lm_head']
#         ori_param_names = ['attn.c_attn.query','attn.c_attn.key','attn.c_attn.value', 
#             'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj']
#     # else:
#     #     factorization_param_names_for_matching= [
#     #         'attention.query_key_value.query','attention.query_key_value.key','attention.query_key_value.value',
#     #         'attention.dense', 'mlp.dense_h_to_4h', 'mlp.dense_4h_to_h','embed_out']
#     if emb:
#         factorization_param_names_for_matching = factorization_param_names_for_matching + ['wte']
#     state_dict = model.state_dict()
#     for n in state_dict:
#         for pattern in factorization_param_names_for_matching:
#             if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
#                 if pattern == 'wte':
#                      print(n)
#                      module_name = n.replace('.weight', '')
#                      fc_weights = state_dict[n]
#                      q_module = QEmbedding(fc_weights.size(0), fc_weights.size(1),
#                                             bits=bits, tile_cols=group_size)
#                 else:

#                     print(n)
#                     module_name = n.replace('.weight', '')
#                     bias_name = n.replace('.weight', '.bias')
#                     fc_weights = state_dict[n]
#                     if bias_name in state_dict:
#                         bias = model.state_dict()[bias_name]
#                     else:
#                         bias = None

#                     q_module = ColBlockQuantizedLinear(fc_weights.size(1), fc_weights.size(0), bias is not None,
#                                             bits=bits, tile_cols=group_size)
                
#                 deepsetattr(model, module_name, q_module)
#                 # each pattern should only be matched once
#                 break

#         for pattern in ori_param_names:
#             if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
#                 print(n)
#                 module_name = n.replace('.weight', '')
#                 bias_name = n.replace('.weight', '.bias')
#                 fc_weights = state_dict[n]
#                 if bias_name in state_dict:
#                     bias = model.state_dict()[bias_name]
#                 else:
#                     bias = None

#                 q_module = ColBlockQuantizedLinear(fc_weights.size(1), fc_weights.size(0), bias is not None,
#                                         bits=bits, tile_cols=group_size)
                
#                 deepsetattr(model, module_name, q_module)
#                 break

def defactorize_modules(model, hf_flag=True):
    if hf_flag:
        factorization_param_names_for_matching=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj','mlp.up_proj', 'lm_head']
    else:
        factorization_param_names_for_matching=[ 'attention.query_key_value.query','attention.query_key_value.key','attention.query_key_value.value',
         'attention.dense', 'mlp.dense_h_to_4h', 'mlp.dense_4h_to_h']
    state_dict = model.state_dict()
    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.A.weight' in n:
                # if env.global_rank==0:
                print(n)

                n_a = n 
                n_b = n_a.replace('.A', '.B')

                module_name = n.replace('.A.weight','')


                bias_name = n_b.replace('.weight', '.bias')
                fc_a_weights = state_dict[n_a]
                fc_b_weights = state_dict[n_b]

                if bias_name in state_dict:
                    bias = model.state_dict()[bias_name]
                else:
                    bias = None
                
                weight_matrix = fc_b_weights.mm(fc_a_weights)
                linear_module = nn.Linear(fc_a_weights.size(1), fc_b_weights.size(0), bias = bias is not None)
                linear_module.weight.data.copy_(weight_matrix)
                if bias is not None:
                    linear_module.bias.data.copy_(bias)
                deepsetattr(model, module_name, linear_module)
                # each pattern should only be matched once
                break

def weighted_truncated_svd(W, l, Q=None, group=None, return_s=False):
    #     """Compress the weight matrix W of an inner product (fully connected) layer
    #     using truncated SVD weighted by Q.
    #     Parameters:
    #     W: N x M weights matrix
    #     l: number of singular values to
    #     Q: importance of each row of W
    #     group: This is only for the brute force attack experiment
    #     Returns:
    #     QinvUS, Vl: matrices such that W \approx QinvUS*Vl
    #     """
    if Q is None:
        Q = torch.diag(torch.ones(W.size(0)))
    else:
        Q += 1e-7  # Add a small value of float16 to avoid inverse of rep0
        Q = Q/(Q.max())
        Q = torch.diag(Q)
    Q = Q.to(W.device)
    print(Q.size())
    # print(W.size())
    
    
    if torch.isnan(Q.mean()):
        Q = torch.diag(torch.ones(W.size(0)))
        Q = Q.to(W.device)
    # print(Q)

    QW = torch.mm(Q, W)

    U, s, V = torch.svd(QW, some=True)

    if group is None:
        Ul = U[:, :l]
        sl = s[:l]
        V = V.t()
        Vl = V[:l, :]
    else:
        # This is only for the brute force attack experiment in paper
        max_group = 10
        assert group >= 1 and group <= max_group, 'group can only be an integer between 1 to 10'
        total_r = len(s)
        row_exclude_start = math.floor((group - 1) / max_group * total_r)
        row_exclude_end = math.floor(group / max_group * total_r) - 1
        Ul = torch.cat((U[:, :row_exclude_start], U[:, row_exclude_end + 1:]), dim=1)
        sl = torch.cat((s[:row_exclude_start], s[row_exclude_end + 1:]))
        V = V.t()
        Vl = torch.cat((V[:row_exclude_start, :], V[row_exclude_end + 1:, :]), dim=0)

    torch.cuda.empty_cache()
    if return_s:
        # Q = Q.cpu()
        # Ul = Ul.cpu()
        print(Ul.size())
        print(Q.size())
        QinvU = torch.mm(torch.inverse(Q), Ul)
        return QinvU, sl, Vl
    else:
        QinvUS = torch.mm(torch.inverse(Q), torch.mm(Ul, torch.diag(sl)))
        return QinvUS, Vl

def model_replace(model, seperate_att=True, hf_flag=True, u_rank=1.0):
    torch.cuda.empty_cache()
    if hf_flag:
         factorization_param_names_for_matching=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj','mlp.up_proj', 'lm_head']

    else:
        if seperate_att:
            factorization_param_names_for_matching=['attn.c_attn.query','attn.c_attn.key','attn.c_attn.value', 
            'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj', 'lm_head']
        else:
            factorization_param_names_for_matching=['attn.c_attn', 
            'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj','lm_head' ]
    
    transpose = False
    # model = self.model
    factorization_rank = {}
    factorization_rank['all'] = u_rank
    factorization_rank = factorization_rank['all'] 
    preserve_ratio = factorization_rank

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
                else:
                    bias = None
                # if seperate_att:
                #     n = mod_str + n

                # Q = Q_dict[n] if Q_dict is not None else None
                # Q = Q/Q.max()
                if isinstance(factorization_rank, dict):
                    try:
                        # Each parameter has its rank (defined in a dictionary)
                        preserve_ratio = factorization_rank[pattern]
                    except:
                        # if the dict of rank ratio has no definition for the layer, do not do factorization
                        preserve_ratio = -1
                if preserve_ratio <= -1:
                    # disable factorization for this layer
                    continue
                with torch.no_grad():
                    factorized_module = WSVD(fc_weights, preserve_ratio, bias, Q=None, transpose=transpose,
                                                gate_flag=True, init=True
                                                ).cpu()
                torch.cuda.empty_cache()
                deepsetattr(model, module_name, factorized_module)
                # each pattern should only be matched once
                break
    return model

def model_SVD_and_replace(model, Q_dict=None,seperate_att=False, u_rank=1.0, convert2hf=True):
    torch.cuda.empty_cache()
    if convert2hf:
        factorization_param_names_for_matching=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj','mlp.up_proj', 'lm_head']
    else:
        if seperate_att:
            factorization_param_names_for_matching=['attn.c_attn.query','attn.c_attn.key','attn.c_attn.value', 
            'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj', 'lm_head']
            # "attn.c_attn",
        # "attn.c_proj",
        # "mlp.c_fc1",
        # "mlp.c_fc2",
        # "mlp.c_proj",
        else:
            # factorization_param_names_for_matching=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
            factorization_param_names_for_matching=['attn.c_attn', 
            'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj','lm_head' ]
    transpose = False
    # model = self.model
    factorization_rank = {}
    factorization_rank['all'] = u_rank
    factorization_rank = factorization_rank['all'] 
    preserve_ratio = factorization_rank

    state_dict = model.state_dict()

    if Q_dict is not None:
        for key in Q_dict:
            print(key)
            if 'module.' in key:
                mod_str = 'module.'
            else:
                mod_str = ''
            if '_orig_mod.' in key:
                mod_str = '_orig_mod.'
            else:
                mod_str = ''
            # if 'ori_mo'
        if convert2hf:
            converted_Q_dict = {}
            target_keys=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj','mlp.up_proj', 'lm_head']

            source_keys = ['attn.c_attn.query','attn.c_attn.key','attn.c_attn.value', 
            'attn.c_proj', 'mlp.c_fc1', 'mlp.c_proj','mlp.c_fc2', 'lm_head']
            for key in Q_dict:
                for i in range(len(source_keys)):
                    if source_keys[i] in key:
                        if i == 0:
                            print(new_key)
                        new_key = key.replace(source_keys[i], target_keys[i])
                        new_key = new_key.replace('transformer.h','model.layers')
                        converted_Q_dict[new_key] = Q_dict[key]
            
            Q_dict = converted_Q_dict
            
    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
                # print(n)


                fc_weights = state_dict[n]
                module_name = n.replace('.weight', '')
                bias_name = n.replace('.weight', '.bias')
                if bias_name in state_dict:
                    bias = model.state_dict()[bias_name]
                else:
                    bias = None
                if seperate_att and Q_dict is not None:
                    n = mod_str + n

                Q = Q_dict[n] if Q_dict is not None else None
                if Q is not None:
                    print(Q.mean())
                # Q = Q/Q.max()
                if isinstance(factorization_rank, dict):
                    try:
                        # Each parameter has its rank (defined in a dictionary)
                        preserve_ratio = factorization_rank[pattern]
                    except:
                        # if the dict of rank ratio has no definition for the layer, do not do factorization
                        preserve_ratio = -1
                if preserve_ratio <= -1:
                    # disable factorization for this layer
                    continue
                with torch.no_grad():
                    factorized_module = WSVD(fc_weights, preserve_ratio, bias, Q=Q, transpose=transpose,
                                                gate_flag=True,
                                                ).cpu()
                torch.cuda.empty_cache()
                deepsetattr(model, module_name, factorized_module)
                # each pattern should only be matched once
                break
    return model

def model_SVD_lm_head(model, Q_dict=None, u_rank=1.0):
    factorization_param_names_for_matching=['lm_head']

    state_dict = model.state_dict()

    if Q_dict is not None:
        for key in Q_dict:
            print(key)
            if 'module.' in key:
                mod_str = 'module.'
            else:
                mod_str = ''
            if '_orig_mod.' in key:
                mod_str = '_orig_mod.'
            else:
                mod_str = ''

    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
                # print(n)


                fc_weights = state_dict[n]
                module_name = n.replace('.weight', '')
                bias_name = n.replace('.weight', '.bias')
                if bias_name in state_dict:
                    bias = model.state_dict()[bias_name]
                else:
                    bias = None
                if Q_dict is not None:
                    n = mod_str + n

                Q = Q_dict[n] if Q_dict is not None else None

                rank = min(fc_weights.size(0), fc_weights.size(1))
                print(int(rank*u_rank))
                with torch.no_grad():
                    U, S, V = weighted_truncated_svd(fc_weights.cuda().t(), int(rank*u_rank), Q, group=None, return_s=True)
        torch.cuda.empty_cache()
    return U, S, V
    
def get_approx_fisher(env,model, dataloader, out_dir, factorization_cache_gradients, ignored_token, num_iter, sep_att=True, fsdp=True) -> Dict[str, Union[torch.Tensor, Any]]:
    # args.factorization_transpose=False: sum gradient over in-dim (default)
    # args.factorization_transpose=True: sum gradient over out-dim

    
    device_id = env.local_rank
    non_batch_dataloader = dataloader
    scaler = GradScaler()
    
    pesudo_optimizer = torch.optim.AdamW([{'params':model.parameters(), 'initial_lr':0}], lr=0, weight_decay=0)

    # model = self._wrap_model(self.model, training=False)
    def compute_Q(Q_dict, key, value, grad_scale, transpose=False, running_mean=False):
        sum_dim = 0
        
        if transpose:
            sum_dim = 1
        nan_flag = torch.isnan(value.mean())
        if torch.isnan(value.mean()):
                # The data may be noisy. Ignore it when loss is nan.
                env.print_master(f"!!! nan Q_dict detected !!!")
                # masking nan values
                value = value[torch.isnan(value)] = 0
                # zero_value = torch.zeros(value.size()).to(value.device)
                # value = zero_value
        value = value.float()/grad_scale
        value = torch.sqrt(torch.sum(torch.square(value), sum_dim))
        if key not in Q_dict:
            Q_dict[key] = value
        else:
            if running_mean:
                Q_dict[key] = 0.95 * Q_dict[key] + 0.05 * value
            else:
                Q_dict[key] += value
        return nan_flag
    Q_dict = {}
    if sep_att:
        Q_dict_path = os.path.join(out_dir, 'Q_dict_sepatt.pt')
    else:
        raise NotImplementedError
       
    try:
        # Load from disk
        assert factorization_cache_gradients  # Skip loading if not using cache
        Q_dict = torch.load(Q_dict_path)
        print(f"Q_dict is loaded from {Q_dict_path}")
        for key, value in Q_dict.items() :
            print(key)
    except:
    # Compute and save
        print('Computing the weighting matrix ...')
        # progress_bar = tqdm(non_batch_dataloade)
        tic = time.time()
        iter_num=0
        for step, batch in enumerate(non_batch_dataloader):

            if iter_num>=num_iter:
                break
            
            input_ids, targets = batch['input_ids'].to(device_id), batch['labels'].to(device_id)
            # with autocast(dtype=torch.bfloat16):
            with autocast( dtype=torch.bfloat16):
                # logits = model(input_ids).logits
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignored_token)
            # loss.backward()
            if torch.isnan(loss):
                # The data may be noisy. Ignore it when loss is nan.
                env.print_master(f"!!! nan loss detected !!!")
                loss._fill(0)

            scaler.scale(loss).backward()
            scaler.unscale_(pesudo_optimizer)
                # scaler.unscale_(pesudo_optimizer)
            
            if fsdp:
                with model.summon_full_params(model, rank0_only=True, with_grads=True, writeback=False):
                    state_dict = model.state_dict()

                    for n in state_dict:
                        w = deepgetattr(model, n)
                        w_grad = w.grad
                        # if env.global_rank==0:
                        #     print(n)
                        #     print(w.grad is not None)
                        # print(w_grad)
                        # if w_grad is not None:
                        #     print(w_grad.size())
                        nan_flag = False
                        if w_grad is not None:
                            grad_scale = 1.0
                            # print(n)
                            current_flag = compute_Q(Q_dict, n, w_grad, grad_scale, False)
                            if current_flag or nan_flag:
                                nan_flag = True
                                Q_dict = {}

            else:
                if hasattr(model, "module"):
                    temp_model = model
                    state_dict = temp_model.module.state_dict()
                else:
                    temp_model = model
                    state_dict = temp_model.state_dict()
                nan_flag = False

                for n in state_dict:
                    if hasattr(model, "module"):
                
                        w = deepgetattr(model.module, n)
                        w_grad = w.grad
                    else:
                        w = deepgetattr(model, n)
                        w_grad = w.grad
                    if w_grad is not None:
                        grad_scale = 1.0
                        # print(n)
                        current_flag = compute_Q(Q_dict, n, w_grad, grad_scale, False)
                        if current_flag or nan_flag:
                            nan_flag = True
                            Q_dict = {}
            # if nan_flag:
            #     scaler.update(new_scale=scaler.backoff_factor)
            # else:
            #     scaler.update()
            scaler.update()
            pesudo_optimizer.zero_grad()

            toc = time.time() - tic
            tic = time.time()
            # progress_bar.update(1)
            if iter_num % 10 == 0:
                env.print_master(f"iter {iter_num}/{num_iter} time: {toc*1000:.2f}ms")
                # env.print_master(input_ids)
            iter_num += 1
        if env.global_rank==0:
            torch.save(Q_dict, Q_dict_path)
            print(f"Q_dict is saved to {Q_dict_path}")

    return Q_dict

def partial_defactorize_modules(model, hf_flag=True, lm_head_rate = 1):
    if hf_flag:
        factorization_param_names_for_matching=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj', 
            'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj','mlp.up_proj', 'lm_head']
    else:
        factorization_param_names_for_matching = ['attn.c_attn.query','attn.c_attn.key','attn.c_attn.value', 
        'attn.c_proj', 'mlp.c_fc1', 'mlp.c_fc2','mlp.c_proj', 'lm_head']
    
    state_dict = model.state_dict()
    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.A.weight' in n:

                n_a = n 
                n_b = n_a.replace('.A', '.B')

                module_name = n.replace('.A.weight','')


                bias_name = n_b.replace('.weight', '.bias')
                fc_a_weights = state_dict[n_a]
                fc_b_weights = state_dict[n_b]

                if bias_name in state_dict:
                    bias = model.state_dict()[bias_name]
                else:
                    bias = None
                
                if fc_a_weights.numel()+fc_b_weights.numel() > fc_a_weights.size(1)*fc_b_weights.size(0):
                    print(n)

                    weight_matrix = fc_b_weights.mm(fc_a_weights)
                    linear_module = nn.Linear(fc_a_weights.size(1), fc_b_weights.size(0), bias = bias is not None)
                    linear_module.weight.data.copy_(weight_matrix)
                    if bias is not None:
                        linear_module.bias.data.copy_(bias)
                    deepsetattr(model, module_name, linear_module)
                # each pattern should only be matched once
                break
    if lm_head_rate < 1:
        factorized_module = WSVD(model.lm_head.weight.data, rank=lm_head_rate, bias=False, Q=None, transpose=False, gate_flag=True, init=True)
        deepsetattr(model, 'lm_head', factorized_module)

        # model.lm_head.rank_pruning(current_rank = int(lm_head_rate*model.lm_head.A.weight.data.size(1)))

    print("number of final pruned parameters after pdf: %.3f" % (count_parameters(model) / 10 ** 6))

def round_mid_dim(model,):
    modules = list(model.modules())
    ind = 0
    for layer_id in range(len(modules)):
        m = modules[layer_id]

        if isinstance(m, WSVD):
            m.round_to_block_size()
            ind+=1
    print("number of final rounded parameters: %.3f" % (count_parameters(model) / 10 ** 6))
    return model

def model_SVD_and_replace_test(model, Q_dict=None, pruning_rate=0.35):
        # param_names_for_matching=['attention.self.query','attention.self.key','attention.self.value',
        # 'intermediate.dense','output.dense'] # attention.output.dense is included
    # preserve_ratio = self.args.factorization_rank
    factorization_param_names_for_matching=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
    transpose = False
    # model = self.model
    factorization_rank = {}
    factorization_rank['all'] = pruning_rate
    factorization_rank = factorization_rank['all'] 
    preserve_ratio = factorization_rank

    state_dict = model.state_dict()
    for n in state_dict:
        for pattern in factorization_param_names_for_matching:
            if pattern in n and '.weight' in n and state_dict[n].ndim == 2:
                fc_weights = state_dict[n]
                module_name = n.replace('.weight', '')
                bias_name = n.replace('.weight', '.bias')
                if bias_name in state_dict:
                    bias = model.state_dict()[bias_name]
                else:
                    bias = None
                Q = Q_dict[n] if Q_dict is not None else None
                if isinstance(factorization_rank, dict):
                    try:
                        # Each parameter has its rank (defined in a dictionary)
                        preserve_ratio = factorization_rank[pattern]
                    except:
                        # if the dict of rank ratio has no definition for the layer, do not do factorization
                        preserve_ratio = -1
                if preserve_ratio <= -1:
                    # disable factorization for this layer
                    continue
                with torch.no_grad():
                    factorized_module = WSVD(fc_weights, preserve_ratio, bias, Q=Q, transpose=transpose,
                                                gate_flag=True,
                                                )
                deepsetattr(model, module_name, factorized_module)
                # each pattern should only be matched once
                break
    print("number of final pruned parameters: %.3f" % (count_parameters(model) / 10 ** 6))

def getSublists(lst,n):
    sub_lists = []
    subListLength = len(lst) // n
    for i in range(0, len(lst), subListLength):
        sub_lists.append(lst[i:i+subListLength])
    
    return sub_lists

import itertools

def reorg_pruning_vector(vectors):
    new_vector = [0]*len(vectors)
    new_vector[:-1] = vectors[1:]
    new_vector[-1] = vectors[0]
    print(len(new_vector))

    sub_vectors = getSublists(new_vector[:-1], 28)
    
    for i in range(len(sub_vectors)):
        temp = sub_vectors[i][-1]
        sub_vectors[i][-1] = sub_vectors[i][-2]
        sub_vectors[i][-2] = temp
    
    sub_vectors = list(itertools.chain.from_iterable(sub_vectors))
    print(len(sub_vectors))
    new_vector[:-1] = sub_vectors

    return new_vector

def hard_rank_pruning(model, hn=None, vectors=None):
    with torch.no_grad():
        cfgs = []
        model.eval()
        if vectors is None:
            vectors = hn()
            hn.eval()
        else:
            vectors = vectors
        for i in range(len(vectors)):
            if isinstance(vectors[i], int):
                cfgs.append(vectors[i])
            else:
                cfgs.append(vectors[i].sum())
            if i==0: print(vectors[i])
        modules = list(model.modules())
        ind = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]

            if isinstance(m, WSVD):
                m.rank_pruning(cfgs[ind], None)
                ind+=1

    print("number of final pruned parameters: %.3f" % (count_parameters(model) / 10 ** 6))
    return model, vectors

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def deepgetattr(obj, attr):
    """Get object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_getattr(a, 'b.c')
    4
    """
    if '.' not in attr:
        return getattr(obj, attr)
    else:
        L = attr.split('.')
        return deepgetattr(getattr(obj, L[0]), '.'.join(L[1:]))

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