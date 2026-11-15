import torch
import torch.nn as nn
import torch.nn.functional as F
# from misc_functions import custom_grad_weight
import numpy as np
import math


class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        ctx.grad_w = grad_w
        input_clone = input.clone()
        return input_clone.float()
    @staticmethod
    def backward(ctx, grad_out):
        grad_input = ctx.grad_w * grad_out
        return grad_input, None

def sample_gumbel(shape,eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U+eps)+eps)

def gaussian_sigmoid_sample(logits, T, offset=0):
    sample = 0.05*torch.randn(logits.size())
    if logits.get_device() == -1:
        logits = logits.cpu()
    else:
        sample = sample.to(logits.get_device())

    y = logits + sample + offset

    return F.sigmoid(y / T)

def gumbel_sigmoid_sample(logits, T, offset=0):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.get_device() == -1:
        logits = logits.cpu()
        gumbel_sample = gumbel_sample.cpu()
    else:
        gumbel_sample = gumbel_sample.to(logits.get_device())

    y = logits + gumbel_sample + offset

    return F.sigmoid(y/T)

def generate_mask_function(th, soft_range, pos):
    outputs = (1/soft_range)*(soft_range + th - pos)
    return torch.clamp(outputs, min=0, max=1)

def hard_sample(out):
    binary_out = torch.round(out)
    binary_out = (binary_out - out).detach() + out
    return binary_out

class hypernetwork_width(nn.Module):
    def __init__(self, t_structures,):
        super(hypernetwork_width, self).__init__()

        self.base = 3.0
        self.soft_range = 0.03

        self.t_sp = t_structures

        self.th = 0.1

        self.h0 = torch.zeros(2, 1, 16)

        self.bi_GRU = nn.GRU(16, 16, bidirectional=True)
        self.inputs = nn.Parameter(torch.Tensor(len(t_structures), 1, 16))
        nn.init.normal_(self.inputs)
        self.inputs.requires_grad = False

        self.linear_list_tp = [nn.Linear(32, 1, bias=False) for i in range(len(self.t_sp))]

        self.linear_list_tp = nn.ModuleList(self.linear_list_tp)

        # self.ln_in = nn.LayerNorm([128])
        self.ln_tp = nn.LayerNorm([32])

    def adjust(self, th):
        # rescale
        th = torch.clamp(th, min=0.1, max=0.9)
        th = (th - 0.1)/(0.9-0.1)

        current_th = (1-self.th)*th + self.th
        return current_th

    def forward(self):
        final_outputs = []
        if self.ln_tp.weight.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.ln_tp.weight.get_device())

        outputs, hn = self.bi_GRU(self.inputs, self.h0)
        tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]
        tp_out = [torch.sigmoid(self.linear_list_tp[i](tp_out[i])+self.base) for i in range(len(self.linear_list_tp))]
        # for i in range(len(tp_out)):
        #     tp_out[i]
        for i in range(len(tp_out)):
            current_th = tp_out[i]
            pos = torch.arange(self.t_sp[i])
            if self.ln_tp.weight.get_device() == -1:
                pos = pos.cpu()
            else:
                pos = pos.to(self.ln_tp.weight.get_device())

            current_th = self.adjust(current_th)

            corrected_th = (self.t_sp[i]-1)*current_th + 1 - self.soft_range

            current_vector = generate_mask_function(corrected_th,
                                                    soft_range=self.soft_range*self.t_sp[i], pos=pos)
            final_outputs.append(current_vector.squeeze())

        if not self.training:
            # tp_out = [hard_sample(tp_out[i]) for i in range(len(self.linear_list_tp))]
            for i in range(len(final_outputs)):
                final_outputs[i][final_outputs[i]>0] = 1
            # outputs = [outputs[i][]=1 for i in range(len(outputs))]
        return final_outputs

    def hard_output(self):
        final_outputs = []
        if self.ln_tp.weight.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.ln_tp.weight.get_device())

        outputs, hn = self.bi_GRU(self.inputs, self.h0)
        tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]
        tp_out = [torch.sigmoid(self.linear_list_tp[i](tp_out[i]) + self.base) for i in range(len(self.linear_list_tp))]

        for i in range(len(tp_out)):
            current_th = tp_out[i].squeeze()
            current_th = self.adjust(current_th)
            corrected_th = (self.t_sp[i] - 1) * current_th + 1 - self.soft_range
            final_outputs.append(corrected_th)

        return final_outputs


class hypernetwork(nn.Module):
    def __init__(self, t_structures, t_in_strucutre=None, sorted=False, group_size=1):
        super(hypernetwork, self).__init__()
        self.T = 0.4
        self.base = 3.0
        self.group_size = group_size

        assert type(self.group_size) == int
        assert self.group_size >= 1

        self.t_sp = t_structures
        self.t_in = t_in_strucutre
        self.h0 = torch.zeros(2,1,64)

        self.bi_GRU = nn.GRU(32, 64, bidirectional=True)
        self.inputs = nn.Parameter(torch.Tensor(len(t_structures),1,32))
        nn.init.normal_(self.inputs)
        self.inputs.requires_grad=False

        self.linear_list_tp = [nn.Linear(128, int(self.t_sp[i]//self.group_size), bias=False) for i in range(len(self.t_sp))]

        self.linear_list_tp = nn.ModuleList(self.linear_list_tp)

        # self.ln_in = nn.LayerNorm([128])
        self.ln_tp = nn.LayerNorm([128])

        self.sorted = sorted

        self.imp_flag = False
        if t_in_strucutre is not None:
            self.linear_list_imp = [nn.Linear(128, self.t_in[i], bias=False) for i in range(len(self.t_in))]

            self.linear_list_imp = nn.ModuleList(self.linear_list_imp)

            self.ln_imp = nn.LayerNorm([128])
            self.imp_flag = True
    def generate_imp(self):
        if self.ln_imp.weight.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.ln_imp.weight.get_device())

        outputs, hn = self.bi_GRU(self.inputs, self.h0)

        tp_out = [F.gelu(self.ln_imp(outputs[i, :])) for i in range(len(self.linear_list_imp))]

        tp_out = [self.linear_list_imp[i](tp_out[i]) for i in range(len(self.linear_list_imp))]

        tp_out = [gaussian_sigmoid_sample(tp_out[i], offset=self.base, T=1).squeeze() for i in
                  range(len(self.linear_list_imp))]

        return tp_out

    def forward(self,):
        if self.ln_tp.weight.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.ln_tp.weight.get_device())
        # print(self.inputs.get_device())
        # print(self.h0.get_device())

        outputs, hn = self.bi_GRU(self.inputs, self.h0)

        if self.imp_flag:

            imp_out = [F.gelu(self.ln_imp(outputs[i, :])) for i in range(len(self.linear_list_imp))]

            imp_out = [self.linear_list_imp[i](imp_out[i]) for i in range(len(self.linear_list_imp))]

            imp_out = [gaussian_sigmoid_sample(imp_out[i], offset=self.base, T=1).squeeze() for i in
                      range(len(self.linear_list_imp))]

            tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]

            tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]

            tp_out = [gumbel_sigmoid_sample(tp_out[i], offset=self.base, T=self.T).squeeze() for i in
                      range(len(self.linear_list_tp))]

            if not self.training:    
                tp_out = [hard_sample(tp_out[i]) for i in range(len(self.linear_list_tp))]
            

            return tp_out, imp_out
        else:
            tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]

            tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]
            
            if not self.training:
                tp_out = [gumbel_sigmoid_sample(tp_out[i], offset=self.base, T=1e-3).squeeze() for i in
                      range(len(self.linear_list_tp))]
                if self.sorted:
                    tp_out = [torch.sort(tp_out[i],descending =True)[0] for i in range(len(self.linear_list_tp))] 
                tp_out = [hard_sample(tp_out[i]) for i in range(len(self.linear_list_tp))]
                for i in range(len(tp_out)):
                    if tp_out[i].sum()==0:
                        tp_out[i][0] = 1
                        
                if self.group_size>1:
                    tp_out = [tp_out[i].repeat_interleave(self.group_size) for i in range(len(self.linear_list_tp))]
            else:
                tp_out = [gumbel_sigmoid_sample(tp_out[i], offset=self.base, T=self.T).squeeze() for i in
                      range(len(self.linear_list_tp))]
                if self.sorted:
                    tp_out = [torch.sort(tp_out[i],descending =True)[0] for i in range(len(self.linear_list_tp))] 
                if self.group_size>1:
                    tp_out = [tp_out[i].repeat_interleave(self.group_size) for i in range(len(self.linear_list_tp))]

            return tp_out

    def hard_output(self):
        if self.ln_tp.weight.get_device() == -1:
            self.h0 = self.h0.cpu()
        else:
            self.h0 = self.h0.to(self.ln_tp.weight.get_device())

        outputs, hn = self.bi_GRU(self.inputs, self.h0)

        tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]

        tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]

        tp_out = [gumbel_sigmoid_sample(tp_out[i], offset=self.base, T=self.T).squeeze() for i in
                  range(len(self.linear_list_tp))]

        tp_out = [hard_sample(tp_out[i]) for i in range(len(self.linear_list_tp))]

        if self.group_size>1:
            tp_out = [tp_out[i].repeat_interleave(self.group_size) for i in range(len(self.linear_list_tp))]

        return tp_out

class Importance_Pruning():
    def __init__(self, t_structures, p):
        super(Importance_Pruning, self).__init__()
        self.strcutre = t_structures
        self.p = p
        self.position_tensors = [torch.zeros(t_structures[i]) for i in range(len(t_structures))]
            # nn.Parameter(torch.Tensor(len(t_structures), 1, 32))
    def collect_Q(self,Q_list):
        self.Q_list = Q_list

    def pruning_model(self):
        # print(self.position_tensors)
        position_tensor = torch.cat(self.position_tensors)
        importance_tensor = torch.cat(self.Q_list)

        print(int(self.p*position_tensor.size(0)))
        print(position_tensor.size(0))

        _,indices = torch.topk(importance_tensor,int(self.p*position_tensor.size(0)))
        position_tensor[indices] = 1
        return position_tensor

    def convert_list_back(self, vector):
        start = 0
        vectors = []
        for i in range(len(self.strcutre)):
            end = start + self.strcutre[i]
            current_vector = vector[start:end]
            vectors.append(current_vector)
            start = end
        cfgs = [vectors[i].sum().item() for i in range(len(vectors))]
        print(cfgs)
        return vectors





class virtual_operation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.pruning_vector = torch.ones(dim)
        self.ori_vector = None
        self.ex_dict = {}
    def forward(self, input):

        # print(input.size())
        if len(input.size())==4:
            p_v = self.pruning_vector[None,None,None,:]
            if input.get_device() == -1:
                p_v = p_v.cpu()
            else:
                p_v = p_v.to(input.get_device())

            input = p_v.expand_as(input)*input
            # return input
        elif len(input.size())==3:
            p_v = self.pruning_vector[None,None,:]

            if input.get_device() == -1:
                p_v = p_v.cpu()
            else:
                p_v = p_v.to(input.get_device())
            input = p_v.expand_as(input) * input
            # return input
        # if self.ori_index is not None:
        #     sizes = list(input.size())
        #     sizes[-1] = self.ori_index.size(0)
        #     place_holder = torch.zeros(torch.Size(sizes))
        #     if input.get_device() == -1:
        #         place_holder = place_holder.cpu()
        #         ori_index = self.ori_index.cpu()
        #     else:
        #         place_holder = place_holder.to(input.get_device())
        #         ori_index = self.ori_index.to(input.get_device())
        #     input = torch.index_copy(input=place_holder,dim=-1,index=ori_index,source=input)

        return input

    def masking_weight(self, weight):
        if len(weight.size())==1:
            p_v = self.pruning_vector
        else:
            p_v = self.pruning_vector[:,None]
        if weight.get_device() == -1:
            p_v = p_v.cpu()
        else:
            p_v = p_v.to(weight.get_device())

        masked_weight = weight.clone()*p_v
        return masked_weight

    def set_vector(self, value):
        self.pruning_vector = torch.ones(self.dim)
        assert value.size() == self.pruning_vector.size()
        if value is not None:
            self.pruning_vector = value.squeeze()
        else:
            self.pruning_vector = value

class finetune_gate(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, input):
        if len(input.size())==4:
            weight = self.weight[None, None, None, :]
            if self.bias is not None:
                bias = self.bias[None, None, None, :]
                input = weight.expand_as(input)*input + bias.expand_as(input)
            else: input = weight.expand_as(input)*input
            # return input
        elif len(input.size())==3:
            weight = self.weight[None, None, :]
            if self.bias is not None:
                bias = self.bias[None, None, None, :]
                input = weight.expand_as(input) * input + bias.expand_as(input)
            else:
                input = weight.expand_as(input) * input

        return input