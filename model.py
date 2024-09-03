import copy
import math
import os
import warnings
import numpy as np
from typing import List, Optional, Tuple, Union


import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.autograd as autograd
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from args import T5_START_DOCSTRING, PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING, T5_INPUTS_DOCSTRING, _CONFIG_FOR_DOC

from transformers.activations import ACT2FN
from transformers import  BertConfig, BertModel, T5EncoderModel, BertGenerationEncoder, ViTModel, T5Model, T5ForConditionalGeneration, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerNorm, T5LayerCrossAttention, T5DenseActDense
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils import ModelOutput
import warnings
from config import T5SC_config
from dataclasses import dataclass
from modulator import qam_mod, qam_mapper, qam_demapper, channel_Awgn
logger = logging.get_logger(__name__)

@dataclass
class Seq2SeqLMSCOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    compression_rate: Optional[Tuple[torch.FloatTensor, ...]] = None
    confidence_rate:  Optional[Tuple[torch.FloatTensor, ...]] = None
    

@dataclass
class BaseModelSCOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    compression_rates: Optional[Tuple[torch.FloatTensor, ...]] = None
    confidence_rate:  Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_dict: Optional[Tuple[torch.FloatTensor, ...]] = None
    sparsity_loss: torch.FloatTensor = None


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x




class chan_mask_gen(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # self.in_conv = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.GELU()
        # )
        
        # self.out_conv = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(embed_dim // 2, embed_dim // 4),
        #     nn.GELU(),
        #     nn.Linear(embed_dim // 4, 2),
        #     nn.LogSoftmax(dim=-1)
        # )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.L = nn.Linear(embed_dim, embed_dim//2, bias=False)
        self.l1 = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        self.l2 = nn.Linear(embed_dim // 2, embed_dim // 4, bias=False)
        self.l3 = nn.Linear(embed_dim // 4, 1, bias=False)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
    


    def forward(self, x, noise_feature):   
        B, N, C = x.size()
        x = self.act(self.L(self.norm1(x)))
        # B, N, C = x.size()
        # local_x = x[:,:, :C//2]
        # global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
 
        # x = torch.cat([local_x, global_x.expand(B, N, C//2), noise_feature.expand(B,N,C//2)], dim=-1)
        # x = self.out_conv(x)
        x = torch.cat([x,noise_feature.expand(B,N,-1)], dim=-1)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x
    
class info_mask_gen(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # self.in_conv = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.GELU()
        # )
        
        # self.out_conv = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(embed_dim // 2, embed_dim // 4),
        #     nn.GELU(),
        #     nn.Linear(embed_dim // 4, 2),
        #     nn.LogSoftmax(dim=-1)
        # )
        # self.bert_config = BertConfig(hidden_size=512, num_hidden_layers=2, num_attention_heads=8)
        self.bert = BertModel.from_pretrained('google/bert_uncased_L-2_H-512_A-8')
        self.norm = nn.LayerNorm(embed_dim)
        self.L = nn.Linear(embed_dim, embed_dim, bias=False)
        self.l1 = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        self.l2 = nn.Linear(embed_dim // 2, embed_dim // 4, bias=False)
        self.l3 = nn.Linear(embed_dim // 4, 1, bias=False)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
    


    def forward(self, embedding, attention_mask): 
        # bert_outputs = self.bert(inputs_embeds=embedding, attention_mask=attention_mask)
        # bert_embedding = bert_outputs.last_hidden_state
        # x = self.norm(bert_embedding) 
        x = self.act(self.L(embedding))
        # print("L weight: ", self.L.weight)
        # print("l1 weight: ", self.l1.weight)
        # print("l2 weight: ", self.l2.weight)
        

        # B, N, C = x.size()
        # local_x = x[:,:, :C//2]
        # global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
 
        # x = torch.cat([local_x, global_x.expand(B, N, C//2), noise_feature.expand(B,N,C//2)], dim=-1)
        # x = self.out_conv(x)
        # print("norm: ", self.norm.weight)
        # print("L: ", self.L.weight)
        # print("l1: ", self.l1.weight)
        # print("l3: ", self.l3.weight)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x
    
class infoFSM(nn.Module):
    def __init__(self, mask_num=512, embed_dim=512):
        super().__init__()
        self.mask_generator = info_mask_gen(embed_dim)
        self.mask_num = mask_num
        
    def forward(self, input_feature, attention_mask, prev_m): 
        
        batch_size = input_feature.shape[0]
        prob = self.mask_generator(input_feature, attention_mask)  # Z^g Z^l Z^c
        self.STE = StraightThroughEstimator()
        
        # print("Prob shape: ", prob.shape)
        # print("Gumble_softmax: ", F.gumbel_softmax(prob, hard=True))
        # print("Input feature: ", input_feature)
        # print("input feature shape: ", input_feature.shape)
        # print("Is it training? ", self.training)
        
        if self.training:
            # curr_m = F.gumbel_softmax(prob, hard=True)[:, :, 0].squeeze() * prev_m
            curr_m = prob.squeeze(-1)*prev_m
            curr_m = self.STE(curr_m)
            curr_m = curr_m+1e-10
            # print("input_feature shape: ", input_feature.shape)
            mask = curr_m.int()
            curr_m_ = curr_m.unsqueeze(-1).expand(-1, -1, input_feature.shape[-1])
            
            input_feature = input_feature*curr_m_
            
            return input_feature, mask, curr_m
        else:
            curr_m = prob.squeeze(-1)*prev_m
            curr_m = self.STE(curr_m)
            curr_m = curr_m+1e-10
            mask = curr_m.int()
            curr_m_ = curr_m.unsqueeze(-1).expand(-1, -1, input_feature.shape[-1])
            
            # print(" prev_m: ",  prev_m.shape)
            # print("input_feature shape: ", input_feature.shape)
            curr_m_ = curr_m.unsqueeze(-1).expand(-1, -1, input_feature.shape[-1])
            input_feature = input_feature * curr_m_
            
            keep_indices = [
                row.nonzero().squeeze().tolist() if not isinstance(row.nonzero().squeeze().tolist(), int)
                else [row.nonzero().squeeze().tolist()]
                for row in mask
            ]  
            mask_indices = [[idx for idx in range(self.mask_num) if idx not in indices] for indices in keep_indices]  
            
            # prob_kept = torch.randn(prob_kept.shape).cuda()
            num_kept = torch.floor(torch.max(torch.sum(mask, dim = 1))).to(dtype=torch.int)
            num_kept = 1 if num_kept<1 else num_kept
            # print(torch.floor(torch.mean(torch.sum(curr_m, dim = 1))).to(dtype=torch.int))
            # print("number kept: ", num_kept)
            for row in range(batch_size):
                if len(keep_indices[row])<num_kept:
                    selected_mask_indices = torch.randperm(len(mask_indices[row]))[:(num_kept-len(keep_indices[row]))].tolist()
                    keep_indices[row] = keep_indices[row]+[mask_indices[row][i] for i in selected_mask_indices]
                    
            keep_indices = torch.tensor(keep_indices).to(input_feature.device)
            input_feature = batch_index_select(input_feature, keep_indices)
            mask = batch_index_select(mask, keep_indices)
            return input_feature, mask, curr_m
            # mask_indices = [[idx for idx in range(self.mask_num) if idx not in indices] for indices in keep_indices]
            
            # indices=[keep_indices[row]+mask_indices[row]  for row in range(batch_size)]
            # # print("indices: ", indices)
            # indices = torch.tensor(indices).to(input_feature.device)
            # input_feature=torch.gather(input_feature, 1, indices.unsqueeze(-1).expand(-1, -1, 512))
            # indices = torch.sum(curr_m, dim=1).to(dtype=torch.int)
            # new_tensor = torch.zeros_like(curr_m)
            # nonzero_indices = (curr_m != 0).int()
            # for idx, row in enumerate(indices):
            #     new_tensor[idx, :row] = torch.tensor(1,dtype=torch.int)
            
            # return input_feature, new_tensor, curr_m

class chanFSM(nn.Module):
    def __init__(self, mask_num=11, embed_dim=512):
        super().__init__()
        self.mask_generator = chan_mask_gen(embed_dim)
        self.mask_num = mask_num
        
    def forward(self, input_feature, noise_feature, prev_m): 
        batch_size = input_feature.shape[0]
        prob = self.mask_generator(input_feature, noise_feature)  # Z^g Z^l Z^c
        self.STE = StraightThroughEstimator()
        
        if self.training:
            # curr_m = F.gumbel_softmax(prob, hard=True)[:, :, 0].squeeze() * prev_m
            curr_m = prob.squeeze(-1)*prev_m
            curr_m = self.STE(curr_m)
            
            # print("input_feature shape: ", input_feature.shape)
            mask = curr_m.int()
            curr_m_ = curr_m.unsqueeze(-1).expand(-1, -1, input_feature.shape[-1])
            
            input_feature = input_feature*curr_m_
            curr_m = curr_m+1e-10
            return input_feature, mask, curr_m
        else:

            curr_m = prob.squeeze(-1)*prev_m
            curr_m = self.STE(curr_m)
            
            mask = curr_m.int()
            curr_m_ = curr_m.unsqueeze(-1).expand(-1, -1, input_feature.shape[-1])
            # print(" prev_m: ",  prev_m.shape)
            # print("input_feature shape: ", input_feature.shape)
            curr_m_ = curr_m.unsqueeze(-1).expand(-1, -1, input_feature.shape[-1])
            input_feature = input_feature * curr_m_
            curr_m = curr_m+1e-10

            keep_indices = [
                row.nonzero().squeeze().tolist() if not isinstance(row.nonzero().squeeze().tolist(), int)
                else [row.nonzero().squeeze().tolist()]
                for row in mask
            ]  
            mask_indices = [[idx for idx in range(self.mask_num) if idx not in indices] for indices in keep_indices]  
            # prob_kept = torch.randn(prob_kept.shape).cuda()
            num_kept = torch.floor(torch.max(torch.sum(mask, dim = 1))).to(dtype=torch.int)
            num_kept = 1 if num_kept<1 else num_kept
            # print(torch.floor(torch.mean(torch.sum(curr_m, dim = 1))).to(dtype=torch.int))
            # print("number kept: ", num_kept)
            for row in range(batch_size):
                if len(keep_indices[row])<num_kept:
                    selected_mask_indices = torch.randperm(len(mask_indices[row]))[:(num_kept-len(keep_indices[row]))].tolist()
                    keep_indices[row] = keep_indices[row]+[mask_indices[row][i] for i in selected_mask_indices]
                    
            keep_indices = torch.tensor(keep_indices).to(input_feature.device)
            input_feature = batch_index_select(input_feature, keep_indices)
            mask = batch_index_select(mask, keep_indices)
            return input_feature, mask, curr_m


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5SC_config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        
        

        hidden_states = self.wo(hidden_states)
        return hidden_states

# class T5DenseGatedActDense(nn.Module):
#     def __init__(self, config: T5SC_config, num_experts=3):
#         super().__init__()
#         self.num_experts = num_experts
#         self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)
#         self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
#         self.experts = nn.ModuleList([nn.Linear(config.d_model, config.d_ff, bias=False) for _ in range(num_experts)])
#         self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
#         self.dropout = nn.Dropout(config.dropout_rate)
#         self.act = ACT2FN[config.dense_act_fn]
#         self.sparsity_weight =1e-2
#         # print(self.experts[0].weight)
#         # print(self.experts[1].weight)

#     def forward(self, hidden_states):
#         # Compute gate values
#         gate_values = self.gate(hidden_states)
#         gate_probabilities = F.softmax(gate_values, dim=-1)
#         # print("gate_probabilities: ", gate_probabilities)

#         # Get top-2 experts
#         top2_experts = torch.topk(gate_probabilities, 2, dim=-1).indices
#         # print("top2_experts: ", top2_experts)

#         # Calculate the output of the top-2 experts
#         outputs = torch.stack([self.experts[i](hidden_states) for i in range(self.num_experts)], dim=-1)

#         # Gather outputs from the top-2 experts
#         batch_size, seq_len, hidden_dim, _ = outputs.shape
#         top2_experts_expanded = top2_experts.unsqueeze(-1).expand(batch_size, seq_len, 2, hidden_dim)
#         top2_outputs = torch.gather(outputs, -1, top2_experts_expanded).permute(0, 1, 3, 2)
#         # print("top2_outputs: ", top2_outputs)

#         # Combine the outputs from the top-2 experts
#         top2_probabilities = torch.gather(gate_probabilities, -1, top2_experts)
#         top2_probabilities = top2_probabilities / top2_probabilities.sum(dim=-1, keepdim=True)
#         # print("top2_probabilities: ", top2_probabilities)
#         output = torch.sum(top2_outputs * top2_probabilities.unsqueeze(-2), dim=-1)
#         hidden_states = self.dropout(output)

#         # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
#         # See https://github.com/huggingface/transformers/issues/20287
#         # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
#         if (
#             isinstance(self.wo.weight, torch.Tensor)
#             and hidden_states.dtype != self.wo.weight.dtype
#             and self.wo.weight.dtype != torch.int8
#         ):
#             hidden_states = hidden_states.to(self.wo.weight.dtype)
        

#         hidden_states = self.wo(hidden_states)

#         # Compute sparsity regularization term
#         sparsity_loss = self.sparsity_regularization(gate_probabilities)

#         return hidden_states, sparsity_loss

#     def sparsity_regularization(self, gate_probabilities):
#         # Compute entropy-based sparsity regularization term
#         sparsity_loss = -torch.sum(gate_probabilities * torch.log(gate_probabilities + 1e-10))  # Add a small constant for numerical stability
#         return self.sparsity_weight * sparsity_loss

class T5LayerFF(nn.Module):
    def __init__(self, config: T5SC_config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
            # self.DenseReluDense = T5MoEDenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        
        return hidden_states



class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights
        
        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]


            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        
        hidden_states = self.layer[-1](hidden_states)
        

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs
        # if self.training:
        #     outputs = outputs + (sparsity_loss,)
        # print("outputs: ", outputs)
        # print("sparsity_loss: ",sparsity_loss)
        # print("attention_outputs: ", attention_outputs)
        # input("Stop")
        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)



class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        
        
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.num_layers = config.num_layers
        self.num_infoFSM = config.num_infoFSM
        self.num_chanFSM = config.num_chanFSM
        if not self.is_decoder:
            self.FSMs = nn.ModuleList([infoFSM(mask_num=config.num_token) for _ in range(self.num_infoFSM)])
            for _ in range(self.num_chanFSM):
                self.FSMs.append(chanFSM(mask_num=config.num_token))
            self.noise_func = nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,64),
                        nn.ReLU(), nn.Linear(64, config.d_model//2),nn.ReLU())
            
            
        

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        noise_std = 10**(-torch.FloatTensor([14.0])/20),
        mode='info',
        sparsity_loss = torch.tensor(0.0, dtype=torch.float32),
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.half()

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # print("Is decoder? ", self.is_decoder)
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)
        compression_rate_group = ()
        confidence_rate_group= ()
        mask_dict = ()

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            # if not self.is_decoder and i==len(self.block)-1 and mode=='info':
            #     break
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
                        

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            # print("------------")
            # print("layer_outputs: ", layer_outputs)
            # if self.training:
            #     sparsity_loss = layer_outputs[-1]
                # sparsity_loss += sparsity_loss
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

            if not self.is_decoder and i>= self.num_layers-(self.num_infoFSM+self.num_chanFSM):
                d = i-(self.num_layers-(self.num_infoFSM+self.num_chanFSM))
                if d<self.num_infoFSM:
                    hidden_states, mask_dict_, curr_m = self.FSMs[d](input_feature=hidden_states, attention_mask=attention_mask, prev_m=attention_mask)
                    # hidden_states = hidden_states * curr_m.unsqueeze(-1)
                    input_shape = hidden_states.size()[:-1]
                    extended_attention_mask = self.get_extended_attention_mask(mask_dict_, input_shape)
                    confidence_rate_group = confidence_rate_group + ( -(curr_m * torch.log(curr_m)).sum(dim=1),)
                    compression_rate_group = compression_rate_group + (torch.sum(curr_m, dim = 1),)
                    mask_dict = mask_dict + (mask_dict_,)
                    compression_rates = compression_rate_group[-1]
                    confidence_rate = confidence_rate_group[-1]
                    # print("origianl length: ", torch.mean(compression_rate_group[0].float()))
                else:
                    # print("In chanFSM")
                    noise_feature = self.noise_func(noise_std.to(input_ids.device).unsqueeze(1))
                    # print("hidden stante: ", hidden_states)
                    # print("curr_m: ", curr_m.shape)
                    hidden_states, mask_dict_, curr_m = self.FSMs[d](input_feature=hidden_states, noise_feature=noise_feature, prev_m=attention_mask)
                    # print("currmask: ", curr_m.squeeze()[0])
                    # print("Attension mask: ", attention_mask[0])
                    # print("currmask: ", curr_m.squeeze()[0])
                    
                    input_shape = hidden_states.size()[:-1]
                    extended_attention_mask = self.get_extended_attention_mask(mask_dict_, input_shape)
                    confidence_rate_group = confidence_rate_group + ( -(curr_m * torch.log(curr_m)).sum(dim=1),)
                    compression_rate_group = compression_rate_group + (torch.sum(curr_m, dim = 1),)
                    mask_dict = mask_dict + (mask_dict_,)
                    compression_rates = compression_rate_group[-1]/compression_rate_group[0]
                    confidence_rate = confidence_rate_group[-1]

            elif not self.is_decoder:
                compression_rate_group = compression_rate_group + (torch.sum(attention_mask, dim = 1),)
                compression_rates = compression_rate_group[-1]
                confidence_rate_group = confidence_rate_group + (None,)
                confidence_rate = confidence_rate_group[-1]
                # print("compression rates: ", compression_rate)
            else:
                mask_dict = mask_dict + (encoder_attention_mask,)
                compression_rate_group = compression_rate_group + (torch.sum(attention_mask, dim = 1),)
                compression_rates = compression_rate_group[-1]
                confidence_rate_group = confidence_rate_group + (None,)
                confidence_rate = confidence_rate_group[-1]
            # print("compression rates: ", torch.mean(compression_rates))


        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    compression_rate_group,
                ]
                if v is not None
            )
        
        if self.training:
            return BaseModelSCOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
                compression_rates=compression_rates,
                confidence_rate=confidence_rate,
                mask_dict=mask_dict,
                # sparsity_loss = sparsity_loss,
            )
        else:
            return BaseModelSCOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
                compression_rates=compression_rates,
                confidence_rate=confidence_rate,
                mask_dict=mask_dict,
            )

@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class T5SC_model(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5SC_config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.is_channel_disable = config.is_channel_disable

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.weight = config.weight
        self.weight_decay = config.weight_decay
        self.distortion = config.distortion

        self.bit_per_digit = 4
        self.codebook = VectorQuantizer(num_embeddings=2**self.bit_per_digit,
                                        embedding_dim=config.d_model,
                                        quan_bits=self.bit_per_digit)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        

    def initial_weights(self):
        for name, module in self.named_modules():
            if ('FSMs' in name) and (not 'bert' in name) and isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif 'noise_func' in name and isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
            elif ('gate' in name ) and isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            # elif isinstance(module, T5DenseGatedActDense):
            #     for layer in module.experts:
            #         layer.weight.data.copy_(module.wi_1.weight.data)
            elif ('lora_B' in name) and isinstance(module, nn.Linear):
                init.uniform_(module.weight, a=-0.01, b=0.01)
            elif 'FSMs' in name and isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            else:
                pass
        
        
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMSCOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task: Optional[str] = '',
        mode: Optional[str] = 'info',
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMSCOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("Return Dict? ", return_dict)
        # print("Encoder output? ", encoder_outputs)

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        if mode == 'info':
            self.is_channel_disable=True
        else: 
            self.is_channel_disable=False

        if self.training:
            snr_list = np.arange(-6, 16, 4)
            SNRdb = (torch.FloatTensor([1]) * np.random.choice(snr_list)).to(self.device)
            # SNRdb = torch.FloatTensor([-6.0]).to(self.device)
        else:
            SNRdb = torch.FloatTensor([14.0]).to(self.device)
        # print("Is training? ", self.training, "mode: ", mode, "SNR: ", SNRdb)
        noise_std = 10**(-SNRdb/20)
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                noise_std = noise_std,
                mode=mode,
            )
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )
        
        
        if self.training:
            hidden_states = encoder_outputs[0]
            compression_rate = encoder_outputs[-3]
            confidence_rate = encoder_outputs[-2]
            mask_dict = encoder_outputs[-1][-1]
            # sparsity_loss = encoder_outputs[-1]
        else:
            hidden_states = encoder_outputs[0]
            compression_rate = encoder_outputs[-3]
            confidence_rate = encoder_outputs[-2]
            mask_dict = encoder_outputs[-1][-1]

        hidden_states, codebook_loss = self.codebook(hidden_states, SNRdb)
        
        

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        # print("mask_dict: ",mask_dict[0])
        
        # Decode
        if self.training:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=mask_dict,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # sparsity_loss = sparsity_loss,
            )
        else:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=mask_dict,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = decoder_outputs[0]
        # sparsity_loss = decoder_outputs[-1]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
        # print("sequence_output: ", sequence_output.shape)
        lm_logits = self.lm_head(sequence_output)
        # print("lm_logits: ", lm_logits.shape)
    
        loss=None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            # print("Average compression rate: ", torch.mean(compression_rate))
            # if self.training:
            #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)) + sparsity_loss
            # else:
            #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))+100*torch.mean(compression_rate)+1e3*codebook_loss
            # if task == 'sen':
            #     loss =  1e4*max(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))-0.4, 0.0)+10*torch.mean(compression_rate)#+5*torch.mean(confidence_rate)
            # elif task == 'trans':
            #     loss = 1e4*max(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))-2.9, 0.0)+1e-1*torch.mean(compression_rate)#+1*torch.mean(confidence_rate)
            #     # loss = 1e5*max(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))-0.0, 0.0)#+1e-3*torch.mean(compression_rate)
            # else:
            #     loss = 1e4*max(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))-0.7, 0.0)+5e-1*torch.mean(compression_rate)#+1*torch.mean(confidence_rate)
            # print("SNR: ", SNRdb)
            # print("CrossEntropyLoss: ",loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)))
            # print("Loss from compression: ", torch.mean(compression_rate))
            # print("Loss from confidence: ", torch.mean(confidence_rate))
            # print("Loss from codebook: ", codebook_loss)
            # print("Final loss: ", loss)
            # input("Pausezzz")
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        # output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        # print(((loss,) ) if loss is not None else output)
        # print(output)
        # input("logits")

        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMSCOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            compression_rate=torch.mean(compression_rate)
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        
        

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
    
    def get_compression_rate(
        self,input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.FloatTensor] = None
    ):
        SNRdb = torch.FloatTensor([14.0]).to(self.device)
        noise_std = 10**(-SNRdb/20)
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                noise_std = noise_std,
                mode='chan'
            )
        compression_rate = encoder_outputs[-3]
        return torch.mean(compression_rate)

def array_to_binaryarray(input_array, num_bits):
    binary_matrix = (np.right_shift(input_array[:, None], np.arange(num_bits)[::-1]) & 1).astype(np.int32)
    return binary_matrix

def binaryarray_to_array(binary_array, num_bits):
    powers_of_2 = 2 ** np.arange(num_bits)[::-1]
    return np.sum(binary_array * powers_of_2, axis=1)


# Settings
M =16  # modulation order
        # signal-to-noise ratio
channel = 'awgn'  # channel type
mapping_table, demapping_table = qam_mod(M)
print(mapping_table)

def commun_sim(data_dec, snr=18, quan_bits=8):
    data_dec = data_dec.cpu().numpy().flatten()
    # print("data_dec: ", data_dec)
    
    data_bin = array_to_binaryarray(data_dec, quan_bits)
    # print("data_bin: ", data_bin)
    
    shape = data_bin.shape
    tx_bits = np.hstack(data_bin)
    # print("tx_bits: ", tx_bits)
    # Communication Process
    tx_symbols = qam_mapper(tx_bits, mapping_table)


    # Wireless Channel
    rx_symbols = channel_Awgn(tx_symbols, snr=snr)
    # rx_symbols = channel_Rayleigh(tx_symbols, snr=snr)
    # rx_symbols = channel_Rician(tx_symbols, snr=snr)
    # M-QAM Demodulation
    rx_bits = qam_demapper(rx_symbols, demapping_table)
    # print("rx_bits: ", rx_bits)
    rx_bits = rx_bits[: len(tx_bits)]
    # # Calculate BER
    # ber = bit_error_rate(tx_bits, rx_bits)
    # print(f"Bit Error Rate: {ber}")
    data_recover = binaryarray_to_array(rx_bits.reshape(shape), quan_bits)
    # print("data_recover: ", data_recover)
    
    # print(data_recover)
    # print(f"Data error number {np.sum(data_dec!=data_recover)/len(data_recover)}" )
    return data_recover

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 quan_bits: int = 4,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.quan_bits = quan_bits
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents, snr=5):
        latents = latents # [B x L x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BL x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]

        # Get the encoding that has the min distance
        # print("embedding: ", self.embedding.weight)
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
        shape = encoding_inds.shape
        # print("encoding_inds: ", encoding_inds)
        Rx_signal = commun_sim(encoding_inds, snr=snr, quan_bits=self.quan_bits)
        encoding_inds = torch.from_numpy(Rx_signal).reshape(shape).to(latents.device)
  
        
        # Convert to one-hot encodings
        device = latents.device

        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        # print((quantized_latents - latents))
        return quantized_latents.contiguous(), vq_loss  # [B x L x D]




