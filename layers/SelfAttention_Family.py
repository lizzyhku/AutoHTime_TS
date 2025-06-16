import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import math
from math import sqrt
import os

from utils.masking import TriangularCausalMask, ProbMask
from layers.Causal_local_masks import CausalLocalMasks


class FullAttention(CausalLocalMasks):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        attn_decay_type=None,
        attn_decay_scale=0,
        patch_num=1,
        train_attn_decay=False,
        record_scores=False,
    ):
        super(FullAttention, self).__init__(
            attn_decay_type=attn_decay_type,
            attn_decay_scale=attn_decay_scale,
            patch_num=patch_num,
            train_attn_decay=train_attn_decay,
        )
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.do_dropout = self.mask_type is None
        self.record_scores = record_scores

    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = scale * torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.record_scores:
            self.raw_scores = scores
            self.raw_weights = torch.softmax(scores, dim=-1)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        if self.record_scores:
            self.raw_weights = torch.softmax(scores, dim=-1)

        decay_mask = self.get_decay_mask().to(scores.device)
        scores = scores + decay_mask
        if self.record_scores:
            self.masked_scores = scores
            self.powerlaw_mask = decay_mask

        A = torch.softmax(scores, dim=-1)
        if self.record_scores:
            self.attn_weights = A
        if self.do_dropout:
            A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    
    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(
            L_K, (L_Q, sample_k)
        )  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    
    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        record_scores=False,
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.record_scores = record_scores

    
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # import pdb
        # pdb.set_trace()
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class AttentionLayerEnhance(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        attention_dropout=0.1,
        record_scores=False,
    ):
        super(AttentionLayerEnhance, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # Projection layers
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        # Autoregressive components
        self.n_heads = n_heads
        self.window_size = 32  # Local attention window
        self.dropout = nn.Dropout(attention_dropout)
        self.d_model = d_model
        self.record_scores = record_scores

        # Learnable decay rate for temporal attention
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def _compute_relative_positions(self, L, S):
        """Compute relative positions for autoregressive attention."""
        t = torch.arange(L).unsqueeze(1)  # (L, 1)
        t_prime = torch.arange(S).unsqueeze(0)  # (1, S)
        return t - t_prime  # (L, S)

    def _create_window_mask(self, L, S):
        """Vectorized sliding window mask"""
        device = next(self.parameters()).device
        t = torch.arange(L, device=device)
        start = (t - self.window_size//2).clamp(min=0)
        end = (t + self.window_size//2 + 1).clamp(max=S)

        # Broadcasting to create mask [L, S]
        return (t.view(-1,1) >= start.view(1,-1)) & (t.view(-1,1) < end.view(1,-1))

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project inputs
        queries = self.query_projection(queries).view(B, L, H, -1)  # [128, 7, 16, d_k]
        keys = self.key_projection(keys).view(B, S, H, -1)          # [128, 7, 16, d_k]
        values = self.value_projection(values).view(B, S, H, -1)    # [128, 7, 16, d_v]

        # Compute attention scores [B, H, L, S] -> [128, 16, 7, 7]
        scores = torch.einsum('blhd,bshd->bhls', queries, keys) / sqrt(self.d_model)

        # Autoregressive components
        if self.record_scores:
            self.raw_scores = scores.clone()

        # 1. Temporal decay (autoregressive weighting)
        rel_pos = self._compute_relative_positions(L, S).to(scores.device)  # [7, 7]
        decay = torch.exp(-self.gamma * torch.abs(rel_pos))  # [7, 7]
        scores = scores * decay.unsqueeze(0).unsqueeze(0)  # Broadcast to [128, 16, 7, 7]

        # 2. Sliding window masking
        window_mask = self._create_window_mask(L, S).to(scores.device)  # [7, 7]
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # 3. Strict causal masking
        if attn_mask is None:
            attn_mask = TriangularCausalMask(B, L, device=scores.device)
        scores.masked_fill_(attn_mask.mask, -np.inf)

        # Attention computation
        attn = torch.softmax(scores, dim=-1)  # [128, 16, 7, 7]
        attn = self.dropout(attn)

        # Context computation [128, 7, 16, d_v]
        context = torch.einsum('bhls,bshd->blhd', attn, values)
        context = context.reshape(B, L, -1)  # [128, 7, 16*d_v]

        if self.record_scores:
            self.attn_weights = attn

        return self.out_projection(context), attn

class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None
# class AttentionLayerEnhance(nn.Module):
#     def __init__(
#         self,
#         attention,
#         d_model,
#         n_heads,
#         d_keys=None,
#         d_values=None,
#         attention_dropout=0.1,
#         record_scores=False,
#     ):
#         super(AttentionLayerEnhance, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#         self.d_k = d_keys
#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#         self.record_scores = record_scores
#         self.window_size = 32
#         self.dropout = nn.Dropout(attention_dropout)
#         self.d_model = d_model
#     def compute_position_encodings(self, B, L, d_model):
#         """
#         Compute precomputed position encodings for all pairs (t, t').
#         """
        
#         t = torch.arange(L).unsqueeze(1)  # (L, 1)
#         t_prime = torch.arange(L).unsqueeze(0)  # (1, L)
#         relative_pos = t - t_prime  # (L, L)
#         relative_pos_batched = torch.stack([t - t_prime for _ in range(B)])  # (B, L, L)
#         relative_pos = torch.diagonal(relative_pos_batched, dim1=1, dim2=2)  # (B, L)
        
        
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

#         pe = torch.zeros(*relative_pos.shape, d_model)  # (L, L, d_model)
#         pe[..., 0::2] = torch.sin(relative_pos.unsqueeze(-1) * div_term)  # even dims
#         pe[..., 1::2] = torch.cos(relative_pos.unsqueeze(-1) * div_term)  # odd dims
#         return pe
#     def forward(self, queries, keys, values, attn_mask):
#         # import pdb
#         # pdb.set_trace()
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)
        
#         # Compute attention scores (batch_size, num_heads, L, S)
#         attention_scores = (queries @ keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

#         # import pdb
#         # pdb.set_trace()
#         relative_pos = self.compute_position_encodings(B, L, self.d_model).cuda().view(B, L,H,H)
#         attention_scores = attention_scores + relative_pos
#         # Vectorized sliding window mask implementation
#         device = attention_scores.device
#         t = torch.arange(L, device=device)
#         # Calculate window boundaries for each position
#         start_indices = (t - self.window_size // 2).clamp(min=0)
#         end_indices = (t + self.window_size // 2 + 1).clamp(max=S)  # Use S for cross-attention cases
        
#         # Create sliding window mask using broadcasting
#         # [1, 1, L, 1] >= [1, 1, 1, S] -> [1, 1, L, S]
#         # import pdb
#         # pdb.set_trace()
#         t = t.view(1, 1, -1, 1).repeat(1,1,1,H)
#         window_mask = (t.view(1, 1, -1, H) >= start_indices.view(1, 1, L, -1)) & \
#                     (t.view(1, 1, -1, H) < end_indices.view(1, 1, L, -1))
        
        
#         # import pdb
#         # pdb.set_trace()
        
#         # Expand mask to match attention scores dimensions [B, H, L, S]
#         window_mask = window_mask.expand(B, H, -1, -1).transpose(1, 2).contiguous()  # [B, L, H, S]
        
#         # Apply mask to attention scores
        
#         attention_scores = attention_scores.masked_fill(~window_mask, float('-inf'))

#         # Apply additional attention mask if provided
#         if attn_mask is not None:
#             attention_scores = attention_scores.masked_fill(~attn_mask, float('-inf'))

#         # Compute attention weights
#         attention_weights = torch.softmax(attention_scores, dim=-1)
        
#         # Apply attention dropout
#         attention_weights = self.dropout(attention_weights)
        
#         # Compute context vectors
#         context = attention_weights @ values
#         # print("check context mean:", torch.mean(context))
#         # println()
#         return context.transpose(1, 2).contiguous().view(B, L, -1),attention_weights
    # def forward(self, queries, keys, values, attn_mask):
    #     B, L, _ = queries.shape
    #     _, S, _ = keys.shape
    #     H = self.n_heads

    #     queries = self.query_projection(queries).view(B, L, H, -1)
    #     keys = self.key_projection(keys).view(B, S, H, -1)
    #     values = self.value_projection(values).view(B, S, H, -1)
    #     # Compute attention scores (batch_size, L, L)  # * tau.unsqueeze(0)
    #     attention_scores = (queries @ keys.transpose(-2, -1))  # (batch_size, L, L)
    #     attention_scores = attention_scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

    #     # # Apply sliding window mask
    #     window_mask = torch.zeros_like(attention_scores, dtype=torch.bool)
    #     for t in range(L):
    #         start = max(0, t - self.window_size // 2)
    #         end = min(L, t + self.window_size // 2 + 1)
    #         window_mask[:, t, start:end] = True
    #     attention_scores = attention_scores.masked_fill(~window_mask, float('-inf'))
    
    #     # Apply the mask to attention scores
    #     # attention_scores = attention_scores.masked_fill(~window_mask, float('-inf'))  # Using -inf for proper softmax
    #     # import pdb
    #     # pdb.set_trace()
    #     # # Compute attention weights (batch_size, L, L)
    #     # attention_scores = torch.softmax(attention_scores, dim=-1)
    #     print("attention_scores mean:", torch.mean(attention_scores))
    #     println()
        
    #     # # Step 4: Attention with Position Encodings
        
    #     # attention_scores = (Q @ K.transpose(-2, -1))*A_time  # (batch_size, L, L)
    #     # attention_scores = attention_scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
    #     # attention_scores = attention_scores.masked_fill(~window_mask, float('-inf'))
    #     # self.A = Fun.softmax(attention_scores, dim=-1)

    #     # # Compute transformer hidden states (batch_size, L, hidden_dim)
    #     # # H_transformer = self.A @ V
    #     # out = torch.matmul(self.A, V)  

    #     # # out, attn = self.inner_attention(queries, keys, values, attn_mask)
    #     out = attention_scores.view(B, L, -1)
        
    #     return self.out_projection(out), values
    