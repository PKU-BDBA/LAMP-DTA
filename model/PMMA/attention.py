import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math

class Attention(nn.Module):
    def __init__(self, config, vis, mm=True):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.embed_dim, self.all_head_size)
        self.key = Linear(config.embed_dim, self.all_head_size)
        self.value = Linear(config.embed_dim, self.all_head_size)

        if mm:
            self.query_mol = Linear(config.embed_dim, self.all_head_size)
            self.key_mol = Linear(config.embed_dim, self.all_head_size)
            self.value_mol = Linear(config.embed_dim, self.all_head_size)
            self.out_mol = Linear(config.embed_dim, config.embed_dim)
            self.attn_dropout_mol = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_pm = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_mp = Dropout(config.transformer["attention_dropout_rate"])
            self.proj_dropout_mol = Dropout(config.transformer["attention_dropout_rate"])
            self.fc = Linear(config.mol_len + config.feat_len + 1, config.feat_len + 1)
            self.fc_mol = Linear(config.mol_len + config.feat_len + 1, config.mol_len)

        self.out = Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # print(self.num_attention_heads, self.attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mol=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if mol is not None:
            mol_q = self.query_mol(mol)      
            mol_k = self.key_mol(mol)      
            mol_v = self.value_mol(mol)      

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if mol is not None:
            query_layer_prot = query_layer
            key_layer_prot = key_layer
            value_layer_prot = value_layer
            query_layer_mol = self.transpose_for_scores(mol_q)
            key_layer_mol = self.transpose_for_scores(mol_k)
            value_layer_mol = self.transpose_for_scores(mol_v)

        if mol is None:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)
            attention_output = self.proj_dropout(attention_output)
            return attention_output, weights, None
        else:
            attention_scores_prot = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores_mol = torch.matmul(query_layer_mol, key_layer_mol.transpose(-1, -2))
            attention_scores_pm = torch.matmul(query_layer_prot, key_layer_mol.transpose(-1, -2))
            attention_scores_mp = torch.matmul(query_layer_mol, key_layer_prot.transpose(-1, -2))

            attention_scores_prot = attention_scores_prot / math.sqrt(self.attention_head_size)
            attention_probs_prot = self.softmax(attention_scores_prot)
            weights = attention_probs_prot if self.vis else None
            attention_probs_prot = self.attn_dropout(attention_probs_prot)

            attention_scores_mol = attention_scores_mol / math.sqrt(self.attention_head_size)
            attention_probs_mol = self.softmax(attention_scores_mol)
            attention_probs_mol = self.attn_dropout_mol(attention_probs_mol)

            attention_scores_pm = attention_scores_pm / math.sqrt(self.attention_head_size)
            attention_probs_pm = self.softmax(attention_scores_pm)
            attention_probs_pm = self.attn_dropout_pm(attention_probs_pm)

            attention_scores_mp = attention_scores_mp / math.sqrt(self.attention_head_size)
            attention_probs_mp = self.softmax(attention_scores_mp)
            guided_weights = attention_probs_mp if self.vis else None
            attention_probs_mp = self.attn_dropout_mp(attention_probs_mp)

            context_layer_prot = torch.matmul(attention_probs_prot, value_layer_prot)
            context_layer_prot = context_layer_prot.permute(0, 2, 1, 3).contiguous()

            context_layer_mol = torch.matmul(attention_probs_mol, value_layer_mol)
            context_layer_mol = context_layer_mol.permute(0, 2, 1, 3).contiguous()

            context_layer_pm = torch.matmul(attention_probs_pm, value_layer_mol)
            context_layer_pm = context_layer_pm.permute(0, 2, 1, 3).contiguous()

            context_layer_mp = torch.matmul(attention_probs_mp, value_layer_prot)
            context_layer_mp = context_layer_mp.permute(0, 2, 1, 3).contiguous()

            new_context_layer_shape = context_layer_prot.size()[:-2] + (self.all_head_size,)
            context_layer_prot = context_layer_prot.view(*new_context_layer_shape)

            new_context_layer_shape = context_layer_mol.size()[:-2] + (self.all_head_size,)
            context_layer_mol = context_layer_mol.view(*new_context_layer_shape)

            new_context_layer_shape = context_layer_pm.size()[:-2] + (self.all_head_size,)
            context_layer_pm = context_layer_pm.view(*new_context_layer_shape)

            new_context_layer_shape = context_layer_mp.size()[:-2] + (self.all_head_size,)
            context_layer_mp = context_layer_mp.view(*new_context_layer_shape)

            # attention_output_prot = self.out((context_layer_prot + context_layer_mp)/2)
            # attention_output_mol = self.out((context_layer_mol + context_layer_pm)/2)
            attention_output_prot = self.fc(torch.concat((context_layer_prot, context_layer_mp), dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            attention_output_prot = self.out(attention_output_prot)
            attention_output_mol = self.fc_mol(torch.concat((context_layer_mol, context_layer_pm), dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            attention_output_mol = self.out_mol(attention_output_mol)
            attention_output_prot = self.proj_dropout(attention_output_prot)
            attention_output_mol = self.proj_dropout_mol(attention_output_mol)
 
            return attention_output_prot, attention_output_mol, weights, guided_weights
