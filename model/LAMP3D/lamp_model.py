import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,global_mean_pool as gep
from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, APPNPConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
from torch.nn import Sequential, Linear, GELU

from Utils import graph_pad
from model.PMMA.paired_multi_model_attention_model import PairedMultimodelAttention
import model.LAMP3D.configs as configs
from model.TGCA.tgt_guided_cross_attention_model import TargetGuidedCrossAttention

class LAMP(nn.Module): # TODO
    def __init__(self, config):
        super().__init__()

        print('LAMPModel Loaded')
        # mol_g
        self.n_output = config.n_output
        nn1 = Sequential(Linear(config.num_features_mol, config.num_features_mol), GELU(), Linear(config.num_features_mol, config.num_features_mol))
        self.mol_conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(config.num_features_mol, config.num_features_mol*2), GELU(), Linear(config.num_features_mol*2, config.num_features_mol*2))
        self.mol_conv2 = GINConv(nn2)
        nn3 = Sequential(Linear(config.num_features_mol*2, config.num_features_mol*4), GELU(), Linear(config.num_features_mol*4, config.num_features_mol*4))
        self.mol_conv3 = GINConv(nn3)
        self.mol_embed = nn.Linear(config.num_features_mol * 4, config.embed_dim * 8)

        # prot_g
        self.prot_convs = nn.ModuleList()
        for _ in range(5):
            self.prot_convs.append(TAGConv(config.num_features_prot, config.num_features_prot, 2))
        self.pool_prot = GlobalAttentionPooling(nn.Linear(config.num_features_prot, 1))
        self.prot_g_bilstm = nn.LSTM(config.num_features_prot, config.num_features_prot, num_layers=2, batch_first=True, bidirectional=True, dropout=config.dropout)
        self.prot_embed = nn.Linear(config.num_features_prot * 2, config.embed_dim)

        # prot_llm
        self.llm_fc = nn.Linear(1024, 128)
        self.llm_embed = nn.Linear(2560, CONFIGS['PMMA'].hidden_size // 2)
        self.prot_llm_bilstm = nn.LSTM(CONFIGS['PMMA'].hidden_size // 2, CONFIGS['PMMA'].hidden_size // 2, num_layers=2, bidirectional=True, dropout=config.dropout)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

        self.pmma = PairedMultimodelAttention(config=CONFIGS['PMMA'], vis=False)
        self.tgca = TargetGuidedCrossAttention(embed_dim=config.embed_dim, num_heads=1)

        # mlp
        self.fc1 = nn.Linear(CONFIGS['PMMA'].hidden_size, CONFIGS['PMMA'].hidden_size * 4)
        self.fc2 = nn.Linear(CONFIGS['PMMA'].hidden_size * 4, CONFIGS['PMMA'].hidden_size * 2)
        self.out = nn.Linear(CONFIGS['PMMA'].hidden_size * 2, self.n_output)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.out.bias, std=1e-6)

    def get_attn_weights(self):
        return self.attn_weights
    
    def get_guided_attn_weights(self):
        return self.guided_attn_weights
    
    def get_tgca_matrix(self):
        return self.A_tgca
    
    def extract_mol_feat(self, mol_g, bsz):
        mol_x, mol_edge_index, mol_batch = mol_g.x, mol_g.edge_index, mol_g.batch
        
        mol_feat = self.gelu(self.mol_conv1(mol_x, mol_edge_index))
        mol_feat = self.gelu(self.mol_conv2(mol_feat, mol_edge_index))
        mol_feat = self.gelu(self.mol_conv3(mol_feat, mol_edge_index))

        mol_feat = gep(mol_feat, mol_batch)

        mol_feat = self.dropout(self.gelu(self.mol_embed(mol_feat)))
        mol_feat = mol_feat.view(8, mol_feat.shape[0], -1)

        return mol_feat
    
    def extract_prot_embed(self, prot_gs):
        prot_embed = []
        for prot_g in prot_gs:
            feat_prot = prot_g.ndata['h']

            for prot_conv in self.prot_convs:
                feat_prot = self.gelu(prot_conv(prot_g, feat_prot))
            
            prot_repr = self.pool_prot(prot_g, feat_prot).view(1, -1, 31)
            prot_repr = F.pad(
                input=prot_repr, 
                pad=(0, 0, 0, 140 - prot_repr.size()[1]), 
                mode='constant', value=0)
            prot_repr, _ = self.prot_g_bilstm(prot_repr)
            prot_embed.append(prot_repr)

        prot_embed = torch.concat(prot_embed).permute(1, 0, 2)
        prot_embed = self.prot_embed(prot_embed)
            
        return prot_embed, len(prot_gs)
    
    def fill_prot_llms(self, prot_llms):
        prot_llms = graph_pad(prot_llms, 1024) # [bsz, llm_len, 2560]
        prot_llms = self.dropout(self.gelu(self.llm_fc(prot_llms.permute(0, 2, 1)))).permute(0, 2, 1)
        h = self.llm_embed(prot_llms)
        prot_llms, _ = self.prot_llm_bilstm(h) # [bsz, llm_len, 128]
        return prot_llms

    def forward(self, mol_g, prot_gs, prot_llms):
        prot_embed, bsz = self.extract_prot_embed(prot_gs) # [140, bsz, 128]
        mol_feat = self.extract_mol_feat(mol_g, bsz) # [8, bsz, 128]
        prot_llms = self.fill_prot_llms(prot_llms) # [bsz, 1024, 128]
        
        mol_embed, self.A_tgca = self.tgca(prot_llms.permute(1, 0, 2), mol_feat, mol_feat) # TODO: add mask;

        mol_embed = mol_embed.permute(1, 0, 2)
        prot_embed = prot_embed.permute(1, 0, 2)
        dt_embed, self.attn_weights, self.guided_attn_weights = self.pmma(prot_embed, mol_embed)

        # interact
        dt_embed = torch.mean(dt_embed, dim=1)
        dt_embed = self.fc1(dt_embed)
        dt_embed = self.gelu(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dt_embed = self.fc2(dt_embed)
        dt_embed = self.gelu(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dta = self.out(dt_embed).view(-1)
        return dta


CONFIGS = {
    'PMMA': configs.get_PMMA_config(),
}