import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,global_mean_pool as gep
from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, APPNPConv
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
from torch.nn import Sequential, Linear, GELU, ReLU

from utils import graph_pad
from model.PMMA.paired_multi_model_attention_model import PairedMultimodelAttention
import model.PMMA.configs as configs
from model.TGCA.tgt_guided_cross_attention_model import TargetGuidedCrossAttention

class LAMP(nn.Module): # TODO
    def __init__(self, config):
        super().__init__()

        print('LAMPModel Loaded')
        # mol_g
        self.n_output = config.n_output
        nn1 = Sequential(Linear(config.num_features_mol, config.num_features_mol), ReLU(), Linear(config.num_features_mol, config.num_features_mol))
        self.mol_conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(config.num_features_mol, config.num_features_mol*2), ReLU(), Linear(config.num_features_mol*2, config.num_features_mol*2))
        self.mol_conv2 = GINConv(nn2)
        nn3 = Sequential(Linear(config.num_features_mol*2, config.num_features_mol*4), ReLU(), Linear(config.num_features_mol*4, config.num_features_mol*4))
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

        self.act = nn.ReLU()
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
        
        mol_feat = self.act(self.mol_conv1(mol_x, mol_edge_index))
        mol_feat = self.act(self.mol_conv2(mol_feat, mol_edge_index))
        mol_feat = self.act(self.mol_conv3(mol_feat, mol_edge_index))

        mol_feat = gep(mol_feat, mol_batch)

        mol_feat = self.dropout(self.act(self.mol_embed(mol_feat)))
        mol_feat = mol_feat.view(8, mol_feat.shape[0], -1)

        return mol_feat
    
    def extract_prot_embed(self, prot_gs):
        prot_embed = []
        for prot_g in prot_gs:
            feat_prot = prot_g.ndata['h']

            for prot_conv in self.prot_convs:
                feat_prot = self.act(prot_conv(prot_g, feat_prot))
            
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
        prot_llms = self.dropout(self.act(self.llm_fc(prot_llms.permute(0, 2, 1)))).permute(0, 2, 1)
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

        dt_embed = torch.cat((dt_embed, prot_embed, mol_embed), 1)

        # interact
        dt_embed = torch.mean(dt_embed, dim=1)
        dt_embed = self.fc1(dt_embed)
        dt_embed = self.act(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dt_embed = self.fc2(dt_embed)
        dt_embed = self.act(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dta = self.out(dt_embed).view(-1)
        return dta


CONFIGS = {
    'LAMPCM': configs.get_LAMPCM_config(),
}
    

class LAMPCM(nn.Module): # TODO
    def __init__(self):
        super().__init__()

        print('LAMPCMModel Loaded')
        config = CONFIGS['LAMPCM']
        # mol_g
        self.n_output = config.n_output
        nn1 = Sequential(Linear(config.num_features_mol, config.num_features_mol), ReLU(), Linear(config.num_features_mol, config.num_features_mol * 2))
        nn2 = Sequential(Linear(config.num_features_mol * 2, config.num_features_mol * 2), ReLU(), Linear(config.num_features_mol * 2, config.num_features_mol * 2))
        nn3 = Sequential(Linear(config.num_features_mol * 2, config.num_features_mol), ReLU(), Linear(config.num_features_mol, config.num_features_mol))
        self.mol_conv1 = GINConv(nn1)
        self.mol_conv2 = GINConv(nn2)
        self.mol_conv3 = GINConv(nn3)
        self.mol_fc1 = nn.Linear(1, config.embed_dim * 2)
        self.mol_fc2 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.mol_norm = nn.LayerNorm(config.embed_dim)

        # prot_g
        self.embed_dim = config.embed_dim
        pro_nn1 = Sequential(Linear(config.num_features_prot, config.num_features_prot), ReLU(), Linear(config.num_features_prot, config.num_features_prot * 2))
        pro_nn2 = Sequential(Linear(config.num_features_prot * 2, config.num_features_prot * 2), ReLU(), Linear(config.num_features_prot * 2, config.num_features_prot * 2))
        pro_nn3 = Sequential(Linear(config.num_features_prot * 2, config.num_features_prot), ReLU(), Linear(config.num_features_prot, config.num_features_prot))
        self.pro_conv1 = GINConv(pro_nn1)
        self.pro_conv2 = GINConv(pro_nn2)
        self.pro_conv3 = GINConv(pro_nn3)
        self.pro_fc1 = nn.Linear(1, config.embed_dim * 2)
        self.pro_fc2 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.pro_norm = nn.LayerNorm(config.embed_dim)

        # prot_llm
        self.llm_fc = nn.Linear(1024, config.num_features_llm)
        self.llm_embed = nn.Linear(2560, config.hidden_size // 2)
        self.prot_llm_bilstm = nn.LSTM(config.hidden_size // 2, config.hidden_size // 2, num_layers=2, bidirectional=True, dropout=config.dropout)

        prot_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.transformer.num_heads, dim_feedforward=config.embed_dim * 2, dropout=config.dropout, activation='relu')
        self.prot_transformer = nn.TransformerEncoder(prot_encoder_layer, num_layers=2)
        mol_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.transformer.num_heads, dim_feedforward=config.embed_dim * 2, dropout=config.dropout, activation='relu')
        self.mol_transformer = nn.TransformerEncoder(mol_encoder_layer, num_layers=2)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

        self.pmma = PairedMultimodelAttention(config=config, vis=False)
        self.mol_tgca = TargetGuidedCrossAttention(embed_dim=config.embed_dim, num_heads=1)
        self.mol_tgca_norm = nn.LayerNorm(config.embed_dim)
        self.prot_tgca = TargetGuidedCrossAttention(embed_dim=config.embed_dim, num_heads=1)
        self.prot_tgca_norm = nn.LayerNorm(config.embed_dim)
        # mlp
        # self.attn_pooling = Attn_Net_Gated(config.embed_dim, config.embed_dim)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.fc2 = nn.Linear(config.hidden_size * 8, config.hidden_size * 4)
        self.out = nn.Linear(config.hidden_size * 4 , self.n_output)

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
    
    def extract_mol_feat(self, mol_g):
        mol_x, mol_edge_index, mol_batch = mol_g.x, mol_g.edge_index, mol_g.batch
        
        mol_feat = self.act(self.mol_conv1(mol_x, mol_edge_index))
        mol_feat = self.act(self.mol_conv2(mol_feat, mol_edge_index))
        mol_feat = self.act(self.mol_conv3(mol_feat, mol_edge_index))

        mol_feat = gep(mol_feat, mol_batch)
        mol_feat = mol_feat.unsqueeze(2) # (bsz, f_m, 1)

        mol_feat = self.dropout(self.act(self.mol_fc1(mol_feat)))
        mol_feat = self.dropout(self.mol_fc2(mol_feat)) # (bsz, f_m, e)

        mol_feat = self.mol_norm(mol_feat)

        return mol_feat
    
    def extract_prot_feat(self, prot_g):
        prot_x, prot_edge_index, prot_batch = prot_g.x, prot_g.edge_index, prot_g.batch
        
        prot_embed = self.act(self.pro_conv1(prot_x, prot_edge_index))
        prot_embed = self.act(self.pro_conv2(prot_embed, prot_edge_index))
        prot_embed = self.act(self.pro_conv3(prot_embed, prot_edge_index))

        prot_embed = gep(prot_embed, prot_batch) # 
        prot_embed = prot_embed.unsqueeze(2)

        prot_embed = self.dropout(self.act(self.pro_fc1(prot_embed)))
        prot_embed = self.dropout(self.pro_fc2(prot_embed))

        prot_embed = self.pro_norm(prot_embed)
            
        return prot_embed
    
    def fill_prot_llms(self, prot_llms):
        prot_llms = graph_pad(prot_llms, 1024)
        prot_llms = self.dropout(self.act(self.llm_fc(prot_llms.permute(0, 2, 1)))).permute(0, 2, 1)
        h = self.llm_embed(prot_llms)
        prot_llms, _ = self.prot_llm_bilstm(h)
        return prot_llms

    def forward(self, mol_g, prot_g, prot_llms):
        prot_feat = self.extract_prot_feat(prot_g) # (bsz, 54, h)
        mol_feat = self.extract_mol_feat(mol_g) # (bsz, 82, h)
        prot_llms = self.fill_prot_llms(prot_llms) # (bsz, f, h)
        
        mol_embed, self.A_tgca = self.mol_tgca(prot_llms.permute(1, 0, 2), mol_feat.permute(1, 0, 2), mol_feat.permute(1, 0, 2)) # (f, bsz, h)
        prot_embed, A_tgca = self.prot_tgca(prot_llms.permute(1, 0, 2), prot_feat.permute(1, 0, 2), prot_feat.permute(1, 0, 2)) # (f, bsz, h)

        prot_embed = self.prot_transformer(prot_embed)
        prot_embed = prot_embed.permute(1, 0, 2) # (bsz, f, h)
        # mol_embed = mol_embed + mol_feat
        prot_embed = self.prot_tgca_norm(prot_embed)

        mol_embed = self.mol_transformer(mol_embed)
        mol_embed = mol_embed.permute(1, 0, 2) # (bsz, f, h)
        # mol_embed = mol_embed + mol_feat
        mol_embed = self.mol_tgca_norm(mol_embed)
        # mol_embed = torch.concat([mol_embed, mol_feat], dim=1)
        # mol_embed = mol_embed + mol_feat
        # prot_embed = prot_embed.permute(1, 0, 2)
        dt_embed, self.attn_weights, self.guided_attn_weights = self.pmma(prot_embed, mol_embed) # （bsz, f_pmma, h）

        dt_embed = torch.cat((dt_embed, prot_embed, mol_embed), 1) # (bsz, f_pmma + f * 2, h)
        # dt_embed = torch.cat((prot_embed, mol_embed), 1)
        # interact
        # A_pool, dt_embed = self.attn_pooling(dt_embed)
        # A_pool = torch.transpose(A_pool, -1, -2)
        # dt_embed = torch.bmm(F.softmax(A_pool, dim=-1), dt_embed)
        dt_embed = torch.mean(dt_embed, dim=1) # Tid: should mean dim 2 ranther than dim 1
        dt_embed = self.fc1(dt_embed) # (bsz, 1024)
        dt_embed = self.act(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dt_embed = self.fc2(dt_embed) # (bsz, 512)
        dt_embed = self.act(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dta = self.out(dt_embed) # (bsz, 1)
        return dta
    

class LAMPCM_re(nn.Module): # TODO
    def __init__(self):
        super().__init__()

        print('LAMPCMReducedModel Loaded')
        config = CONFIGS['LAMPCM']
        # mol_g
        self.n_output = config.n_output
        nn1 = Sequential(Linear(config.num_features_mol, config.num_features_mol), ReLU(), Linear(config.num_features_mol, config.num_features_mol * 2))
        nn2 = Sequential(Linear(config.num_features_mol * 2, config.num_features_mol * 2), ReLU(), Linear(config.num_features_mol * 2, config.num_features_mol * 2))
        nn3 = Sequential(Linear(config.num_features_mol * 2, config.num_features_mol), ReLU(), Linear(config.num_features_mol, config.num_features_mol))
        self.mol_conv1 = GINConv(nn1)
        self.mol_conv2 = GINConv(nn2)
        self.mol_conv3 = GINConv(nn3)
        self.mol_fc1 = nn.Linear(1, config.embed_dim * 2)
        self.mol_fc2 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.mol_norm = nn.LayerNorm(config.embed_dim)

        # prot_g
        self.embed_dim = config.embed_dim
        pro_nn1 = Sequential(Linear(config.num_features_prot, config.num_features_prot), ReLU(), Linear(config.num_features_prot, config.num_features_prot * 2))
        pro_nn2 = Sequential(Linear(config.num_features_prot * 2, config.num_features_prot * 2), ReLU(), Linear(config.num_features_prot * 2, config.num_features_prot * 2))
        pro_nn3 = Sequential(Linear(config.num_features_prot * 2, config.num_features_prot), ReLU(), Linear(config.num_features_prot, config.num_features_prot))
        self.pro_conv1 = GINConv(pro_nn1)
        self.pro_conv2 = GINConv(pro_nn2)
        self.pro_conv3 = GINConv(pro_nn3)
        self.pro_fc1 = nn.Linear(1, config.embed_dim * 2)
        self.pro_fc2 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.pro_norm = nn.LayerNorm(config.embed_dim)

        # prot_llm
        self.llm_fc = nn.Linear(64, config.num_features_llm)
        self.llm_embed = nn.Linear(160, config.hidden_size // 2)
        self.prot_llm_bilstm = nn.LSTM(config.hidden_size // 2, config.hidden_size // 2, num_layers=2, bidirectional=True, dropout=config.dropout)

        prot_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=8, dim_feedforward=config.embed_dim * 2, dropout=config.dropout, activation='relu')
        self.prot_transformer = nn.TransformerEncoder(prot_encoder_layer, num_layers=2)
        mol_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=8, dim_feedforward=config.embed_dim * 2, dropout=config.dropout, activation='relu')
        self.mol_transformer = nn.TransformerEncoder(mol_encoder_layer, num_layers=2)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

        self.pmma = PairedMultimodelAttention(config=config, vis=False)
        self.mol_tgca = TargetGuidedCrossAttention(embed_dim=config.embed_dim, num_heads=1)
        self.mol_tgca_norm = nn.LayerNorm(config.embed_dim)
        self.prot_tgca = TargetGuidedCrossAttention(embed_dim=config.embed_dim, num_heads=1)
        self.prot_tgca_norm = nn.LayerNorm(config.embed_dim)
        # mlp
        # self.attn_pooling = Attn_Net_Gated(config.embed_dim, config.embed_dim)
        self.fc1 = nn.Linear(config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

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
    
    def extract_mol_feat(self, mol_g):
        mol_x, mol_edge_index, mol_batch = mol_g.x, mol_g.edge_index, mol_g.batch
        
        mol_feat = self.act(self.mol_conv1(mol_x, mol_edge_index))
        mol_feat = self.act(self.mol_conv2(mol_feat, mol_edge_index))
        mol_feat = self.act(self.mol_conv3(mol_feat, mol_edge_index))

        mol_feat = gep(mol_feat, mol_batch)
        mol_feat = mol_feat.unsqueeze(2)

        mol_feat = self.dropout(self.act(self.mol_fc1(mol_feat)))
        mol_feat = self.dropout(self.mol_fc2(mol_feat))

        mol_feat = self.mol_norm(mol_feat)

        return mol_feat
    
    def extract_prot_feat(self, prot_g):
        prot_x, prot_edge_index, prot_batch = prot_g.x, prot_g.edge_index, prot_g.batch
        
        prot_embed = self.act(self.pro_conv1(prot_x, prot_edge_index))
        prot_embed = self.act(self.pro_conv2(prot_embed, prot_edge_index))
        prot_embed = self.act(self.pro_conv3(prot_embed, prot_edge_index))

        prot_embed = gep(prot_embed, prot_batch)
        prot_embed = prot_embed.unsqueeze(2)

        prot_embed = self.dropout(self.act(self.pro_fc1(prot_embed)))
        prot_embed = self.dropout(self.pro_fc2(prot_embed))

        prot_embed = self.pro_norm(prot_embed)
            
        return prot_embed
    
    def fill_prot_llms(self, prot_llms):
        # prot_llms = graph_pad(prot_llms, 1024)
        prot_llms = torch.stack(prot_llms, dim=0).float()
        prot_llms = self.dropout(self.act(self.llm_fc(prot_llms.permute(0, 2, 1)))).permute(0, 2, 1)
        h = self.llm_embed(prot_llms)
        prot_llms, _ = self.prot_llm_bilstm(h)
        return prot_llms

    def forward(self, mol_g, prot_g, prot_llms):
        prot_feat = self.extract_prot_feat(prot_g)
        mol_feat = self.extract_mol_feat(mol_g)
        prot_llms = self.fill_prot_llms(prot_llms)
        
        mol_embed, self.A_tgca = self.mol_tgca(prot_llms.permute(1, 0, 2), mol_feat.permute(1, 0, 2), mol_feat.permute(1, 0, 2))
        prot_embed, A_tgca = self.prot_tgca(prot_llms.permute(1, 0, 2), prot_feat.permute(1, 0, 2), prot_feat.permute(1, 0, 2))

        prot_embed = self.prot_transformer(prot_embed)
        prot_embed = prot_embed.permute(1, 0, 2)
        # mol_embed = mol_embed + mol_feat
        prot_embed = self.prot_tgca_norm(prot_embed)

        mol_embed = self.mol_transformer(mol_embed)
        mol_embed = mol_embed.permute(1, 0, 2)
        # mol_embed = mol_embed + mol_feat
        mol_embed = self.mol_tgca_norm(mol_embed)
        # mol_embed = torch.concat([mol_embed, mol_feat], dim=1)
        # mol_embed = mol_embed + mol_feat
        # prot_embed = prot_embed.permute(1, 0, 2)
        dt_embed, self.attn_weights, self.guided_attn_weights = self.pmma(prot_embed, mol_embed)

        # interact
        # A_pool, dt_embed = self.attn_pooling(dt_embed)
        # A_pool = torch.transpose(A_pool, -1, -2)
        # dt_embed = torch.bmm(F.softmax(A_pool, dim=-1), dt_embed)
        dt_embed = torch.mean(dt_embed, dim=1)
        dt_embed = self.fc1(dt_embed)
        dt_embed = self.act(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dt_embed = self.fc2(dt_embed)
        dt_embed = self.act(dt_embed)
        dt_embed = self.dropout(dt_embed)
        dta = self.out(dt_embed)
        return dta
    

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
 
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x