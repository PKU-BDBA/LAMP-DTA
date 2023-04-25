#import torch.utils.data as DATA
# from transformers import AdamW, AutoModel, AutoTokenizer
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
#import rdkit
#from rdkit import Chem
#from rdkit.Chem import AllChem
import numpy as np
import torch
from sklearn.utils import shuffle
import torch
import random
from utils import *


def get_test(dataset, cutoff: int):
    msa_path = '/data5/luozc/projects/DTA/LAMP-DTA/GINCM-DTA/data/' + dataset + '/aln'
    contac_path = '/data5/luozc/projects/DTA/LAMP-DTA/GINCM-DTA/data/' + dataset + '/pconsc4'
    test_csv = '../data/' + dataset + '/' + dataset  + '_test_cm.csv'

    df_test_fold = pd.read_csv(test_csv)
    test_drugs, test_prot_keys, test_Y, test_pro, test_pids = list(df_test_fold['compound_iso_smiles']), list(
        df_test_fold['target_original_key']), list(df_test_fold['affinity']), list(df_test_fold['target_sequence']), list(df_test_fold['protein_id'])

    test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)
    compound_iso_smiles = set(test_drugs)
    target_key = set(test_prot_keys)

    proteins = {}
    for i in range(df_test_fold.shape[0]):
        proteins[test_prot_keys[i]] = test_pro[i]
        
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', drugs=test_drugs, target_keys=test_prot_keys,
                               y=test_Y, smile_graphs=smile_graph, target_graphs=target_graph, cutoff=cutoff, pid_list=test_pids)
    return test_dataset

def get_train_valid(dataset,fold, cutoff: int):
    # load dataset and CM
    msa_path = '/data5/luozc/projects/DTA/LAMP-DTA/GINCM-DTA/data/' + dataset + '/aln'
    contac_path = '/data5/luozc/projects/DTA/LAMP-DTA/GINCM-DTA/data/' + dataset + '/pconsc4'

    train_csv = '../data/' + dataset + '/' + dataset + '_train_fold_' + str(fold) + '_cm.csv'
    valid_csv = '../data/' + dataset + '/' + dataset + '_valid_fold_' + str(fold) + '_cm.csv'
    df_train_fold = pd.read_csv(train_csv)
    df_valid_fold = pd.read_csv(valid_csv)

    train_drugs, train_prot_keys, train_Y, train_prot_seqs, train_pids = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_original_key']), list(df_train_fold['affinity']) , list(df_train_fold['target_sequence']), list(df_train_fold['protein_id'])
    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)

    valid_drugs, valid_prot_keys, valid_Y, valid_prot_seqs, valid_pids = list(df_valid_fold['compound_iso_smiles']), list(
        df_valid_fold['target_original_key']), list(df_valid_fold['affinity']), list(df_valid_fold['target_sequence']), list(df_valid_fold['protein_id'])
    valid_drugs, valid_prot_keys, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prot_keys), np.asarray(valid_Y)

    compound_iso_smiles = set.union(set(valid_drugs),set(train_drugs))
    target_keys = set.union(set(valid_prot_keys),set(train_prot_keys))

    proteins = {}
    for i in range(df_train_fold.shape[0]):
        proteins[train_prot_keys[i]] = train_prot_seqs[i]
    for i in range(df_valid_fold.shape[0]):
        proteins[valid_prot_keys[i]] = valid_prot_seqs[i]

    # create smile graph
    smile_graphs = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graphs[smile] = g

    # create target graph
    target_graphs = {}
    for key in target_keys:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graphs[key] = g

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graphs), len(target_graphs))
    if len(smile_graphs) == 0 or len(target_graphs) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    if cutoff is not None:
        train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', drugs=train_drugs, target_keys=train_prot_keys,
                                y=train_Y, smile_graphs=smile_graphs, target_graphs=target_graphs, cutoff=cutoff * 4, pid_list=train_pids)
    else:
        train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', drugs=train_drugs, target_keys=train_prot_keys,
                                y=train_Y, smile_graphs=smile_graphs, target_graphs=target_graphs, cutoff=None, pid_list=train_pids)
    valid_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', drugs=valid_drugs, target_keys=valid_prot_keys, 
                               y=valid_Y, smile_graphs=smile_graphs, target_graphs=target_graphs, cutoff=cutoff, pid_list=valid_pids)

    return train_dataset, valid_dataset


class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 drugs=None, y=None, transform=None,
                 pre_transform=None, smile_graphs=None, target_keys=None, target_graphs=None, cutoff: int =None, pid_list=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.cutoff = cutoff
        self.process(drugs, target_keys, y, smile_graphs, target_graphs, pid_list)

    @property
    def raw_file_names(self): # FIXME: Add fold info
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self): # FIXME: Add fold info
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drugs, target_keys, y, smile_graphs, target_graphs, pid_list):
        assert (len(drugs) == len(target_keys) and len(drugs) == len(y)), 'The three lists must be the same length!'
        mol_list = []
        prot_list = []
        task_len = len(drugs)
        for i in range(task_len):
            smiles = drugs[i]
            tar_key = target_keys[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            if smile_graphs.__contains__(smiles) is False:
                continue
            c_size, feats, edge_index = smile_graphs[smiles]
            tar_size, tar_feats, tar_edge_index = target_graphs[tar_key]
            mol = DATA.Data(x=torch.Tensor(np.array(feats)),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            mol.__setitem__('c_size', torch.LongTensor([c_size]))

            prot = DATA.Data(x=torch.Tensor(np.array(tar_feats)),
                                    edge_index=torch.LongTensor(tar_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            prot.__setitem__('target_size', torch.LongTensor([tar_size]))
            mol_list.append(mol)
            prot_list.append(prot)

        if self.pre_filter is not None:
            mol_list = [mol for mol in mol_list if self.pre_filter(mol)]
            prot_list = [prot for prot in prot_list if self.pre_filter(prot)]
        if self.pre_transform is not None:
            mol_list = [self.pre_transform(mol) for mol in mol_list]
            prot_list = [self.pre_transform(prot) for prot in prot_list]
        self.mols = mol_list
        self.prots = prot_list
        self.pids = pid_list

    def __len__(self):
        if self.cutoff is not None:
            return min(len(self.mols), self.cutoff)
        return len(self.mols)

    def __getitem__(self, idx):
        return self.mols[idx], self.prots[idx], self.pids[idx]

