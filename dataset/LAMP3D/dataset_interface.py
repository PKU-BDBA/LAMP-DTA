from abc import ABC, abstractmethod
from dataset.lamp_dataset import *
import importlib
import pickle
import dgl
import pandas as pd
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class BaseKFoldDataModule(pl.LightningDataModule, ABC):

    @abstractmethod
    def setup_fold_index(self, fold_index: int, is_test: bool):
        pass

class DInterface(BaseKFoldDataModule):
    train_dataset: Optional[Dataset] = None
    valid_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None

    def __init__(self, config):
        super().__init__()
        self.config = config

        # DataFrame for selection based on fold index
        self.dataset = config.dataset
        self.pid2llm = np.load(f'{self.dataset}.npz',allow_pickle=True)['dict'][()]
        with open(f'{self.dataset}_site.pkl', 'rb') as fp:
            self.prot_key2prot_graph = pickle.load(fp)

        # Loader configs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.load_data_module()

    def setup(self, stage=None):
        pass

    def setup_fold_index(self, fold_index: int, is_test: bool=False) -> None:
        if is_test == False:
            # self.train_dataset, self.valid_dataset = self.get_train_valid(dataset=self.dataset , fold=fold_index)
            self.train_dataset = self.get_train_light(dataset=self.dataset , fold=fold_index)
            # self.test_dataset = self.get_test(dataset=self.dataset)
            self.test_dataset = self.get_test_light(dataset=self.dataset)
        else:
            # self.test_dataset = self.get_test(dataset=self.dataset)
            pass

    # Prot seq -> LLM, use pid2llm dict, exist for sure
    def pid_to_llm(self, pid):
        return torch.from_numpy(self.pid2llm[pid]).squeeze()

    # Prot seq -> Hybrid graphs, seq2[AF | pdb_id[: 4]], then use site.pkl, may not exist(no [pdb_id] or None)
    def prot_key_to_graph(self, prot_key):
        return self.prot_key2prot_graph[prot_key]

    # def get_finetune(self, dataset):
    #     msa_path = 'data/' + dataset + '/aln'
    #     contac_path = 'data/' + dataset + '/pconsc4'
    #     finetune_csv = 'data/'+ 'covid' + '.csv'

    #     df_finetune_fold = pd.read_csv(finetune_csv)
    #     finetune_drugs, finetune_prot_keys, finetune_Y, finetune_pro = list(df_finetune_fold['compound_iso_smiles']), list(
    #         df_finetune_fold['target_key']), list(df_finetune_fold['affinity']), list(df_finetune_fold['target_sequence'])

    #     finetune_drugs, finetune_prot_keys, finetune_Y = np.asarray(finetune_drugs), np.asarray(finetune_prot_keys), np.asarray(finetune_Y)
    #     compound_iso_smiles = set(finetune_drugs)
    #     target_key = set(finetune_prot_keys)

    #     proteins = {}
    #     for i in range(df_finetune_fold.shape[0]):
    #         proteins[finetune_prot_keys[i]] = finetune_pro[i]
    #     smile_graph = {}
    #     for smile in compound_iso_smiles:
    #         g = smile_to_graph(smile)
    #         if g is None:
    #             continue
    #         smile_graph[smile] = g

    #     target_graph = {}
    #     for key in target_key:
    #         if not valid_target(key, dataset):  # ensure the contact and aln files exists
    #             continue
    #         g = target_to_graph(key, proteins[key], contac_path, msa_path)
    #         target_graph[key] = g

    #     # count the number of  proteins with aln and contact files
    #     print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    #     if len(smile_graph) == 0 or len(target_graph) == 0:
    #         raise Exception('no protein or drug, run the script for datasets preparation.')

    #     finetune_dataset = LAMPDataset(root='data', dataset=dataset + '_' + 'train', drugs=finetune_drugs, target_keys=finetune_prot_keys,
    #                             y=finetune_Y, smile_graphs=smile_graph, target_graphs=target_graph)
    #     return finetune_dataset

    # def get_test(self, dataset):
    #     test_csv = f'data/{dataset}/{dataset}_test_clr.csv'
    #     df_test_fold = pd.read_csv(test_csv)

    #     mols = set(df_test_fold.compound_iso_smiles.to_list())
    #     tgts = set(df_test_fold.prot_key.to_list())
    #     print('effective drugs,effective prot:', len(mols), len(tgts))
    #     if len(mols) == 0 or len(tgts) == 0:
    #         raise Exception('no protein or drug, run the script for datasets preparation.')

    #     ys, mol_gs, tgt_gs, tgt_llms = [], [], [], []
    #     for idx, row in df_test_fold.iterrows():
    #         y, cis, pid, key = row.affinity, row.compound_iso_smiles, row.protein_id, row.prot_key
    #         ys.append(y)
    #         mol_gs.append(smile_to_graph(cis))
    #         tgt_llms.append(self.pid_to_llm(pid))
    #         tgt_gs.append(self.prot_key_to_graph(key))

    #     test_dataset = self.data_module(self.config, 'test', ys, mol_gs, tgt_gs, tgt_llms)
    #     return test_dataset
    
    def get_test_light(self, dataset):
        test_dataset = self.data_module(dataset, 'test', '')
        return test_dataset

    # def get_train_valid(self, dataset, fold):
    #     train_csv = f'data/{dataset}/{dataset}_train_fold_{fold}_clr.csv'
    #     valid_csv = f'data/{dataset}/{dataset}_valid_fold_{fold}_clr.csv'
    #     df_train_fold = pd.read_csv(train_csv)
    #     df_valid_fold = pd.read_csv(valid_csv)

    #     mols = set.union(set(df_train_fold.compound_iso_smiles.to_list()), set(df_valid_fold.compound_iso_smiles.to_list()))
    #     tgts = set.union(set(df_train_fold.prot_key.to_list()), set(df_valid_fold.prot_key.to_list()))
    #     print('effective drugs,effective prot:', len(mols), len(tgts))
    #     if len(mols) == 0 or len(tgts) == 0:
    #         raise Exception('no protein or drug, run the script for datasets preparation.')

    #     ys, mol_gs, tgt_gs, tgt_llms = [], [], [], []
    #     for idx, row in df_train_fold.iterrows():
    #         y, cis, pid, key = row.affinity, row.compound_iso_smiles, row.protein_id, row.prot_key
    #         ys.append(y)
    #         mol_gs.append(smile_to_graph(cis))
    #         tgt_llms.append(self.pid_to_llm(pid))
    #         tgt_gs.append(self.prot_key_to_graph(key))
        
    #     train_dataset = self.data_module(self.config, 'train', ys, mol_gs, tgt_gs, tgt_llms)

    #     ys, mol_gs, tgt_gs, tgt_llms = [], [], [], []
    #     for idx, row in df_valid_fold.iterrows():
    #         y, cis, pid, key = row.affinity, row.compound_iso_smiles, row.protein_id, row.prot_key
    #         ys.append(y)
    #         mol_gs.append(smile_to_graph(cis))
    #         tgt_llms.append(self.pid_to_llm(pid))
    #         tgt_gs.append(self.prot_key_to_graph(key))
        
    #     valid_dataset = self.data_module(self.config, 'valid', ys, mol_gs, tgt_gs, tgt_llms)

    #     return train_dataset, valid_dataset
    
    def get_train_light(self, dataset, fold):
        train_dataset = self.data_module(dataset, 'train', fold)
        return train_dataset

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate, pin_memory=True)
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate_raw, pin_memory=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate, pin_memory=True)
    
    def val_dataloader(self):
        # val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate, pin_memory=True)
        # test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate, num_workers=self.num_workers, pin_memory=True)

        # val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_raw, pin_memory=True)
        # test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_raw, num_workers=self.num_workers, pin_memory=True)

        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate, num_workers=self.num_workers, pin_memory=True)
        return test_loader
    
    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate, num_workers=self.num_workers, pin_memory=True)
        return None

    def load_data_module(self):
        try:
            # self.data_module = getattr(importlib.import_module(
            #     '.lamp_dataset', package=__package__), 'LAMPDataset')
            self.data_module = getattr(importlib.import_module(
                '.lamp_dataset', package=__package__), 'LAMPDatasetLight')
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.')