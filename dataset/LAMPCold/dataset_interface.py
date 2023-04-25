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
import random
random.seed(42)

class BaseColdDataModule(pl.LightningDataModule, ABC):

    @abstractmethod
    def setup_cold_way(self, fold_index: int, is_test: bool):
        pass

class DInterface(BaseColdDataModule):
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

    def setup_cold_way(self, cold_way: str, is_test: bool=False) -> None:
        if is_test == False:
            # self.train_dataset, self.valid_dataset = self.get_train_valid(dataset=self.dataset , fold=fold_index)
            self.train_dataset, self.test_dataset = self.get_train_test_cold(dataset=self.dataset, cold_way=cold_way)
        else:
            # self.test_dataset = self.get_test(dataset=self.dataset)
            pass

    # Prot seq -> LLM, use pid2llm dict, exist for sure
    def pid_to_llm(self, pid):
        return torch.from_numpy(self.pid2llm[pid]).squeeze()

    # Prot seq -> Hybrid graphs, seq2[AF | pdb_id[: 4]], then use site.pkl, may not exist(no [pdb_id] or None)
    def prot_key_to_graph(self, prot_key):
        return self.prot_key2prot_graph[prot_key]
    
    def get_train_test_cold(self, dataset, cold_way):
        df_cold = pd.read_csv(f'data/{dataset}_cold_cm.csv')
        if cold_way == 'p':
            pid_len = max(df_cold.protein_id)
            test_pid_candidates = random.sample(list(range(pid_len)), int(pid_len / 5))

            test_idxs = []
            train_idxs = []
            for idx in df_cold.index:
                if df_cold.iloc[idx].protein_id in test_pid_candidates:
                    test_idxs += [idx]
                else:
                    train_idxs += [idx]
            
            df_train_cold = df_cold.iloc[train_idxs]
            df_test_cold = df_cold.iloc[test_idxs]

        elif cold_way == 'd':
            did_len = max(df_cold.drug_id)
            test_did_candidates = random.sample(list(range(did_len)), int(did_len / 5))

            test_idxs = []
            train_idxs = []
            for idx in df_cold.index:
                if df_cold.iloc[idx].drug_id in test_did_candidates:
                    test_idxs += [idx]
                else:
                    train_idxs += [idx]
            
            df_train_cold = df_cold.iloc[train_idxs]
            df_test_cold = df_cold.iloc[test_idxs]
            
        else:
            pid_len = max(df_cold.protein_id)
            did_len = max(df_cold.drug_id)
            test_pid_candidates = random.sample(list(range(pid_len)), int(pid_len / 5))
            test_did_candidates = random.sample(list(range(did_len)), int(did_len / 5))

            test_idxs = []
            train_idxs = []
            for idx in df_cold.index:
                row = df_cold.iloc[idx]
                if (row.drug_id in test_did_candidates) and (row.protein_id in test_pid_candidates):
                    test_idxs += [idx]
                elif (row.drug_id not in test_did_candidates) and (row.protein_id not in test_pid_candidates):
                    train_idxs += [idx]
            
            df_train_cold = df_cold.iloc[train_idxs]
            df_test_cold = df_cold.iloc[test_idxs]

        train_dataset = self.data_module(dataset, 'train', df_train_cold)
        test_dataset = self.data_module(dataset, 'test', df_test_cold)

        return train_dataset, test_dataset

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate, pin_memory=True)
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate_raw, pin_memory=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate_cm, pin_memory=True)
    
    def val_dataloader(self):
        # val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate, pin_memory=True)
        # test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate, num_workers=self.num_workers, pin_memory=True)

        # val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_raw, pin_memory=True)
        # test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_raw, num_workers=self.num_workers, pin_memory=True)

        # val_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=collate_cm, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_cm, num_workers=self.num_workers, pin_memory=True)
        return test_loader
    
    def test_dataloader(self):
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate, num_workers=self.num_workers, pin_memory=True)
        return None

    def load_data_module(self):
        try:
            # self.data_module = getattr(importlib.import_module(
            #     '.lamp_dataset', package=__package__), 'LAMPDataset')
            self.data_module = getattr(importlib.import_module(
                '.lamp_dataset', package=__package__), 'LAMPDatasetCold')
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.')