from Utils import *
import os
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch_geometric import data as DATA
import numpy as np
import torch
from sklearn.utils import shuffle
import torch
import datetime
from log.train_logger import TrainLogger

def train_eval(
    model, 
    optimizer,
    scheduler, 
    train_loader, 
    valid_loader,
    test_loader, 
    epochs=2, 
    dataset='davis',
    gpu=0,
    fold=None,
):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('-----Training-----')
    starttime = datetime.datetime.now()
    last_epoch_time = starttime

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=dataset,
        save_model=True,
        fold=fold,
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1
    for epoch in range(epochs):
        endtime = datetime.datetime.now()
        print('total run time: ', endtime - starttime)
        print('last epoch run time: ', endtime - last_epoch_time)
        last_epoch_time = endtime
        print('Epoch', epoch)
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        val1 = get_mse(G, P)
        if test_loader is not None:
            print('predicting for test data')
            G, P = predicting(model, device, test_loader)
            val2 = get_mse(G, P)
            if val2 < best_test_mse:
                best_test_mse = val2
                best_test_epoch = epoch + 1
                msg = f"test mse has improved at epoch {best_test_epoch}, test mse: {best_test_mse}"
                logger.info(msg)
        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            if test_loader is not None:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse, "test mse:", val2)
                msg = "epoch-%d, loss-%.4f, test_loss-%.4f" % (epoch, val1, val2)
            else:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
                msg = "epoch-%d, loss-%.4f" % (epoch, val1)
            model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
            torch.save(model.state_dict(), model_path)
            print("model has been saved to %s." % (model_path))
        else:
            if test_loader is not None:
                print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse,
                      "Best test at:", best_test_epoch, '; best_test_mse', best_test_mse)
            else:
                print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)
        scheduler.step()
        #print(optimizer.state_dict()['param_groups'][0]['lr'])


@dataclass
class LAMPDataset(Dataset):
    config: object
    stage: str
    ys: list
    mol_gs: list
    tgt_gs: list
    tgt_llms: list

    def __post_init__(self):
        self.process(self.ys, self.mol_gs, self.tgt_gs, self.tgt_llms)

    def process(self, ys, mol_gs, tgt_gs, tgt_llms):
        assert (len(ys) == len(mol_gs) and len(mol_gs) == len(tgt_gs)), 'The three lists must be the same length!'
        mol_g_list = []
        prot_g_list = []
        prot_llm_list = []
        task_len = len(ys)
        for i in range(task_len):
            c_size, feats, edge_index = mol_gs[i]

            # group mol's graph
            mol_g = DATA.Data(x=torch.Tensor(np.array(feats)), # XXX
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            mol_g.__setitem__('c_size', torch.LongTensor([c_size]))

            # group prot's graphs
            prot_g = tgt_gs[i]

            # group prot's llm TOOD
            prot_llm = tgt_llms[i]

            mol_g_list.append(mol_g)
            prot_g_list.append(prot_g)
            prot_llm_list.append(prot_llm)

        self.labels = torch.from_numpy(np.array(ys)).float()
        self.drug_gs = mol_g_list
        self.prot_gs = prot_g_list
        self.prot_llms = prot_llm_list

    def __len__(self):
        return len(self.labels)
        # return 200

    def __getitem__(self, idx):
        return self.labels[idx], self.drug_gs[idx], self.prot_gs[idx], self.prot_llms[idx]
    

@dataclass
class LAMPDatasetLight(Dataset):
    dataset: str
    stage: str
    fold: str

    def __post_init__(self):
        if self.stage == 'test':
            data_path = f'data/{self.dataset}/{self.dataset}_test_clr.csv'
        else:
            data_path = f'data/{self.dataset}/{self.dataset}_{self.stage}_fold_{self.fold}_clr.csv'
        data = pd.read_csv(data_path, low_memory=False)

        self.data = data[~data['affinity'].isna()]

        self.pid2llm = np.load(f'{self.dataset}.npz',allow_pickle=True)['dict'][()]
        with open(f'{self.dataset}_site.pkl', 'rb') as fp:
            self.prot_key2prot_graph = pickle.load(fp)

        self.process()

    def pid_to_llm(self, pid):
        return torch.from_numpy(self.pid2llm[pid]).squeeze()
    
    def prot_key_to_graph(self, prot_key):
        return self.prot_key2prot_graph[prot_key]

    def process(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        label, cis, pid, key = (
            row.affinity,
            row.compound_iso_smiles,
            row.protein_id,
            row.prot_key,
        )
        c_size, feats, edge_index = smile_to_graph(cis)
        drug_g = DATA.Data(x=torch.Tensor(np.array(feats)),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0))
        drug_g.__setitem__('c_size', torch.LongTensor([c_size]))

        return torch.FloatTensor([label]), drug_g, self.prot_key_to_graph(key), self.pid_to_llm(pid)