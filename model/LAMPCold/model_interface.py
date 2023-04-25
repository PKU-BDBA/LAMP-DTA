import inspect
from abc import ABC, abstractmethod
import torch
from Utils import *
import importlib
import pandas as pd
from collections import defaultdict
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist
from tqdm import tqdm

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class MInterface(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_train_epochs = config.max_epochs
        self.log_dir = Path(os.path.join(config.log_dir, 
                                         config.date, 
                                         f'ld_{config.lr}_wd_{config.wd}'))
        self.printed = ''
        self.log_path = None

        self.test_labels = []
        self.test_preds = []

        self.best_test_ci = 0
        self.best_test_rm2 = 0
        self.best_test_pr = 0
        self.best_test_sp = 0

        self.best_test_ci_epoch = -1
        self.best_test_rm2_epoch = -1
        self.best_test_pr_epoch = -1
        self.best_test_sp_epoch = -1

        self.lr = config.lr
        self.wd = config.wd
        self.load_model()
        self.configure_loss()

    def forward(self, mol_g, prot_g, prot_llm): # XXX
        return self.model(mol_g, prot_g, prot_llm)
    
    def setup_fold_index(self, fold_index: int, is_test: bool=False):
        self.fold = fold_index
        if self.global_rank == 0:
            self.printed += f"task={self.config.model}_{self.config.dataset}, fold=cold" + '\n'

        if is_test == False:
            if self.global_rank == 0:
                pass # print something
        else:
            if self.global_rank == 0:
                pass # print something

    def on_train_start(self) -> None:
        if self.log_path is None:
            log_dir = self.log_dir
            max_vn = max([vn.split('_')[1] for vn in os.listdir(log_dir) if vn.startswith('version_')])
            self.log_path = log_dir / f'version_{max_vn}'

    def training_step(self, batch, batch_idx):
        label, mol_g, prot_g, prot_llm = batch
        output = self(mol_g, prot_g, prot_llm)
        label = label.view(-1)
        loss = self.loss_function(output, label)

        self.log('train_mse', loss, prog_bar=False, sync_dist=True, on_epoch=True)

        if self.global_rank == 0:
            pass
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label, mol_g, prot_g, prot_llm = batch
        output = self(mol_g, prot_g, prot_llm)
        label = label.view(-1)
        loss = self.loss_function(output, label)

        self.log("test_mse", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        
        self.test_labels.append(label)
        self.test_preds.append(output)

    def on_validation_epoch_end(self) -> None:
        all_test_labels = torch.concat(self.test_labels).cpu().numpy().flatten()
        all_test_preds = torch.concat(self.test_preds).cpu().numpy().flatten()

        test_ci = get_ci(all_test_labels, all_test_preds)
        self.log('test_ci', test_ci, prog_bar=False, rank_zero_only=True)
        test_rm2 = get_rm2(all_test_labels, all_test_preds)
        self.log('test_rm2', test_rm2, prog_bar=False, rank_zero_only=True)
        test_pr = get_pearson(all_test_labels, all_test_preds)
        self.log('test_pr', test_pr, prog_bar=False, rank_zero_only=True)
        test_sp = get_spearman(all_test_labels, all_test_preds)
        self.log('test_sp', test_sp, prog_bar=False, rank_zero_only=True)
        self.test_labels.clear()
        self.test_preds.clear()

        if self.global_rank == 0:
            if test_ci > self.best_test_ci:
                self.best_test_ci = test_ci
                self.best_test_ci_epoch = self.current_epoch
                self.printed += f'test ci improved at epoch {self.best_test_ci_epoch}, best test ci: {self.best_test_ci}' + '\n'
            if test_rm2 > self.best_test_rm2:
                self.best_test_rm2 = test_rm2
                self.best_test_rm2_epoch = self.current_epoch
                self.printed += f'test rm2 improved at epoch {self.best_test_rm2_epoch}, best test rm2: {self.best_test_rm2}' + '\n'
            if test_pr > self.best_test_pr:
                self.best_test_pr = test_pr
                self.best_test_pr_epoch = self.current_epoch
                self.printed += f'test pr improved at epoch {self.best_test_pr_epoch}, best test pr: {self.best_test_pr}' + '\n'
            if test_sp > self.best_test_sp:
                self.best_test_sp = test_sp
                self.best_test_sp_epoch = self.current_epoch
                self.printed += f'test sp improved at epoch {self.best_test_sp_epoch}, best test sp: {self.best_test_sp}' + '\n'
    
    def on_validation_end(self) -> None:
        pass

    def on_train_end(self) -> None:
        if self.global_rank == 0:
            ensure_path(f'{self.log_path}')
            Path(f'{self.log_path}/printed.txt').write_text(self.printed)

    def on_test_start(self):
        pass

    # def test_step(self, batch, batch_idx):
    #     label, mol_g, prot_g, prot_llm = batch
    #     output = self(mol_g, prot_g, prot_llm)
        
    #     self.test_labels.append(label)
    #     self.test_preds.append(output)

    # def on_test_epoch_end(self) -> None:
    #     all_test_labels = torch.concat(self.test_labels).cpu().numpy().flatten()
    #     all_test_preds = torch.concat(self.test_preds).cpu().numpy().flatten()

    #     test_mse = get_mse(all_test_labels, all_test_preds)
    #     self.log('test_mse', test_mse, prog_bar=True, sync_dist=True)
    #     test_ci = get_ci(all_test_labels, all_test_preds)
    #     self.log('test_ci', test_ci, prog_bar=True, sync_dist=True)
    #     test_rm2 = get_rm2(all_test_labels, all_test_preds)
    #     self.log('test_rm2', test_rm2, prog_bar=True, sync_dist=True)
    #     test_pr = get_pearson(all_test_labels, all_test_preds)
    #     self.log('test_pr', test_pr, prog_bar=True, sync_dist=True)
    #     test_sp = get_spearman(all_test_labels, all_test_preds)
    #     self.log('test_sp', test_sp, prog_bar=True, sync_dist=True)
    #     self.test_labels.clear()
    #     self.test_preds.clear()
    #     if self.global_rank == 0:
    #         if test_mse < self.best_test_mse:
    #             self.best_test_mse = test_mse
    #             self.best_test_mse_epoch = self.current_epoch
    #             print(f'test mse improved at epoch {self.best_test_mse_epoch}, best test mse: {self.best_test_mse}')

    #         if test_ci > self.best_test_ci:
    #             self.best_test_ci = test_ci
    #             self.best_test_ci_epoch = self.current_epoch
    #             print(f'test ci improved at epoch {self.best_test_ci_epoch}, best test ci: {self.best_test_ci}')

    #         if test_rm2 > self.best_test_rm2:
    #             self.best_test_rm2 = test_rm2
    #             self.best_test_rm2_epoch = self.current_epoch
    #             print(f'test rm2 improved at epoch {self.best_test_rm2_epoch}, best test rm2: {self.best_test_rm2}')

    #         if test_pr > self.best_test_pr:
    #             self.best_test_pr = test_pr
    #             self.best_test_pr_epoch = self.current_epoch
    #             print(f'test pr improved at epoch {self.best_test_pr_epoch}, best test pr: {self.best_test_pr}')

    #         if test_sp > self.best_test_sp:
    #             self.best_test_sp = test_sp
    #             self.best_test_sp_epoch = self.current_epoch
    #             print(f'test sp improved at epoch {self.best_test_sp_epoch}, best test sp: {self.best_test_sp}')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.wd
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=self.num_train_epochs,
            max_lr=self.lr,
            min_lr=1e-8,
            warmup_steps=int(self.num_train_epochs * 0.1)
        )

        return [optimizer], [scheduler]

    def configure_loss(self):
        self.loss_function = torch.nn.MSELoss()

    def load_model(self):
        try:
            self.Model = getattr(importlib.import_module(
                '.lamp_model', package=__package__), 'LAMPCM')
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name data.')
        
        self.model = self.Model(self.config)
