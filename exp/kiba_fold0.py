import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from copy import deepcopy
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import sys
sys.path.append(str(Path('.').absolute()))
import time
import datetime

import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from model.model_interface import MInterface
from dataset.dataset_interface import DInterface
from Utils import *


def load_callbacks(): # FIXME
    # ckpt_dir = Path(ckpt_dir)
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='test_mse',
        mode='min',
        patience=100,
        min_delta=0.001
    ))
    # savers = []

    progress_bar = plc.RichProgressBar(
        refresh_rate=1,
        theme=RichProgressBarTheme(
            description='khaki3',
            progress_bar='dark_turquoise',
            progress_bar_finished='khaki3',
            progress_bar_pulse='#6206E0',
            batch_progress='khaki3',
            time='grey66',
            processing_speed='grey66',
            metrics='salmon1',
        ),
        # leave=True,
    )

    callbacks.append(progress_bar)
    
    # savers.append('best-valid_loss')
    callbacks.append(plc.ModelCheckpoint(
        monitor='test_mse',
        filename='min_{test_mse:.3f}',
        save_top_k=1,
        mode='min',
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='test_ci',
        filename='max_{test_ci:.3f}',
        save_top_k=1,
        mode='max',
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='test_rm2',
        filename='max_{test_rm2:.3f}',
        save_top_k=1,
        mode='max',
    ))

    # savers.append('last-epoch')
    callbacks.append(plc.ModelCheckpoint(
        # dirpath=ckpt_dir,
        filename='last_{epoch:02d}',
        save_last=True,
        # save_weights_only=True,
        # save_on_train_epoch_end=True,
    ))

    callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch',
    ))

    return callbacks


class KFoldExp(object):
    def __init__(self, args, model_class, dataset_class, config):
        super().__init__()
        self.folds = args.folds
        # self.task = config['task']
        self.log_dir = config.log_dir
        self.date = config.date

        self.args = args
        self.config = config
        self.model_class = model_class
        self.data_class = dataset_class

        # params to load the checkpoint
    
    def run(self):
        for fold_index in self.folds:
            trainer = Trainer.from_argparse_args(self.args)
            data_module = self.data_class(self.config)
            model_module = self.model_class(self.config)
            logger = CSVLogger(save_dir=self.log_dir, name=os.path.join(self.date, f'ld_{self.config.lr}_wd_{self.config.wd}', f'fold_{fold_index}'))
            trainer.logger = logger

            # setup train/eval data in this fold
            if self.args.train:
                model_module.setup_fold_index(fold_index)
                data_module.setup_fold_index(fold_index)
                trainer.fit(model_module, datamodule=data_module)

            if self.args.test:
                dist.barrier() # HACK

                # setup test data in this fold
                model_module.setup_fold_index(fold_index)
                data_module.setup_fold_index(fold_index, is_test=True)
                trainer.test(datamodule=data_module)
                # on_test_end


def args2config(args):
    pl.seed_everything(args.seed)

    args.dataset_class = DInterface
    args.model_class = MInterface
    args.train = True
    args.test = False
    args.folds = list(map(int, args.folds))
    
    args.num_workers = 4
    # args.num_workers = 0

    return args


# Training the model
def main(args):
    config = args2config(args)

    args.callbacks = load_callbacks()
    args.accelerator = 'auto'
    args.strategy = 'ddp'
    args.num_sanity_val_steps = 0
    args.log_every_n_steps = 5

    k_fold_exp = KFoldExp(args, args.model_class, args.dataset_class, config)
    k_fold_exp.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=64, type=int) # XXX
    parser.add_argument('--seed', default=42, type=int)

    # Model setting
    parser.add_argument('--n_output', default=1, type=int)
    parser.add_argument('--num_features_prot', default=54, type=int)
    parser.add_argument('--num_features_mol', default=82, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    # Training Info
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=5e-2, type=float)
    parser.add_argument('--date', default=time.strftime("%Y-%m-%d", time.localtime()), type=str)
    parser.add_argument('--dataset', default='kiba', type=str)
    parser.add_argument('--model', default='lamp_cm', type=str)
    parser.add_argument('--folds', default='0', type=str) # Tid: use multi-fold(e.g: '012') would run extremely slow

    parser.add_argument_group(title="pl.Trainer args")
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=1000)

    args = parser.parse_args()
    
    args.log_dir = f'{args.model}_{args.dataset}_fold[{args.folds}]_logs'

    main(args)