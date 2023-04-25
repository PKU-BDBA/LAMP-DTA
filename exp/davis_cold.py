import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
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

from model.LAMPCold.model_interface import MInterface
from dataset.LAMPCold.dataset_interface import DInterface
from Utils import *


class ColdExp(object):
    def __init__(self, args, model_class, dataset_class, config):
        super().__init__()
        # self.task = config['task']
        self.log_dir = config.log_dir
        self.date = config.date

        self.args = args
        self.config = config
        self.model_class = model_class
        self.data_class = dataset_class

        # params to load the checkpoint
    
    def run(self):
        trainer = Trainer.from_argparse_args(self.args)
        data_module = self.data_class(self.config)
        model_module = self.model_class(self.config)
        logger = CSVLogger(save_dir=self.log_dir, name=os.path.join(self.date, f'ld_{self.config.lr}_wd_{self.config.wd}'))
        trainer.logger = logger

        # setup train/eval data in this fold
        if self.args.train:
            model_module.setup_fold_index(0)
            data_module.setup_cold_way('Both')
            trainer.fit(model_module, datamodule=data_module)

        if self.args.test:
            dist.barrier() # HACK

            # setup test data in this fold
            model_module.setup_fold_index(0)
            data_module.setup_cold_way('Both', is_test=True)
            trainer.test(datamodule=data_module)
            # on_test_end


def args2config(args):
    pl.seed_everything(args.seed)

    args.dataset_class = DInterface
    args.model_class = MInterface
    args.train = True
    args.test = False

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

    cold_exp = ColdExp(args, args.model_class, args.dataset_class, config)
    cold_exp.run()


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
    parser.add_argument('--dataset', default='davis', type=str)
    parser.add_argument('--model', default='lamp_cm', type=str)

    parser.add_argument_group(title="pl.Trainer args")
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=1000)

    args = parser.parse_args()
    
    args.log_dir = f'{args.model}_{args.dataset}_cold_logs'

    main(args)