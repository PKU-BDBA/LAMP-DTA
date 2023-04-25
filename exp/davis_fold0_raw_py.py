import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from copy import deepcopy
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import sys
sys.path.append(str(Path('.').absolute()))
import time
import datetime
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torch
from model.lamp_model import LAMP

from Utils import *
gpu = 0
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

dataset = 'davis'

pid2llm = np.load(f'{dataset}.npz',allow_pickle=True)['dict'][()]
def pid_to_llm(pid):
    return torch.from_numpy(pid2llm[pid]).squeeze()

with open(f'{dataset}_site.pkl', 'rb') as fp:
    prot_key2prot_graph = pickle.load(fp)
def prot_key_to_graph(prot_key):
    return prot_key2prot_graph[prot_key]

def args2config(args):
    return args

def train_eval(
    model, 
    optimizer,
    scheduler, 
    df_train, 
    df_valid,
    df_test, 
    config,
    epochs=2, 
):
    model.to(device)
    print('-----Training-----')
    starttime = datetime.datetime.now()
    last_epoch_time = starttime

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

        # train(model, device, df_train, optimizer, epoch + 1)
        print('Training on {} samples...'.format(len(df_train)))
        model.train()
        LOG_INTERVAL = 130
        # TRAIN_BATCH_SIZE = 256
        BATCH_SIZE = 8
        loss_fn = torch.nn.MSELoss()
        for batch_idx, i in enumerate(tqdm(range(0, len(df_train), BATCH_SIZE))):
            if i + BATCH_SIZE < len(df_train):
                df_sub = df_train[i: i + BATCH_SIZE]
            else:
                df_sub = df_train
            labels = [torch.FloatTensor([y]).to(device) for y in df_sub.affinity]
            mol_gs = [*map(smile_to_graph, df_sub.compound_iso_smiles)]
            tgt_gs = [prot_key_to_graph(key).to(device) for key in df_sub.prot_key]
            tgt_llms = [pid_to_llm(pid).to(device) for pid in df_sub.protein_id]

            drug_gs = []
            for mol_g in mol_gs:
                c_size, feats, edge_index = mol_g

                mol_g = DATA.Data(x=torch.Tensor(np.array(feats)),
                                  edge_index=torch.LongTensor(edge_index).transpose(1, 0))
                mol_g.__setitem__('c_size', torch.LongTensor([c_size]))
                drug_gs.append(mol_g.to(device))

            optimizer.zero_grad()
            output = model(Batch.from_data_list(drug_gs), tgt_gs, tgt_llms).to(device)
            loss = loss_fn(output, torch.stack(labels, 0).view(-1).to(device))
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * BATCH_SIZE,
                    len(df_train),
                    100. * batch_idx / (len(df_train) // BATCH_SIZE + (len(df_train) % BATCH_SIZE > 0)),
                    loss.item()))
                
        print('predicting for valid data')

        # G, P = predicting(model, device, df_valid)
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        print('Make prediction for {} samples...'.format(len(df_valid)))
        with torch.no_grad():
            for batch_idx, i in enumerate(tqdm(range(0, len(df_valid), BATCH_SIZE))):
                if i + BATCH_SIZE < len(df_valid):
                    df_sub = df_valid.loc[i: i + BATCH_SIZE]
                else:
                    df_sub = df_valid
                labels = [torch.FloatTensor([y]).to(device) for y in df_sub.affinity]
                mol_gs = [*map(smile_to_graph, df_sub.compound_iso_smiles)]
                tgt_gs = [prot_key_to_graph(key).to(device) for key in df_sub.prot_key]
                tgt_llms = [pid_to_llm(pid).to(device) for pid in df_sub.protein_id]

                drug_gs = []
                for mol_g in mol_gs:
                    c_size, feats, edge_index = mol_g

                    mol_g = DATA.Data(x=torch.Tensor(np.array(feats)),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0))
                    mol_g.__setitem__('c_size', torch.LongTensor([c_size]))
                    drug_gs.append(mol_g.to(device))

                output = model(Batch.from_data_list(drug_gs), tgt_gs, tgt_llms)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, torch.stack(labels, 0).view(-1).cpu()), 0)
        G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()

        val1 = get_mse(G, P)
        if df_test is not None:
            print('predicting for test data')
            
            # G, P = predicting(model, device, df_test)
            model.eval()
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            print('Make prediction for {} samples...'.format(len(df_test)))
            with torch.no_grad():
                for batch_idx, i in enumerate(tqdm(range(0, len(df_test), BATCH_SIZE))):
                    if i + BATCH_SIZE < len(df_test):
                        df_sub = df_test.loc[i: i + BATCH_SIZE]
                    else:
                        df_sub = df_test
                    labels = [torch.FloatTensor([y]).to(device) for y in df_sub.affinity]
                    mol_gs = [*map(smile_to_graph, df_sub.compound_iso_smiles)]
                    tgt_gs = [prot_key_to_graph(key).to(device) for key in df_sub.prot_key]
                    tgt_llms = [pid_to_llm(pid).to(device) for pid in df_sub.protein_id]

                    drug_gs = []
                    for mol_g in mol_gs:
                        c_size, feats, edge_index = mol_g

                        mol_g = DATA.Data(x=torch.Tensor(np.array(feats)),
                                            edge_index=torch.LongTensor(edge_index).transpose(1, 0))
                        mol_g.__setitem__('c_size', torch.LongTensor([c_size]))
                        drug_gs.append(mol_g.to(device))

                    output = model(Batch.from_data_list(drug_gs), tgt_gs, tgt_llms)
                    total_preds = torch.cat((total_preds, output.cpu()), 0)
                    total_labels = torch.cat((total_labels, torch.stack(labels, 0).view(-1).cpu()), 0)
            G, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()

            val2 = get_mse(G, P)
            if val2 < best_test_mse:
                best_test_mse = val2
                best_test_epoch = epoch + 1
                msg = f"test mse has improved at epoch {best_test_epoch}, test mse: {best_test_mse}"
                print(msg)
        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            if df_test is not None:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse, "test mse:", val2)
                msg = "epoch-%d, loss-%.4f, test_loss-%.4f" % (epoch, val1, val2)
            else:
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
                msg = "epoch-%d, loss-%.4f" % (epoch, val1)
            model_path = os.path.join(config.log_dir, msg + '.pt')
            ensure_path(config.log_dir)
            torch.save(model.state_dict(), model_path)
            print("model has been saved to %s." % (model_path))
        else:
            if df_test is not None:
                print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse,
                      "Best test at:", best_test_epoch, '; best_test_mse', best_test_mse)
            else:
                print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)
        scheduler.step()

# Training the model
def main(args):
    config = args2config(args)

    model = LAMP(config=config)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.wd
    )
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=config.max_epochs,
        max_lr=config.lr,
        min_lr=1e-8,
        warmup_steps=int(config.max_epochs * 0.1)
    )

    fold = '0'
    df_test = pd.read_csv(f'data/{dataset}/{dataset}_test_clr.csv', low_memory=False)
    df_train_fold = pd.read_csv(f'data/{dataset}/{dataset}_train_fold_{fold}_clr.csv', low_memory=False)
    df_valid_fold = pd.read_csv(f'data/{dataset}/{dataset}_valid_fold_{fold}_clr.csv', low_memory=False)

    train_eval(model, optimizer,scheduler, df_train_fold, df_valid_fold, df_test, config, config.max_epochs)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=8, type=int) # TODO
    parser.add_argument('--seed', default=42, type=int)

    # Model setting
    parser.add_argument('--n_output', default=1, type=int)
    parser.add_argument('--num_features_prot', default=31, type=int)
    parser.add_argument('--num_features_mol', default=82, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    # Training Info
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--date', default=time.strftime("%Y-%m-%d", time.localtime()), type=str)
    parser.add_argument('--model', default='lamp_hs-128', type=str)
    parser.add_argument('--fold', default='0', type=str)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=10)

    args = parser.parse_args()
    
    args.log_dir = f'{args.model}_davis_fold[{args.fold}]_logs'

    main(args)