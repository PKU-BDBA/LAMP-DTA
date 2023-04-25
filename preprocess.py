import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
import re
import csv
import esm
import torch
from rdkit import Chem
import pypdb
from tqdm import tqdm


def generate_protein_pretraining_representation(dataset_name, prots):
    data_dict = {}
    prots_tuple = [(str(i), prots[i][:1022]) for i in range(len(prots))]
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    i = 0
    batch = 1
    
    while (batch*i) < len(prots):
        print('converting protein batch: '+ str(i))
        if (i + batch) < len(prots):
            pt = prots_tuple[batch*i:batch*(i+1)]
        else:
            pt = prots_tuple[batch*i:]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(pt)
        #model = model.cuda()
        #batch_tokens = batch_tokens.cuda()
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[36], return_contacts=True)
        token_representations = results["representations"][36].numpy()
        data_dict[i] = token_representations
        i += 1
    np.savez(dataset_name + '.npz', dict=data_dict)
    
def query_pdb_id(seq):
    q = pypdb.Query(seq,
              query_type="sequence", 
              return_type="polymer_entity")
    res = q.search()
    
    if res is not None:
        return res['result_set'][0]['identifier']
    else:
        return 'TOFOLD'
    
ESM = False
datasets = ['davis','kiba']
for dataset in datasets:
    fpath = 'data/' + dataset + '/raw/'
    train_valid_folds = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    valid_ids = [0, 1, 2, 3, 4]
    valid_folds = [train_valid_folds[vid] for vid in valid_ids]
    train_folds = []
    for valid_id in valid_ids:
        temp = []
        for idx in range(5):
            if idx != valid_id:
                temp += train_valid_folds[idx]
        train_folds.append(temp)
    
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    drug_smiles = []
    prot_seqs = []
    prot_pdb_ids = []
    prot_ori_keys = []
    for d in tqdm(ligands.keys()):
        #lg = ligands[d]
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    for t in tqdm(proteins.keys()):
        prot_seqs.append(proteins[t])
        prot_pdb_ids.append(query_pdb_id(proteins[t]))
        prot_ori_keys.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]

    # protein pretraing presentation
    if ESM:
        generate_protein_pretraining_representation(dataset, prot_seqs)

    affinity = np.asarray(affinity)
    opts = ['train','valid']

    print('generating test data')
    rows, cols = np.where(np.isnan(affinity) == False)  
    test_rows, test_cols = rows[test_fold], cols[test_fold]
    with open('data/' + dataset + '/' + dataset  + '_test.csv', 'w') as f:
        f.write('compound_iso_smiles,target_sequence,target_pdb_id,target_original_key,affinity,protein_id,drug_id\n')
        for pair_ind in range(len(test_rows)):
            ls = []
            ls += [ drugs[test_rows[pair_ind]] ]
            ls += [ prot_seqs[test_cols[pair_ind]] ]
            ls += [ prot_pdb_ids[test_cols[pair_ind]] ]
            ls += [ prot_ori_keys[test_cols[pair_ind]] ]
            ls += [ affinity[test_rows[pair_ind], test_cols[pair_ind]] ]
            ls += [ test_cols[pair_ind] ]
            ls += [ test_rows[pair_ind] ]
            f.write(','.join(map(str, ls)) + '\n')

    for i in range(5):
        train_fold = train_folds[i]
        valid_fold = valid_folds[i]
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity) == False)  
            if opt == 'train':
                rows, cols = rows[train_fold], cols[train_fold]
                
                #generating cold data
                with open('data/' + dataset + '_cold' + '.csv', 'w') as f:
                    f.write('compound_iso_smiles,target_sequence,target_original_key,affinity,protein_id,drug_id\n')
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ drugs[rows[pair_ind]] ]
                        ls += [ prot_seqs[cols[pair_ind]] ]
                        ls += [ prot_ori_keys[cols[pair_ind]] ]
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]] ]
                        ls += [ cols[pair_ind] ]
                        ls += [ rows[pair_ind] ]
                        f.write(','.join(map(str,ls)) + '\n') 
            elif opt == 'valid':
                rows, cols = rows[valid_fold], cols[valid_fold]
                
                #generating cold data
                with open('data/' + dataset + '_cold' + '.csv', 'a') as f:
                    for pair_ind in range(len(rows)):
                        ls = []
                        ls += [ drugs[rows[pair_ind]] ]
                        ls += [ prot_seqs[cols[pair_ind]] ]
                        ls += [ prot_ori_keys[cols[pair_ind]] ]
                        ls += [ affinity[rows[pair_ind],cols[pair_ind]] ]
                        ls += [ cols[pair_ind] ]
                        ls += [ rows[pair_ind] ]
                        f.write(','.join(map(str,ls)) + '\n') 
                      
            #5-fold data
            print('generating 5-fold data')
            with open('data/' + dataset + '/' + dataset + '_' + opt + '_fold_' + str(i) + '.csv', 'w') as f:
                f.write('compound_iso_smiles,target_sequence,target_pdb_id,target_original_key,affinity,protein_id,drug_id\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]] ]
                    ls += [ prot_seqs[cols[pair_ind]] ]
                    ls += [ prot_pdb_ids[cols[pair_ind]] ]
                    ls += [ prot_ori_keys[cols[pair_ind]] ]
                    ls += [ affinity[rows[pair_ind], cols[pair_ind]] ]
                    ls += [ cols[pair_ind] ]
                    ls += [ rows[pair_ind] ]
                    f.write(','.join(map(str,ls)) + '\n')