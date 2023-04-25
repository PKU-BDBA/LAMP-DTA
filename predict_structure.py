import os
import json
from collections import OrderedDict
import re
import csv
import pypdb
from tqdm import tqdm
import glob

import pandas as pd
import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from Bio.PDB import *
import deepchem
import pickle
import time

    
# def query_pdb_id(seq):
#     q = pypdb.Query(seq,
#               query_type="sequence", 
#               return_type="polymer_entity")
#     res = q.search()
    
#     if res is not None:
#         return res['result_set'][0]['identifier']
#     else:
#         return 'TOFOLD'
    
# datasets = ['davis','kiba']
# for dataset in datasets:
#     fpath = 'data/' + dataset + '/raw/'
#     proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
#     prot_seq_pdb_dict = OrderedDict()
#     for t in tqdm(proteins.keys()):
#         prot_seq_pdb_dict[proteins[t]] = query_pdb_id(proteins[t])
        
#     print('generating sequence pdb ids map')       
#     with open(dataset + '_seq_pdb.csv', 'w') as f:
#         f.write('target_sequence,target_pdb_id\n')
#         for seq, pdb_id in prot_seq_pdb_dict.items():
#             ls = []
#             ls += [ seq ]
#             ls += [ pdb_id ]
#             f.write(','.join(map(str, ls)) + '\n')

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # (10, 9, 5, 6, 1) --> total 31


def get_atom_feature(m):
    H = []
    for i in range(len(m)):
        H.append(atom_feature(m[i][0]))
    H = np.array(H)

    return H

pk = deepchem.dock.ConvexHullPocketFinder()
def process_protein(pdb_file):
    m = Chem.MolFromPDBFile(pdb_file)
    if m is None:
        return None, None, None
    am = GetAdjacencyMatrix(m) # n2 x n2
    pockets = pk.find_pockets(pdb_file)
    n2 = m.GetNumAtoms()
    c2 = m.GetConformers()[0] # Tid
    d2 = np.array(c2.GetPositions()) # n2 x 3
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)

        ami = am[np.array(idxs)[:, None], np.array(idxs)] # len(idxs) x len(idxs)
        H = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_matrix(ami)
        graph = dgl.from_networkx(g)
        graph.ndata['h'] = torch.Tensor(H)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)
        binding_parts.append(binding_parts_atoms)

    constructed_graphs = dgl.batch(constructed_graphs)

    return binding_parts, not_in_binding, constructed_graphs

PROCESS = True
datasets = ['davis','kiba']
for dataset in datasets:
    df = pd.read_csv(f'{dataset}_seq_pdb.csv')
    print(len(df['target_pdb_id'].unique()))

    if not PROCESS:
        # Download from PDB web
        for pdb_id in tqdm(df['target_pdb_id'].unique()):
            try:
                pdb_code = pdb_id[: 4].lower()
                if any(pdb_code in s for s in os.listdir('./pdbs/')):
                    continue
                if pdb_id != "TOFOLD":
                    pdbl = PDBList(verbose=False)
                    pdbl.retrieve_pdb_file(
                        pdb_code, pdir='./pdbs/', overwrite=False, file_format="pdb"
                    )
                    # Rename file to .pdb from .ent
                    os.rename(
                        './pdbs/' + "pdb" + pdb_code + ".ent", './pdbs/' + pdb_code + ".pdb"
                    )
            except Exception as e:
                print(f'PDB id: {pdb_id}', e)
                continue

    if PROCESS:
        print('Processing proteins...')
        hard_codes = ['6zxf', "6g5i"]

        p_graphs = {}
        i = 1
        msg = []   
        for pdb_id in tqdm(df['target_pdb_id'].unique()):
            msg.append(str(i))
            i += 1
            try:
                pdb_code = pdb_id[: 4].lower()
                if pdb_id != "TOFOLD" and (pdb_code not in hard_codes):
                    if pdb_code not in p_graphs.keys():
                        # Assert file has been downloaded
                        assert any(pdb_code in s for s in os.listdir('./pdbs/'))
                        #print(f"Downloaded PDB file for: {pdb_code}")
                        _, _, constructed_graphs = process_protein(f"./pdbs/{pdb_code}.pdb")

                        p_graphs[pdb_code] = constructed_graphs
                msg.pop()
            except Exception as e:
                msg[-1] = msg[-1] + f' PDB id: {pdb_id} ' + str(e)
                print(f'{time.strftime("%Y-%m-%d|%H:%M:%S", time.localtime())}: pdb_id={pdb_id}, i={i}')
                continue

        with open(f'{dataset}_site.pkl', 'wb') as f:
            pickle.dump(p_graphs, f)
        with open(f'{dataset}_site_msg.pkl', 'wb') as f:
            pickle.dump(msg, f)