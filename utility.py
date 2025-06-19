import re
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
import os
import scipy.sparse as sp

def load_seqs(fn):
    ids = []
    seqs = []
    labels = []
    pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if line[0] == '>':
                cur = []
                for i in range(1, 9):
                    cur.append(0 if line[i] == '0' else 1)
                labels.append(cur)
                ids.append(line[1:])
            elif len(pattern.findall(line)) == 0:
                seqs.append(line)
            else:
                print(line)
    return ids, seqs, np.array(labels)

def readesm(directory, file):
    results = []
    f = open(directory + '/' + file, 'r')
    list = f.readlines()
    for i in range(0, len(list), 2):
        sequence = list[i].strip('\n')[1:] + '.npy'
        cur = torch.tensor(np.load(directory + '/' + sequence))
        if cur.shape[0] < 50:
            cur = torch.cat((cur, torch.zeros(50 - cur.shape[0], 1280)), 0)
        results.append(cur)
    return results

def readnetsurfp(directory, file):
    results = []
    df = pd.read_csv(directory + '/' + file, header=None, skiprows=1)
    temp = df.values.tolist()
    seq = [[0 for i in range(24)] for j in range(50)]
    k = 0
    for i in range(len(temp)):
        ch = temp[i][1]
        if ch == 'A':
            seq[k][0] = 1
        elif ch == 'R':
            seq[k][1] = 1
        elif ch == 'N':
            seq[k][2] = 1
        elif ch == 'D':
            seq[k][3] = 1
        elif ch == 'C':
            seq[k][4] = 1
        elif ch == 'Q':
            seq[k][5] = 1
        elif ch == 'E':
            seq[k][6] = 1
        elif ch == 'G':
            seq[k][7] = 1
        elif ch == 'H':
            seq[k][8] = 1
        elif ch == 'I':
            seq[k][9] = 1
        elif ch == 'L':
            seq[k][10] = 1
        elif ch == 'K':
            seq[k][11] = 1
        elif ch == 'M':
            seq[k][12] = 1
        elif ch == 'F':
            seq[k][13] = 1
        elif ch == 'P':
            seq[k][14] = 1
        elif ch == 'S':
            seq[k][15] = 1
        elif ch == 'T':
            seq[k][16] = 1
        elif ch == 'W':
            seq[k][17] = 1
        elif ch == 'Y':
            seq[k][18] = 1
        elif ch == 'V':
            seq[k][19] = 1
        seq[k][20] = float(temp[i][3])
        seq[k][21] = float(temp[i][6])
        seq[k][22] = float(temp[i][7])
        seq[k][23] = float(temp[i][8])
        k += 1
        if i == len(temp) - 1 or temp[i + 1][0] != temp[i][0]:
            results.append(torch.tensor(seq, dtype=torch.float32))
            seq = [[0 for i in range(24)] for j in range(50)]
            k = 0
    return results

def load_data(fasta_path, directory, file, folder, strurcture, contact):
    ids, seqs, labels = load_seqs(fasta_path)
    esm_encodings = readesm(directory, file)
    netsurfp = readnetsurfp(folder, strurcture)

    n_samples = len(ids)
    data_list = []
    for i in range(n_samples):
        data_list.append(to_parse_matrix(contact + '/' + ids[i] + '.npy', netsurfp[i], esm_encodings[i], labels[i]))

    return data_list

def to_parse_matrix(name, N, S, Y):
    edge_index = torch.LongTensor(np.load(name))
    x = torch.tensor(torch.cat((N, S), 1), dtype=torch.float32)
    y = torch.tensor([Y], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)
