import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.nn import GATConv, LayerNorm
from torch.nn import Linear
import scipy.sparse as sp
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
from utility import load_data
from loss_functions import *
import math
import evaluation

import warnings
warnings.filterwarnings("ignore")

data_list = load_data('sequence/test/test.fasta', 'sequence/test/esm1b', 'test.fasta', 'sequence/test', 'test.csv', 'sequence/test/contact')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AddNorm(nn.Module):
    def __init__(self, normalized, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized)

    def forward(self, X, y):
        return self.ln(self.dropout(y) + X)

class AttentionEncode(nn.Module):
    def __init__(self, dropout, embedding_size, num_heads):
        super(AttentionEncode, self).__init__()
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.at1 = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=num_heads, dropout=0.3)

        self.addNorm1 = AddNorm(normalized=[50, self.embedding_size], dropout=self.dropout)

    def forward(self, x):
        Multi, _ = self.at1(x, x, x)
        Multi_encode = self.addNorm1(x, Multi)
        return Multi_encode

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output

class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads):
        super(GATModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)

        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(nheads * hidden_dim)

        self.t1 = torch.nn.Conv1d(in_channels=1304, out_channels=128, kernel_size=2, stride=1)
        self.t2 = torch.nn.Conv1d(in_channels=1304, out_channels=128, kernel_size=3, stride=1)
        self.t3 = torch.nn.Conv1d(in_channels=1304, out_channels=128, kernel_size=4, stride=1)
        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=5)

        self.lin1 = Linear(4480, 1024)
        self.lin2 = Linear(1024, 256)
        self.lin3 = Linear(256, 64)
        self.lin4 = Linear(64, 32)
        self.lin5 = Linear(32, output_dim)

        self.attention_encode = AttentionEncode(self.drop, 1304, 4)

        self._reset_parameters()

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def TextCNN(self, x):
        x1 = self.t1(x)
        x1 = torch.nn.ReLU()(x1)
        x1 = self.MaxPool1d(x1)

        x2 = self.t2(x)
        x2 = torch.nn.ReLU()(x2)
        x2 = self.MaxPool1d(x2)

        x3 = self.t3(x)
        x3 = torch.nn.ReLU()(x3)
        x3 = self.MaxPool1d(x3)

        y = torch.cat([x1, x2, x3], dim=-1)

        x = F.dropout(y, p=self.drop, training=self.training)

        x = x.view(x.size(0), -1)

        return x

    def forward(self, x, edge_index, batch):
        gnn_encode = self.conv1(x, edge_index)
        gnn_encode = self.norm1(gnn_encode, batch)
        gnn_encode = F.relu(gnn_encode)
        gnn_encode = F.dropout(gnn_encode, p=self.drop, training=self.training)

        gnn_encode = self.conv2(gnn_encode, edge_index)
        gnn_encode = self.norm2(gnn_encode, batch)
        gnn_encode = F.relu(gnn_encode)
        gnn_encode = F.dropout(gnn_encode, p=self.drop, training=self.training)

        gnn_encode = self.conv3(gnn_encode, edge_index)
        gnn_encode = self.norm3(gnn_encode, batch)
        gnn_encode = F.dropout(gnn_encode, p=self.drop, training=self.training)
        gnn_encode = pyg_nn.global_mean_pool(gnn_encode, batch)

        feature = x.view(batch[-1].item() + 1, 50, -1)
        attention_output = self.attention_encode(feature)

        cnn_input = attention_output
        cnn_input = cnn_input.permute(0, 2, 1)
        cnn_output = self.TextCNN(cnn_input)

        gnn_encode = gnn_encode.view(batch[-1].item() + 1, -1)
        print(gnn_encode.shape, cnn_output.shape)
        vector = torch.cat((gnn_encode, cnn_output), 1)
        output = self.lin1(vector)
        output = F.relu(output)
        output = self.lin2(output)
        output = F.relu(output)
        output = self.lin3(output)
        output = F.relu(output)
        output = self.lin4(output)
        output = F.relu(output)
        output = self.lin5(output)

        return output

def evaluate(loader):
    model.eval()
    pred = []
    label = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred.extend(model(data.x, data.edge_index, data.batch).to('cpu').numpy().tolist())
            label.extend(data.y.to('cpu').numpy().tolist())
    return pred, label

loaders = DataLoader(data_list, batch_size=32, shuffle=False)

model = torch.load('gat.pkl').to(device)
pred, label = evaluate(loaders)

print(evaluation.evaluate(np.array(pred), np.array(label)))
p = np.array(pred)
l = np.array(label)
for i in range(8):
    print(evaluation.scores(l[:, i], p[:, i]))