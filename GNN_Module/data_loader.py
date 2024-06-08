import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def load_graph(dataset: str):
    fpath = "/home/tangyinzhou/inter_crime_pred/data/{0}/{0}.csv".format(dataset)
    i2n_path = "/home/tangyinzhou/inter_crime_pred/data/{0}/index2name.csv".format(
        dataset
    )
    edge_data = pd.read_csv(fpath)
    index2name = pd.read_csv(i2n_path)
    src_nodes = []
    dst_nodes = []
    for _, row in edge_data.iterrows():
        index1 = index2name[index2name["name"] == row["Community1"]]["index"].values[0]
        index2 = index2name[index2name["name"] == row["Community2"]]["index"].values[0]
        src_nodes.append(index1)
        dst_nodes.append(index2)
    g_dgl = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
    return g_dgl

def build_dataset(dataset: str):
    fpath = "/home/tangyinzhou/inter_crime_pred/data/{0}/{0}.csv".format(dataset)