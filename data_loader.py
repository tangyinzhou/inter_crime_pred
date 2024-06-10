import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json


def load_graph(dataset: str):
    fpath = (
        "/home/tangyinzhou/inter_crime_pred/data/{0}/{0}_community_neighbor.csv".format(
            dataset
        )
    )
    i2n_path = "/home/tangyinzhou/inter_crime_pred/data/{0}/index2name.csv".format(
        dataset
    )
    edge_data = pd.read_csv(fpath)
    index2name = pd.read_csv(i2n_path)
    adj = np.zeros((len(index2name), len(index2name)))
    for _, row in edge_data.iterrows():
        index1 = index2name[index2name["name"] == row["Community1"]]["index"].values[0]
        index2 = index2name[index2name["name"] == row["Community2"]]["index"].values[0]
        adj[index1, index2] = 1
        adj[index2, index1] = 1
    return adj


class LLMDataset(Dataset):  # LLM输入用的Dataset，里面放的GNN的输出
    def __init__(self, data, gnn_pred):
        self.data = data
        self.gnn_pred = gnn_pred

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx][:-1]
        label = self.data[idx][-1]
        gnn_pred = self.gnn_pred[idx]

        return feature, label, gnn_pred


class GNNDataset(Dataset):  # GNN输入用的Dataset，里面存放LLM输出
    def __init__(self, data, llm_pred):
        self.data = data
        self.llm_pred = llm_pred

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx][:-1]
        label = self.data[idx][-1]
        llm_pred = self.llm_pred[idx]

        return feature, label, llm_pred


def build_dataset(
    data, start_year: int, end_year: int, history=11, pred=1, cat_or_all="cat"
):
    train_data = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            month_data = data["{0}-{1}".format(year, month)]
            month_list = []
            for cat, cat_data in month_data.items():
                cat_list = []
                for area, area_data in cat_data.items():
                    cat_list.append(area_data)
                month_list.append(cat_list)
            month_list = np.array(month_list).T.tolist()
            train_data.append(month_list)
    train_data = np.array(train_data)
    history = 11
    pred = 1
    data_list = []
    for x in range(0, train_data.shape[0] - (history + pred)):
        data_list.append(train_data[x : x + history + pred])
    if cat_or_all == "all":
        persudo_gnn_pred = np.zeros((len(data_list), 1, data_list[0].shape[1], 1))
    elif cat_or_all == "cat":
        persudo_gnn_pred = np.zeros((len(data_list), 1, data_list[0].shape[1], 30))
    dataset = LLMDataset(data_list, persudo_gnn_pred)
    return dataset


def split_dataset(dataset: str):
    if dataset == "CHI":
        start_year = 2001
        train_year = 2012
        val_year = 2016
        test_year = 2020
    elif dataset == "NYC":
        start_year = 2006
        train_year = 2015
        val_year = 2018
        test_year = 2021
    elif dataset == "SF":
        start_year = 2006
        train_year = 2014
        val_year = 2016
        test_year = 2018
    fpath = "/home/tangyinzhou/inter_crime_pred/data/{0}/dataset.json".format(dataset)
    with open(fpath, "r") as f:
        data = json.load(f)
    train_dataset = build_dataset(data, start_year, train_year)
    val_dataset = build_dataset(data, train_year, val_year)
    test_dataset = build_dataset(data, val_year, test_year)
    return train_dataset, val_dataset, test_dataset


def get_hyperparams(use_dataset):
    if use_dataset == "CHI":
        input_dim = 31
        output_dim = 30
        hidden_dim = 64
        num_nodes = 77
    return input_dim, output_dim, hidden_dim, num_nodes
