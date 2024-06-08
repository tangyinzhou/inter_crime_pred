import pandas as pd
import numpy as np

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json


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


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data.iloc[idx][:-1]
        label = self.data.iloc[idx][-1]

        return feature, label


def build_dataset(data, start_year: int, end_year: int, history=11, pred=1):
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
    dataset = CustomDataset(data_list)
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
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader
