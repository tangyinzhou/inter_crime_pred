from data_loader import *
from llm_pred import *
from train_GNN import *
import os
from util import *
import pickle as pkl

llm_prompt_weight = 0   # 可选0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
gnn_weight = 0
use_dataset = "CHI"  # 可选"CHI","NYC", "SF"
year = "2016"  # 可选"2016","2017","2018","2019","2020"
openai_key = read_openai_key()

with open(
    "./data/community_feature_dict.json", "r"
) as f:
    city_dict = json.load(f)
adj = load_graph(dataset=use_dataset)
llm_train_dataset, llm_val_dataset, llm_test_dataset = split_dataset(
    dataset=use_dataset
)
input_dim, output_dim, hidden_dim, num_nodes = get_hyperparams(use_dataset)
hyper_params = {
    "input_dim": input_dim,
    "output_dim": output_dim,
    "hidden_dim": hidden_dim,
    "num_nodes": num_nodes,
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
for round in range(1, 51):
    if round % 10 == 0:
        llm_prompt_weight += 0.2
        gnn_weight += 0.2

    gnn_train_dataset = llm_pred_func(
        use_dataset,
        llm_train_dataset,
        llm_prompt_weight,
        adj,
        use_dataset,
        year,
        city_dict,
        openai_key,
    )
    with open('./data/dataset/gnn_train_dataset.pkl', 'wb') as f:
        pkl.dump((gnn_train_dataset.data, gnn_train_dataset.llm_pred), f)
    exit(0)
    # with open('./data/dataset/gnn_train_dataset.pkl', 'rb') as f:
    #     data, llm_preds = pkl.load(f)
    # gnn_train_dataset = GNNDataset(data, llm_preds)

    # gnn_val_dataset = llm_pred_func(
    #     use_dataset,
    #     llm_val_dataset,
    #     llm_prompt_weight,
    #     adj,
    #     use_dataset,
    #     year,
    #     city_dict,
    #     openai_key,
    # )
    # with open('./data/dataset/gnn_val_dataset.pkl', 'wb') as f:
    #     pkl.dump((gnn_val_dataset.data, gnn_val_dataset.llm_pred), f)
    with open('./data/dataset/gnn_val_dataset.pkl', 'rb') as f:
        data, llm_preds = pkl.load(f)
    gnn_val_dataset = GNNDataset(data, llm_preds)

    # gnn_test_dataset = llm_pred_func(
    #     use_dataset,
    #     llm_test_dataset,
    #     llm_prompt_weight,
    #     adj,
    #     use_dataset,
    #     year,
    #     city_dict,
    #     openai_key,
    # )
    # with open('./data/dataset/gnn_test_dataset.pkl', 'wb') as f:
    #     pkl.dump((gnn_test_dataset.data, gnn_test_dataset.llm_pred), f)
    with open('./data/dataset/gnn_test_dataset.pkl', 'rb') as f:
        data, llm_preds = pkl.load(f)
    gnn_test_dataset = GNNDataset(data, llm_preds)

    trained_GNN = train_GNN(
        gnn_train_dataset, gnn_val_dataset, gnn_weight, adj, hyper_params, device
    )
    llm_train_dataset = gnn_pred_func(gnn_train_dataset, trained_GNN, adj)
    llm_val_dataset = gnn_pred_func(gnn_val_dataset, trained_GNN, adj)
    llm_test_dataset = gnn_pred_func(gnn_test_dataset, trained_GNN, adj)
print("training finished!")
