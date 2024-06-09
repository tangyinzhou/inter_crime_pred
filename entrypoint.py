from data_loader import *
from llm_pred import *
from train_GNN import *

G = load_graph(dataset="CHI")
llm_train_dataset, llm_val_dataset, llm_test_dataset = split_dataset(dataset="CHI")
llm_prompt_weight = 0
gnn_weight = 0
for round in range(1, 101):
    if round % 10 == 0:
        llm_prompt_weight += 0.1
        gnn_weight += 0.1
    gnn_train_dataset = llm_pred_func(llm_train_dataset, llm_prompt_weight, G)
    gnn_val_dataset = llm_pred_func(llm_val_dataset, llm_prompt_weight, G)
    gnn_test_dataset = llm_pred_func(llm_test_dataset, llm_prompt_weight, G)
    trained_GNN = train_GNN(gnn_train_dataset, gnn_val_dataset, gnn_weight, G)
    llm_train_dataset = gnn_pred_func(gnn_train_dataset, G)
    llm_val_dataset = gnn_pred_func(gnn_val_dataset, G)
    llm_test_dataset = gnn_pred_func(gnn_test_dataset, G)
print("training finished!")
