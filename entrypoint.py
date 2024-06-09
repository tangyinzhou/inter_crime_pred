from data_loader import *
from llm_pred import *
from train_GNN import *

adj = load_graph(dataset="CHI")
llm_train_dataset, llm_val_dataset, llm_test_dataset = split_dataset(dataset="CHI")
llm_prompt_weight = 0
gnn_weight = 0
for round in range(1, 101):
    if round % 10 == 0:
        llm_prompt_weight += 0.1
        gnn_weight += 0.1
    gnn_train_dataset = llm_pred_func(llm_train_dataset, llm_prompt_weight, adj)
    gnn_val_dataset = llm_pred_func(llm_val_dataset, llm_prompt_weight, adj)
    gnn_test_dataset = llm_pred_func(llm_test_dataset, llm_prompt_weight, adj)
    trained_GNN = train_GNN(gnn_train_dataset, gnn_val_dataset, gnn_weight, adj)
    llm_train_dataset = gnn_pred_func(gnn_train_dataset, trained_GNN, adj)
    llm_val_dataset = gnn_pred_func(gnn_val_dataset, trained_GNN, adj)
    llm_test_dataset = gnn_pred_func(gnn_test_dataset, trained_GNN, adj)
print("training finished!")
