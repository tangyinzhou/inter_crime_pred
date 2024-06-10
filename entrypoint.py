from data_loader import *
from llm_pred import *
from train_GNN import *

llm_prompt_weight = 0
gnn_weight = 0
use_dataset = "CHI"
adj = load_graph(dataset=use_dataset)
llm_train_dataset, llm_val_dataset, llm_test_dataset = split_dataset(
    dataset=use_dataset
)
input_dim, output_dim, hidden_dim, num_nodes = get_hyperparams(use_dataset)
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
