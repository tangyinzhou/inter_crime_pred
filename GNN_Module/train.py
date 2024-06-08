from data_loader import *
from model import *

G = load_graph(dataset="CHI")
train_dataloader, val_dataloader, test_dataloader = split_dataset(dataset="CHI")
print(1)