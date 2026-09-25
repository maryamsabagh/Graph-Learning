import numpy as np
import pickle
import os
from pathlib import Path as Data_Path
import networkx as nx
import networkx_load_data as ld
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from gcn_class import GCN
from train_and_test import train
from graph_building_routines import graph_relabel_and_sort, train_valid_test_split

MAIN_DIR='/home/jovyan/code4'
os.chdir(MAIN_DIR)
AMAZON_DATA_DIR = MAIN_DIR + '/amazon_example/Light_GCN_Git_Clone/amazon_data'

amazon_data = ld.Data(AMAZON_DATA_DIR,kcore=26)
Gred = amazon_data.G
kcore_selected_users = [x for x,y in Gred.nodes(data=True) if y["node_type"]=="user"]
kcore_selected_items = [x for x,y in Gred.nodes(data=True) if y["node_type"]=="item"]
num_users = len(kcore_selected_users)
num_items = len(kcore_selected_items)
print(f"Number of kcore selected playlists: {num_users}")
print(f"Number of kcore selected tracks: {num_items}")
print(f"Number of kcore selected nodes: {Gred.number_of_nodes()}")

Gred, node2id, id2node = graph_relabel_and_sort(Gred)
train_split, val_split, test_split = train_valid_test_split(Gred)


num_nodes = num_users +  num_items

# create a dictionary of the dataset splits
datasets = {
    'train':train_split,
    'val':val_split,
    'test': test_split
}

# initialize our arguments
args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_layers' :  3,
    'emb_size' : 64,
    'weight_decay': 1e-5,
    'lr': 0.01,
    'epochs': 301
}

# initialize model and and optimizer

num_neg_edges = 1

model = GCN(
    num_nodes = num_nodes, num_layers = args['num_layers'],
    embedding_dim = args["emb_size"], conv_layer = "SAGE"
)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

# send data, model to GPU if available
#playlists_idx = torch.Tensor(users_idx).type(torch.int64).to(args["device"])
#tracks_idx =torch.Tensor(items_idx).type(torch.int64).to(args["device"])
datasets['train'].to(args['device'])
datasets['val'].to(args['device'])
datasets['test'].to(args['device'])
model.to(args["device"])

# create directory to save model_stats
MODEL_STATS_DIR = "amazon_model_stats"
if not os.path.exists(MODEL_STATS_DIR):
  os.makedirs(MODEL_STATS_DIR)

runs = 1
for run in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    stats = train(datasets, model, optimizer, "BPR", args, num_items, num_users, num_neg_edges, #track_artist_edge_index,
                  neg_samp = "random")
    pickle.dump(stats, open(f"amazon_model_stats/1_neg/{model.name}_run{run}_{num_neg_edges}_1e-8_BPR_random_.pkl", "wb"))
    
  

