import json
import numpy as np
import pickle
import random
import os
from pathlib import Path as Data_Path
from cbasic_types import JSONFile
from tqdm import tqdm
import networkx as nx

from graph_building_routines import graph_relabel_and_sort, graph_attribute_builder, train_valid_test_split, extra_attribute_edge_index

import torch

from gcn_class import GCN
from train_and_test import train



def playlist_track_JSON_loader(JSONs):
  playlist_data = {}
  playlists = []
  tracks = []
  artists = []
  playlist_track_edges = []
  track_artist_edges = []
  # build list of all unique playlists, tracks, and artists 
  for json_file in tqdm(JSONs):
    playlists += [p.name for p in json_file.playlists.values()]
    tracks += [track.uri for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]
    artists += [track.artist_uri for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]
    
    playlist_track_edges += [(playlist.name, track.uri) for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())]
    track_artist_edges += [(track.uri, track.artist_uri) for playlist in json_file.playlists.values() for track in list(playlist.tracks.values())] #edges between track and artist nodes

    playlist_data = playlist_data | json_file.playlists
  return playlists, tracks, artists, playlist_track_edges, track_artist_edges, playlist_data

###### Directory settings

# set the seed for reproducibility
seed = 224
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

###### Data loading

DATA_PATH = Data_Path('data')

file_names_to_use = sorted([f for f in os.listdir(DATA_PATH) 
                            if os.path.isfile(os.path.join(DATA_PATH, f)) and f.endswith('.json')])

n_playlists = 0

# load each json file, and store it in a list of files
JSONs = []
for file_name in tqdm(file_names_to_use, desc='Files processed: ', unit='files', total=len(file_names_to_use)):
  json_file = JSONFile(DATA_PATH, file_name, n_playlists)
  json_file.process_file()
  n_playlists += len(json_file.playlists)
  JSONs.append(json_file)

playlists, tracks, artists, playlist_track, track_artist, playlist_data = playlist_track_JSON_loader(JSONs)

# Note if you've already generated the graph above, you can skip those steps, and simply run set reload to True!
reload = False
if reload:
  Gred = pickle.load(open("30core_first_12.pkl", "rb"))
else:
  kcore = 30
  Gfull = graph_attribute_builder(playlists, 'playlist', tracks, 'track', playlist_track)
  Gred = nx.k_core(Gfull, kcore)
  pickle.dump(Gred, open(f"{kcore}core_first_{12}.pkl", "wb"))

kcore_selected_playlists = [x for x,y in Gred.nodes(data=True) if y["node_type"]=="playlist"]
kcore_selected_tracks = [x for x,y in Gred.nodes(data=True) if y["node_type"]=="track"]
num_playlists = len(kcore_selected_playlists)
num_tracks = len(kcore_selected_tracks)
print(f"Number of kcore selected playlists: {num_playlists}")
print(f"Number of kcore selected tracks: {num_tracks}")
print(f"Number of kcore selected nodes: {Gred.number_of_nodes()}")

Gred, node2id, id2node = graph_relabel_and_sort(Gred)

train_split, val_split, test_split = train_valid_test_split(Gred)

num_artists, track_artist_edge_index = extra_attribute_edge_index(kcore_selected_tracks, track_artist, num_playlists, num_tracks)

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
num_nodes = num_playlists + num_tracks + num_artists
model = GCN(
    num_nodes = num_nodes, num_layers = args['num_layers'],
    embedding_dim = args["emb_size"], conv_layer = "LGC"
)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

# send data, model to GPU if available
#playlists_idx = torch.Tensor(playlists_idx).type(torch.int64).to(args["device"])
#tracks_idx =torch.Tensor(tracks_idx).type(torch.int64).to(args["device"])

datasets['train'].to(args['device'])
datasets['val'].to(args['device'])
datasets['test'].to(args['device'])
track_artist_edge_index = track_artist_edge_index.to(args['device'])

model.to(args["device"])

# create directory to save model_stats
MODEL_STATS_DIR = "model_stats"
if not os.path.exists(MODEL_STATS_DIR):
  os.makedirs(MODEL_STATS_DIR)

train_stats = train(datasets, model, optimizer, "BPR", args, num_tracks, num_playlists, track_artist_edge_index, neg_samp = "random")

