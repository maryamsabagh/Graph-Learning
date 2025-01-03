import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

def playlist_track_graph(playlists, tracks, playlist_data, kcore):

    ##### Create graph from these lists

    # adding nodes
    G = nx.Graph()
    G.add_nodes_from([
        (p, {'name':p, "node_type" : "playlist"}) for p in playlists
    ])
    G.add_nodes_from([
        (t, {'name':t, "node_type" : "track"}) for t in tracks
    ])

    # adding edges
    edge_list = []
    for p_name, playlist in playlist_data.items():
        edge_list += [(p_name, t) for t in playlist.tracks]
    G.add_edges_from(edge_list)

    print('Num nodes:', G.number_of_nodes(), '. Num edges:', G.number_of_edges())

    G = nx.k_core(G, kcore)

    return G   

def graph_attribute_builder(nodes1, attribute1, nodes2, attribute2, edges):

    ##### Create graph from these lists

    # adding nodes
    Gsub = nx.Graph()
    Gsub.add_nodes_from([
        (n1, {'name':n1, "node_type" : attribute1}) for n1 in nodes1
    ])
    Gsub.add_nodes_from([
        (n2, {'name':n2, "node_type" : attribute2}) for n2 in nodes2
    ])
    Gsub.add_edges_from(edges)
    #Gsub = nx.k_core(Gsub, kcore)
    return Gsub

def extra_attribute_edge_index(kcore_selected_tracks, track_artist, num_playlists, num_tracks):
    kcore_track_artist = []
    kcore_selected_artists = []
    for x, y in track_artist:
        if x in kcore_selected_tracks:
            kcore_track_artist.append((x, y))
            if y not in kcore_selected_artists:
                kcore_selected_artists.append(y)

    kcore_reduced_track_artist_graph = graph_attribute_builder(kcore_selected_tracks, 'track', kcore_selected_artists, 'artist', kcore_track_artist)
    num_artists = len(kcore_selected_artists)

    track_artist_nodes = sorted(list(kcore_reduced_track_artist_graph.nodes()))
    # put tracks before artists so it matches ordering from above
    track_artist_nodes = track_artist_nodes[num_artists:] + track_artist_nodes[:num_artists]
    track_artist_node2id = dict(zip(track_artist_nodes, np.arange(num_playlists, num_playlists+num_tracks+num_artists)))
    kcore_reduced_track_artist_graph = nx.relabel_nodes(kcore_reduced_track_artist_graph, track_artist_node2id)

    coo_graph = np.array(kcore_reduced_track_artist_graph.edges()).T
    edge_idx = torch.tensor(coo_graph, dtype=torch.long)
    kcore_reduced_track_artist_data = Data(edge_index = edge_idx, num_nodes = kcore_reduced_track_artist_graph.number_of_nodes())
    kcore_reduced_track_artist_data.edge_index = kcore_reduced_track_artist_data.edge_index.type(torch.int64)

    print(f"Number of kcore selected artists: {num_artists}")
    return num_artists, kcore_reduced_track_artist_data.edge_index

def graph_relabel_and_sort(G):
    ##### Separation into train, validation, and testing sets.

    n_nodes = G.number_of_nodes()

    # by sorting them we get an ordering playlist1, ..., playlistN, track1, ..., trackN
    sorted_nodes = sorted(list(G.nodes()))

    # create dictionaries to index to 0 to n_nodes, will be necessary for when we are using tensors
    node2id = dict(zip(sorted_nodes, np.arange(n_nodes)))
    id2node = dict(zip(np.arange(n_nodes), sorted_nodes))

    G = nx.relabel_nodes(G, node2id)
    return G, node2id, id2node

def playlist_track_label_generator(node2id, n_nodes):
    
    # also keep track of how many playlists, tracks we have
    playlists_idx = [i for i, v in enumerate(node2id.keys()) if "playlist" in v]
    tracks_idx = [i for i, v in enumerate(node2id.keys()) if "track" in v]
    
    return playlists_idx, tracks_idx

def train_valid_test_split(G):

    coo_graph = np.array(G.edges()).T
    num_nodes = G.number_of_nodes()

    edge_idx = torch.tensor(coo_graph, dtype=torch.long)
    graph_data = Data(edge_index = edge_idx, num_nodes = num_nodes)

    # convert to train/val/test splits
    transform = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.,
        num_val=0.15, num_test=0.15
    )
    train_split, val_split, test_split = transform(graph_data)

    # note these are stored as float32, we need them to be int64 for future training

    # Edge index: message passing edges
    train_split.edge_index = train_split.edge_index.type(torch.int64)
    val_split.edge_index = val_split.edge_index.type(torch.int64)
    test_split.edge_index = test_split.edge_index.type(torch.int64)
    # Edge label index: supervision edges
    train_split.edge_label_index = train_split.edge_label_index.type(torch.int64)
    val_split.edge_label_index = val_split.edge_label_index.type(torch.int64)
    test_split.edge_label_index = test_split.edge_label_index.type(torch.int64)

    print(f"Train set has {train_split.edge_label_index.shape[1]} positive supervision edges")
    print(f"Validation set has {val_split.edge_label_index.shape[1]} positive supervision edges")
    print(f"Test set has {test_split.edge_label_index.shape[1]} positive supervision edges")

    print(f"Train set has {train_split.edge_index.shape[1]} message passing edges")
    print(f"Validation set has {val_split.edge_index.shape[1]} message passing edges")
    print(f"Test set has {test_split.edge_index.shape[1]} message passing edges")

    return train_split, val_split, test_split
