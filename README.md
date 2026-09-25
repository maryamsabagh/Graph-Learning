# Graph-Learning

Link prediction and recommendation on a bipartite playlist–track graph built from Spotify playlists.

This project focuses on a specific subclass of GNNs known as message passing neural networks (MPNNs), which have demonstrated considerable success in recommendation and link prediction tasks. This work explores how MPNNs perform on real-world bipartite graphs through a combination of structural analysis and empirical evaluation. We  GNN models, LightGCN and GraphSAGE on link prediction tasks. Using Bayesian Personalized Ranking (BPR) loss and controlled negative sampling, we test each model's ability to recover hidden edges between two well defined classes. Evaluating all models on the same datasets allows us to directly compare how they handle the structure of large bipartite graphs.

## How it works

1. **Load the data.** Read the Spotify playlist JSON files in `data/` into `Playlist` and `Track` objects.
2. **Build a bipartite graph.** Playlists are one kind of node and tracks are the other. An edge means "this track is in this playlist".
3. **Keep the dense core.** Take the 30-core of the graph: repeatedly remove nodes with fewer than 30 connections, so every remaining playlist and track has at least 30 neighbours.
4. **Add artists.** Each track is also linked to its artist. Artist nodes give the model extra structure to pass messages through during training.
5. **Split the edges** into train (70%), validation (15%) and test (15%) sets. The held-out edges are the ones the model has to recover.
6. **Train a GNN** that learns an embedding vector for every node. The score for a playlist–track pair is the dot product of their embeddings.
7. **Evaluate** by how well the model separates real edges from sampled non-edges (ROC-AUC) and how many hidden tracks it finds in its top recommendations (recall@K).


## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point. Loads the data, builds and splits the graph, creates the model, and runs training. |
| `basic_types.py` | Classes for the raw data: `Track`, `Playlist`, `JSONFile`. |
| `spotify_data_loader.py` | Reusable function for loading the first N data files. |
| `graph_building_routines.py` | Builds the graph, relabels nodes as integers, and creates the edge splits. |
| `gcn_class.py` | The `GCN` model: node embeddings, message-passing layers, scoring, and losses. |
| `BPR_class.py` | The `BPRLoss` loss function. |
| `sampling_methods.py` | Random and hard negative edge sampling. |
| `train_and_test.py` | The training loop, the evaluation function, and the ROC-AUC metric. |
| `recall_measurement.py` | Recall@K evaluation. |
| `graph_analyzer.py` | Small helpers for degree statistics and plots. |

## Main functions and classes

### Data loading: `basic_types.py`, `spotify_data_loader.py`

- **`Track`**: one track, with its URI, name, artist URI, artist name, and the playlist it belongs to.
- **`Playlist`**: one playlist. `load_tracks()` fills it with its `Track` objects. Playlists are named `playlist_<index>`.
- **`JSONFile`**: loads one data file. `process_file()` turns every playlist in the file into a `Playlist`. Each file starts numbering where the previous one ended, so playlist names never collide.
- **`spotify_data_loader(N_FILES_TO_USE, DATA_DIR)`**: loads the first N data files and returns the lists of playlists, tracks and artists, plus the playlist–track and track–artist edge lists. It also sets the random seeds.

### Graph construction: `graph_building_routines.py`

- **`graph_attribute_builder(nodes1, attribute1, nodes2, attribute2, edges)`**: builds a NetworkX graph with two node types, each tagged with a `node_type`, connected by the given edges. Used for both the playlist–track graph and the track–artist graph.
- **`playlist_track_graph(...)`**: builds the playlist–track graph with k-core 
- **`extra_attribute_edge_index(...)`**: builds the track–artist edges for the tracks that survived the k-core, numbers the artist nodes after the tracks, and returns the artist count and an edge index tensor.
- **`graph_relabel_and_sort(G)`**: replaces node names with integer ids (`node2id`, `id2node`). Sorting the names puts all playlists first and all tracks after them, so ids `0 … num_playlists-1` are playlists and the following ids are tracks.
- **`train_valid_test_split(G)`**: uses PyG's `RandomLinkSplit` to split the edges 70/15/15. It returns three data objects. Each has `edge_index` (the edges used for message passing) and `edge_label_index` (the edges the model is asked to predict).

### The model: `gcn_class.py`

**`GCN(num_nodes, embedding_dim, num_layers, conv_layer=...)`** is adapted from PyTorch Geometric's LightGCN. It keeps one learnable embedding vector per node (playlists, tracks and artists), and refines those vectors by passing them through a stack of graph convolution layers. The final embedding is a weighted sum of the embeddings from every layer, including the starting one. By default the weights are equal, and they can be made learnable with `alpha_learnable=True`.


Key methods:

- **`get_embedding(edge_index)`**: runs the message-passing layers over the graph and returns the final embedding of every node.
- **`predict_link_embedding(embed, edge_label_index)`**: scores each playlist–track pair as the dot product of the two embeddings. A higher score means a more likely edge.
- **`predict_link(...)`** and **`forward(...)`**: score edges directly from an edge index. `predict_link` can return probabilities or 0/1 predictions.
- **`recommend(edge_index, src_index, dst_index, k)`**: returns the top-k highest-scoring destination nodes for each source node.
- **`recommendation_loss(pos, neg)`**: the BPR loss, using `BPRLoss`.
- **`link_pred_loss(pred, label)`**: binary cross-entropy loss, as an alternative.

### Loss: `BPR_class.py`

**`BPRLoss`** implements Bayesian Personalized Ranking. For each pair, it takes the score of a real edge minus the score of a sampled non-edge, and it rewards the model when the real edge scores higher. The result is the mean over all pairs, with optional L2 regularization.

### Negative sampling: `sampling_methods.py`

Training needs examples of "no edge" to contrast with real edges.

- **`sample_negative_edges(...)`** (random): for every real edge, picks one playlist–track pair uniformly at random from all pairs that are not real edges.
- **`sample_hard_negative_edges(...)`** (hard): for each real edge's playlist, scores every track with the current model, ignores real edges, and picks a negative at random from the highest-scoring tracks. These are the tracks the model wrongly likes the most. The pool shrinks from 100% to 50% of the tracks over training, so the negatives get harder as it goes.
- **`sample_negative_edges_nocheck(...)`**: a faster random sampler that does not check the picked pairs are really non-edges.

### Training and evaluation: `train_and_test.py`

- **`train(datasets, model, optimizer, loss_fn, args, ...)`**: the training loop. Each epoch it samples negatives, computes embeddings using the training graph plus the track–artist edges, scores real and negative edges, computes the loss (`"BPR"` or `"BCE"`), and updates the model. It also evaluates on the validation set each epoch and prints train and validation loss and ROC-AUC. Along the way it:
  - computes validation recall@300 every 10 epochs,
  - saves the embeddings every 20 epochs to `model_embeddings/`,
  - saves all the statistics at the end to `model_stats/` as a `.pkl` file.
- **`test(model, data, ...)`**: evaluates a model on a data split without changing it, and returns the loss and ROC-AUC. Used on the validation set during training, and can be used on the test set.
- **`metrics(labels, preds)`**: ROC-AUC, the chance that a real edge scores higher than a negative one.

### Recall: `recall_measurement.py`

**`recall_at_k(data, model, k, ...)`**: for each playlist, scores all tracks, removes tracks the model has already seen as edges, keeps the top k, and checks how many of the hidden edges appear. Recall is the fraction of hidden edges found, averaged over playlists.

### Analysis helpers: `graph_analyzer.py`

- **`degree_sequence(G)`**: node degrees sorted from highest to lowest.
- **`plt_node_rank(deg_seq)`**: degree rank plot.
- **`plt_deg_distribution(deg_seq, nbins)`**: degree histogram.



