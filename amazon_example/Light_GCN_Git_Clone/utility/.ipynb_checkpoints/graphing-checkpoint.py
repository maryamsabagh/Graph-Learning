import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

def graph(G):
    rand_nodes_lg = random.sample(list(G.nodes()), 2000)
    sub_G_lg = G.subgraph(rand_nodes_lg)
    largest_cc_lg = max(nx.connected_components(sub_G_lg.to_undirected()), key=len)
    sub_G_lg = nx.Graph(sub_G_lg.subgraph(largest_cc_lg))
    print('Large subgraph Num nodes:', sub_G_lg.number_of_nodes(),
      '. Num edges:', sub_G_lg.number_of_edges())
    color_map = {"user": 0, "item": 1}
    node_color = [color_map[attr["node_type"]] for (id, attr) in sub_G_lg.nodes(data=True)]
    plt.figure(figsize=(20,20))
    top = nx.bipartite.sets(sub_G_lg)[0]
    pos = nx.bipartite_layout(sub_G_lg, top)
    plt.figure(figsize=(10,10))
    nx.draw(sub_G_lg,
        pos=pos,
        cmap=plt.get_cmap('coolwarm'),
        node_color=node_color,
        node_size = 30,
        width = 3,
        edge_color=(0, 0, 0, 0.1))
    plt.show()

    
    



    