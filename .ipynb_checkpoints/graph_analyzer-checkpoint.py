import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def degree_sequence(G):
    return sorted((d for n, d in G.degree()), reverse=True)
    
def plt_node_rank(deg_seq):
    plt.plot(deg_seq, "b-", marker="o")
    plt.title("Degree Rank Plot")
    plt.ylabel("Degree")
    plt.xlabel("Rank")

def plt_deg_distribution(deg_seq, nbins):
    plt.hist(deg_seq, bins=nbins, density=True)

