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
    counts, edges = np.histogram(deg_seq, bins=nbins, density=True)
    etrim = edges[:-1]
    plt.plot(np.log10(etrim), np.log10(counts))
    plt.title(r"$P(k)$ vs $k$")
    plt.xlabel(r"$\log k$")
    plt.ylabel(r"$P(k)$")
    return counts, edges
    