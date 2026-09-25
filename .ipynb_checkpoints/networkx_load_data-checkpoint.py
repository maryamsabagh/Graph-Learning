'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import networkx as nx
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, kcore):
        self.path = path
        
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.G = nx.Graph()
            
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 1:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users += 1
                    self.n_train += len(items)
                    
        self.G.add_nodes_from([(uid, {'name':uid, "node_type" : "user"}) for uid in self.exist_users])

        edge_list = []
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 1:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.G.add_nodes_from([(i+self.n_users, {'name':i, "node_type": "item"}) for i in items])                                        
                    edge_list += [(uid, i+self.n_users) for i in items]

        self.G.add_edges_from(edge_list)
        print(f"Total number of users: {self.n_users}")
        print(f"Total number of items: {self.n_items+1}")
        print(f"Total number of nodes should be: {self.G.number_of_nodes()}")

        self.G = nx.k_core(self.G, kcore)
        print('Num nodes after kcore:', self.G.number_of_nodes(), '. Num edges after kcore:', self.G.number_of_edges())
        
    

        
        
    













