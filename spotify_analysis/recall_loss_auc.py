import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

import matplotlib
plt.rcParams.update({ "text.usetex": True, "font.family": "serif" })
font = {'size': 16}
matplotlib.rc('font', **font)
fmt = 'pdf'


def load_model_runs(model_name, neg, num_runs=5, art=True):
    runs = []
    if art==True:
        for i in range(num_runs):
            filename = f"art_LGCN_{model_name}_3_e64_nodes22110__run{i}_{neg}_1e-8_BPR_random_.pkl"
            with open(filename, "rb") as f:
                runs.append(pickle.load(f))
    else:
        for i in range(num_runs):
            filename = f"LGCN_{model_name}_3_e64_nodes22110__run{i}_{neg}_1e-8_BPR_random_.pkl"
            with open(filename, "rb") as f:
                runs.append(pickle.load(f))
    return runs

def compute_recall(runs):
    recalls = [run['val']['recall'] for run in runs]
    mean_recall = np.mean(recalls, axis=0)
    std_recall = np.std(recalls, axis=0)
    return mean_recall, std_recall

def to_numpy(data):
    if isinstance(data, list):
        return [tensor.cpu().numpy() for tensor in data]
    return data.cpu().numpy()  

def compute_loss(runs):
    losses= np.array([to_numpy(run['val']['loss']) for run in runs])
    mask = (losses > 0) & (losses < 0.7)
    filtered_losses = np.where(mask, losses, np.nan)  
    std_filtered = np.nanstd(filtered_losses, axis=0)
    avg_filtered = np.nanmean(filtered_losses, axis=0)
    return avg_filtered[::10], std_filtered[::10] 


def compute_AUC(runs):
    ROCs = [run['val']['roc'] for run in runs]
    mean_ROC = np.mean(ROCs, axis=0)
    std_ROC = np.std(ROCs, axis=0)
    return mean_ROC[::10], std_ROC[::10]

def plot(avgs, stds, label, color):
    epochs = 10 * np.arange(1, len(avgs) + 1)
    plt.plot(epochs, avgs,label=label, color=color)
    plt.errorbar(epochs, avgs, 
             yerr=stds, fmt='-o', capsize=4, capthick=1, color=color,markersize=4)






