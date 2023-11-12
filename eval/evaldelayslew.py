import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import csv
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from dataprocess import makegraphs, makejths
from networks.delaynet import TransposeTREE, GAT, GCN, GraphSAGE, DeepGCN
from networks.graphtrans import GNNTransformer, graph_trans_preprocess
from networks.ntree import NeuralTreeNetwork
from utils import *
from numpy import *


def delay_slew_results(args):
    args.batch_size = 1
    pth = args.save_pth + args.exp + ".pth"
    model = torch.load(pth)

    #  prepare dataset
    if args.model != "NT":
        dset = makegraphs(args.data_pth)
    else:
        dset = makejths(args, args.jth_pth)
    
    loader = DataLoader(dataset=dset, batch_size=1, shuffle=False, drop_last=True)

    if args.solver == 'delay':
        dist = [0, 0.1, 1, 10, 100, 200]
    else:
        dist = [1, 10, 100, 200, 400, 800]
    pred_count = np.zeros(len(dist))
    target_count = np.zeros(len(dist))
    # timecost = torch.zeros(20)
    # fre = torch.zeros(20)
    for data in loader:
        if args.solver == 'Delay':
            target = data.delays
        else:
            target = data.slews
        if args.model == "TREE":
            # start = time.time()
            pred, embs = model(data)
            # cost = time.time() - start
        else:
            # start = time.time()
            pred = model(data)
            # cost = time.time() - start
        mask = data.mask
        # size = mask[0].size
        # timecost[size] += cost
        # fre[size] += 1

        filter_pred, filter_target = filter_pred_target(pred.detach(), mask, target, args)
        filter_pred = filter_pred[:, :target[0].size].numpy().flatten()
        filter_target = filter_target[:, :target[0].size].numpy().flatten()

        if data.id == 4300:  # specify circuit id to show results
            idx = torch.LongTensor([1, 0])
            x1=pred[:,:target[0].size].reshape(target[0].size,-1)
            edge_index=data.edge_index[idx]
            G1 = Data(x=x1, edge_index=edge_index)
            g1 = to_networkx(G1, to_undirected=False)
            pos = nx.nx_agraph.graphviz_layout(g1, prog="dot")
            color_lookup = {k: torch.abs(x1[k,0]).item() for v, k in enumerate(sorted(set(g1.nodes())))}
            low, *_, high = sorted(color_lookup.values())
            norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
            mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

            rcParams['figure.figsize'] = 18,12
            nx.draw(g1, pos=pos,
                    nodelist=color_lookup,
                    node_size=1000,
                    node_color=[mapper.to_rgba(i)
                                for i in color_lookup.values()],
                    with_labels=False)
            plt.savefig('imgs/' + args.exp + '_pred_treegraph.png')
            plt.close()

            x2=torch.tensor(target).reshape(target[0].size,-1)
            G2 = Data(x=x2, edge_index=edge_index)
            g2 = to_networkx(G2, to_undirected=False)
            pos = nx.nx_agraph.graphviz_layout(g2, prog="dot")
            color_lookup = {k: torch.abs(x2[k, 0]).item() for v, k in enumerate(sorted(set(g2.nodes())))}
            low, *_, high = sorted(color_lookup.values())
            norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
            mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

            rcParams['figure.figsize'] = 18, 12
            nx.draw(g2, pos=pos,
                    nodelist=color_lookup,
                    node_size=1000,
                    node_color=[mapper.to_rgba(i)
                                for i in color_lookup.values()],
                    with_labels=False)
            plt.savefig('imgs/' + 'target_treegraph.png')
            plt.close()
            sim_score = np.corrcoef(x1.squeeze().detach().numpy(), x2.squeeze().numpy())
            print(sim_score)
            
            plot(filter_pred, filter_target, args)
            break

        for idx in range(len(dist) - 1):
            tmp = np.sum((filter_pred > dist[idx]) & (filter_pred < dist[idx + 1]))
            pred_count[idx] += tmp
        pred_count[-1] += np.sum(filter_pred > dist[-1])

        for idx in range(len(dist) - 1):
            tmp = np.sum((filter_target > dist[idx]) & (filter_target < dist[idx + 1]))
            target_count[idx] += tmp
        target_count[-1] += np.sum(filter_target > dist[-1])
    
    # show timecost with spice, we only compare v10.
    # fre[fre == 0] = 1
    # avg_timecost = timecost /fre
    # file = args.csv_pth + args.exp + '_avg_timecost.csv'
    # np.savetxt(file, avg_timecost, delimiter=',')
    # show_timecost(args, file)

    # show predicted and golden timing distribution.
    if args.solver == 'Delay':
        tick_label = ['0~0.1', '0.1~1', '1~10', '10~100', '100~200', '200~']
        x = range(6)
        x1 = [i - 0.2 for i in x]
        x2 = [i + 0.2 for i in x]
        plt.figure('Delays Distribution', figsize=(8, 8))
        plt.title('Delays Distribution')
        plt.bar(x1, target_count, 0.4, label='Target')
        plt.bar(x2, pred_count, 0.4, label='Predict')
        plt.xticks(ticks=x, labels=tick_label, rotation=45)
        plt.xlabel('Delay Range')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('imgs/' + args.exp + '_pred_target_dist.png')
        plt.close()
    else:
        tick_label = ['1~10', '10~100', '100~200', '200~400', '400~800', '800~']
        x = range(6)
        x1 = [i - 0.2 for i in x]
        x2 = [i + 0.2 for i in x]
        plt.figure('Slews Distribution', figsize=(8, 8))
        plt.title('Slews Distribution')
        plt.bar(x1, target_count, 0.4, label='Target')
        plt.bar(x2, pred_count, 0.4, label='Predict')
        plt.xlabel('Slew Range')
        plt.ylabel('Frequency   zx')
        plt.xticks(ticks=x, labels=tick_label, rotation=45)
        plt.legend()
        plt.savefig('imgs/' + args.exp + '_pred_target_dist.png')
        plt.close()


def plot(pred, target, args):
    if args.solver == 'Delay':
        plt.figure('Delay Predictions & Ground Truth')
        plt.title('Delay Predictions & Ground Truth')
        plt.plot(target, label='Target')
        plt.plot(pred, label='Predict')
        plt.xlabel('Node Index')
        plt.ylabel('Delay(ps)')
        plt.legend()
        plt.savefig('imgs/' + args.exp + '_pred_target_comp.png')
        plt.close()
    else:
        plt.figure('Slew Predictions & Ground Truth')
        plt.title('Slew Predictions & Ground Truth')
        plt.plot(target, label='Target')
        plt.plot(pred, label='Predict')
        plt.legend()
        plt.savefig('imgs/' + args.exp + '_pred_target_comp.png')
        plt.close()


def show_timecost(args, sourth_pth):
    spice_runtime = 'csvs/v10_SPICE_timecost.csv'
    tree = sourth_pth
    timecost = np.ones(21)
    with open(spice_runtime, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            size = int(row[0])
            if size < 21:
                if float(row[1]) < timecost[size]:
                    timecost[size] = float(row[1])
    avg_spice_timecost = timecost
    avg_tree_timecost = np.loadtxt(tree, delimiter=',')
    avg_tree_timecost[2] = 0
    dic = {'Circuit Size': np.hstack((np.arange(3,21), np.arange(3,21))),
           'Computation Time': np.hstack((avg_spice_timecost[3:], avg_tree_timecost[3:])),
           'Type': ['SPICE']*18+['Our Model']*18}
    df = pd.DataFrame(dic)

    print(df)
    sns.set()
    plt.figure(figsize=(6, 2))
    sns.set_theme(style="whitegrid", font_scale=2)
    palette = {'SPICE':'green', 'Our Model':'orange'}
    ax = sns.lmplot(df, x='Circuit Size', y='Computation Time', hue='Type', palette=palette, height=6)
    ax.set(xticks=np.arange(2, 21, 2), xticklabels=np.arange(2, 21, 2))
    plt.ylabel("Computation Time(s)")
    plt.savefig('imgs/' + args.exp + "_timecost_comp.png")
    plt.close()


def comp_TC(args):
    tc_acc = pd.read_csv('csvs/' + args.exp + '_acc_' + str(args.loss_type) +'.csv')
    ori_acc = pd.read_csv('csvs/' + args.exp + '_acc_' +'0.csv')
    diff = tc_acc - ori_acc
    mask_acc = diff == 0
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set_theme(style="white", font_scale=3)
    cmap = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=1, as_cmap=True)
    ax1 = sns.heatmap(data=diff, mask=mask_acc, cmap=cmap,
                      cbar_kws={'label': args.solver + 'MAE Difference (ps)', 'shrink': .8}, vmin=-0.1, vmax=0.1)
    ax1.invert_yaxis()
    ax1.set(xticks=np.arange(0, args.leaf_num + 1, 3), xticklabels=np.arange(0, args.leaf_num + 1, 3),
            yticks=np.arange(0, args.max_len + 1, 10), yticklabels=np.arange(0, args.max_len + 1, 10))
    plt.xlabel("Number of Leaf Node", labelpad=32)
    plt.ylabel("Circuit Size",labelpad=32)
    plt.tight_layout()
    plt.xticks(rotation=0)
    figure1 = ax1.get_figure()
    figure1.savefig('imgs/' + args.exp + '_wo_tc_comp.png', dpi=200)
    plt.close()






