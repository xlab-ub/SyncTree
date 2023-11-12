import torch
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


# Split dataset into training and test dataset.
# def loader_split(dataloader, ratio):
#     index = 0
#     length = len(dataloader)
#     tr_set = []
#     te_set = []
#     for data in dataloader:
#         if index >= (length * ratio):
#             te_set.append(data)
#         else:
#             tr_set.append(data)
#         index += 1
#     return tr_set, te_set
def dset_split(dataset, train_cut=1, ratio=0.8):
    length = dataset.len()
    idx = list(range(length))
    random.shuffle(idx)

    split = int(length * ratio)
    tr_dset = [dataset.get(i) for i in idx[:int(split*train_cut)]]
    va_dset = [dataset.get(i) for i in idx[split:]]
    
    return tr_dset, va_dset

def jths_split(dataset, train_cut=1, ratio=0.8):
    length = len(dataset)
    idx = list(range(length))
    random.shuffle(idx)

    split = int(length * ratio)
    tr_dset = [dataset[i] for i in idx[:int(split*train_cut)]]
    va_dset = [dataset[i] for i in idx[split:]]
    
    return tr_dset, va_dset

def make_params_string(params):
    str_result = ""
    for i in params.keys():
        str_result = str_result + str(i) + '=' + str(params[i]) + '\n'
    return str_result[:-1]


def parser_from_dict(dic):
    parser = argparse.ArgumentParser()
    for k, v in dic.items():
        parser.add_argument("--" + k, default=v)
    args = parser.parse_args()

    return args


# zero padding
def zeropad(data, length, batch_size):
    pad = []
    d = data[0].size()[1]
    for i in range(batch_size):
        padding = torch.zeros((length - data[i].size()[0], d))
        pad.append(torch.cat((data[i], padding)))
    return pad


# flatten finnal embedding
def dataflat(feats, masks, dists, path_enc, args):
    cut = [mask.size for mask in masks]
    flat_feats = torch.tensor([])
    flat_masks = torch.tensor([])
    flat_dists = torch.tensor([])
    flat_path_encs = torch.tensor([])
    segs = []
    for i in range(len(masks)):
        feat = feats[i][:cut[i]]
        mask = torch.tensor(masks[i])
        dist = torch.tensor(dists[i])
        enc = torch.tensor(path_enc[i])
        flat_feats = torch.cat((flat_feats, feat), dim=0)
        flat_masks = torch.cat((flat_masks, mask), dim=0)
        flat_dists = torch.cat((flat_dists, dist), dim=0)
        flat_path_encs = torch.cat((flat_path_encs, enc), dim=0)
    for i in range(len(masks)):
        tmp = 0
        for j in range(0, i+1):
            tmp += cut[j]
        segs.append(tmp)
    segs = torch.tensor(segs)

    return flat_feats, flat_masks, flat_dists, segs, flat_path_encs


def gen_mask(mask, args):
    mask_leaf = torch.tensor([], dtype=torch.float32)
    for i in range(len(mask)):
        tmp = mask[i]
        tmp = np.pad(tmp, (0, args.max_len - tmp.size))
        tmp = torch.tensor(tmp, dtype=torch.float32).unsqueeze(0)  # (1, max_len)
        mask_leaf = torch.cat([mask_leaf, tmp], dim=0)  # (batch_size, max_len)

    return mask_leaf


def filter_pred_target(pred, mask, target, args):
    # make mask
    mask_leaf = gen_mask(mask, args)

    # filter to only contain predictions at leaf nodes
    filter_pred = torch.mul(pred, mask_leaf)

    padded_target = torch.tensor([], dtype=torch.float32)
    for i in range(len(mask)):
        tmp = target[i].T
        tmp = np.pad(tmp, (0, args.max_len - tmp.size))
        tmp = torch.tensor(tmp, dtype=torch.float32).unsqueeze(0)
        padded_target = torch.cat([padded_target, tmp], dim=0)

    # filter to only contain target at leaf nodes
    filter_target = torch.mul(padded_target, mask_leaf)
    return filter_pred, filter_target


def sample(batchfeats, data, args):
    i = 0
    mat = torch.tensor([])
    circuit_size = torch.tensor([])
    pos_mask = torch.tensor([])
    neg_mask = torch.tensor([])
    rd = torch.tensor([])
    feats, leaf_mask, dist, segs, path_encs = dataflat(batchfeats, data.mask, data.dist, data.path_enc, args)
    leafs = torch.where(leaf_mask == True)
    while i<args.B:
        target_idx = torch.tensor(np.random.choice(leafs[0].numpy(), 1))
        target_feat = feats[target_idx]
        pos_idx, pos_err = pick_pos(target_idx, segs, data, args)
        neg_idx, neg_err, size = pick_neg(target_idx, segs, data, args)
        if pos_err or neg_err:
            continue
        else:
            i+=1
            pos_feats = feats[pos_idx]
            neg_feats = feats[neg_idx]
            target_path_r = path_encs[target_idx][:, 0]
            pos_path_r = path_encs[pos_idx][:, 0]
            neg_path_r = path_encs[neg_idx][:, 0]
            sp = cal_sim(pos_feats, target_feat)
            sn = cal_sim(neg_feats, target_feat)
            mat_unit = torch.cat((sp, sn), dim=0)
            pos_mask_unit = torch.cat((torch.ones_like(sp), torch.zeros_like(sn)), dim=0)
            neg_mask_unit = torch.cat((torch.zeros_like(sp), torch.ones_like(sn)), dim=0)
            rd_unit = cal_rd(target_path_r, pos_path_r, neg_path_r)
            mat = torch.cat((mat, mat_unit.unsqueeze(0)), dim=0)
            pos_mask = torch.cat((pos_mask, pos_mask_unit.unsqueeze(0)), dim=0)
            neg_mask = torch.cat((neg_mask, neg_mask_unit.unsqueeze(0)), dim=0)
            rd = torch.cat((rd, rd_unit.unsqueeze(0)), dim=0)  # [B, N, M]
            circuit_size = torch.cat((circuit_size, torch.tensor(size).unsqueeze(0)), dim=0)
    circuit_size = circuit_size.view(-1,1)
    # factor= 0.2/circuit_size
    # mat = torch.exp(-factor*mat)
    # mat = 1/(1+2*mat)  # for risc-v
    mat = 1 / (1 + 0.05 * mat)
    return mat, pos_mask, neg_mask, rd


def pick_pos(target, segs, data, args):
    # edge_index: downtoup dir
    # set node within dist as positive node
    target_idx_in_batch = torch.where(segs > target)[0][0]  # target idx in batch
    if target_idx_in_batch == 0:
        target_idx_in_x = target
    else:
        target_idx_in_x = target - segs[target_idx_in_batch - 1]
    G = Data(x=torch.range(0, data[target_idx_in_batch].x.numel() - 1).reshape(-1, 1),
             edge_index=data[target_idx_in_batch].edge_index)
    g = to_networkx(G)
    pos = list(nx.shortest_path(g, int(target_idx_in_x), list(g.nodes)[0]))[1:args.dist + 1]  # select nodes within dist
    if len(list(g.nodes)) > args.M + args.dist + 1:
        if target_idx_in_batch == 0:
            pos = torch.tensor(pos)
        else:
            pos = torch.tensor(pos) + segs[target_idx_in_batch - 1]
        pos_err = False
    else:
        pos = None
        pos_err = True

    return pos, pos_err


def pick_neg(target, segs, data, args):
    target_idx_in_batch = torch.where(segs > target)[0][0]  # target idx in batch
    if target_idx_in_batch == 0:
        target_idx_in_x = target
    else:
        target_idx_in_x = target - segs[target_idx_in_batch - 1]
    G = Data(x=torch.range(0,data[target_idx_in_batch].x.numel()-1).reshape(-1,1), edge_index=data[target_idx_in_batch].edge_index)
    g = to_networkx(G)
    neg = list(nx.shortest_path(g, int(target_idx_in_x), list(g.nodes)[0]))[args.dist + 1:-1]  # exclude start, parent,end nodes
    size = len(list(g.nodes))
    if len(neg) > args.M:
        if target_idx_in_batch == 0:
            neg = torch.tensor(np.random.choice(neg, args.M, replace=False))
        else:
            neg = torch.tensor(np.random.choice(neg, args.M, replace=False)) + segs[target_idx_in_batch - 1]
        neg_err = False
    else:
        neg_err = True
        neg = None

    return neg, neg_err, size


def cal_sim(sample, target):
    # cosin = nn.CosineSimilarity(dim=0, eps=1e-6)
    # sim = []
    # for i in range(sample.shape[0]):
    #     sim.append(cosin(sample[i], target))
    sim = torch.abs(sample - target)  # if use final output

    return sim


def cal_rd(t_r, p_rs, n_rs):
    rd = torch.tensor([], dtype=torch.float32)
    p_rd = torch.abs(p_rs - t_r)
    n_rd = torch.abs(n_rs - t_r)
    # replace all zeros in rd as 1
    p_rd = torch.where(p_rd== 0, torch.full_like(p_rd, 1), p_rd)
    n_rd = torch.where(n_rd == 0, torch.full_like(n_rd, 1), n_rd)
    for i in range(p_rd.numel()):
        tmp = p_rd[i] / n_rd
        rd = torch.cat((rd, tmp.unsqueeze(0)), dim=0)

    return rd   # [N, M]


def percent_error(pred, target):
    target = target.numpy()
    target = np.ma.masked_equal(target, 0).compressed()
    pred = pred.detach().numpy()
    pred = np.ma.masked_equal(pred, 0).compressed()
    if target.size != pred.size:
        return False
    else:
        percent = np.abs((pred-target)/target)*100
        return np.mean(percent)


def percent_error_v(pred, target):
    target = target.numpy()
    target = np.ma.masked_equal(target, 0).compressed()
    pred = pred.detach().numpy()
    pred = np.ma.masked_equal(pred, 0).compressed()
    if target.size != pred.size:
        return False
    else:
        percent = np.abs((pred-target)/target)*100
        return percent

def elmore_delay(data):
    elmore_delay = []
    graph = Data(x=data.x, edge_index=data.edge_index)
    G = to_networkx(graph)
    G = G.to_undirected()
    paths = nx.shortest_path(G, source=0)
    leafs = np.where(data.mask[0] == True)[0].tolist()
    r = data.r[0]
    c = data.c[0]
            
    for leaf in leafs:
        # the shortest path from the source to the current leaf.
        target_pth = paths[leaf]
        rc = 0
        for node, pth in paths.items():
            common = list(set(target_pth) & set(pth))
            common.remove(0)
            if len(common):
                r_com = [ r[i] for i in common ]
                rc += sum(r_com) * c[node]
        elmore_delay.append(rc)

    return torch.tensor(elmore_delay)