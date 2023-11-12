import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from torch_geometric.utils import to_networkx
from networks.delaynet import TransposeTREE, GAT, GCN, GraphSAGE, DeepGCN, Transfer_Net
from networks.graphtrans import GNNTransformer, graph_trans_preprocess
from networks.ntree import NeuralTreeNetwork
from networks.treecircloss import TreeCIRLoss
from networks.circleloss import CircleLoss
from torch_geometric.data import DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError
from dataprocess import makegraphs, makejths
from utils import *
from numpy import *


class GNN4DelaySlew:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        if self.args.wandb:
            wandb.init(entity="yuthu", project=self.args.wandb_proj, name=self.args.exp, config=self.args)
        # define model
        if self.args.model == "GCN":
            self.model = GCN(self.args)
        elif self.args.model == "GAT":
            self.model = GAT(self.args)
        elif self.args.model == "SAGE":
            self.model = GraphSAGE(self.args)
        elif self.args.model == "DeepGCN":
            self.model = DeepGCN(self.args)
        elif self.args.model == "TREE":  # ours
            self.model = TransposeTREE(self.args)
        elif self.args.model == "GraphTrans":
            node_encoder_cls, edge_encoder_cls = graph_trans_preprocess(args)
            node_encoder = node_encoder_cls()
            self.model = GNNTransformer(args=args, node_encoder=node_encoder, edge_encoder_cls=edge_encoder_cls)
        elif self.args.model == "NT":
            self.model = NeuralTreeNetwork(self.args)
        elif self.args.model == "Transfer":
            self.model = Transfer_Net(self.args)
        else:
            print('error')

        if self.args.model != "NT":
            dset = makegraphs(self.args.data_pth)
            tr_dset, va_dset = dset_split(dset, self.args.train_cut, 0.8)
            self.tr_loader = DataLoader(dataset=tr_dset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            self.va_loader = DataLoader(dataset=va_dset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            self.va_loader_copy = DataLoader(dataset=va_dset, batch_size=1, shuffle=True, drop_last=True)
        else:
            dset = makejths(self.args, self.args.jth_pth)
            tr_dset, va_dset = dset_split(dset, self.args.train_cut, 0.8)
            self.tr_loader = DataLoader(dataset=tr_dset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            self.va_loader = DataLoader(dataset=va_dset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            self.va_loader_copy = DataLoader(dataset=va_dset, batch_size=1, shuffle=True, drop_last=True)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.sh_gamma)
        self.TC = TreeCIRLoss(self.args)
        self.C = CircleLoss(self.args)
        self.mape = MeanAbsolutePercentageError()


    def train(self):
        for epoch in range(self.args.epochs):
            for data in self.tr_loader:
                if self.args.solver == 'Delay':
                    target = data.delays
                else:
                    target = data.slews
                    
                if self.args.model == "TREE":
                    pred, embs = self.model(data)
                    mask = data.mask  # mask for leaf nodes
                    filter_pred, filter_target= filter_pred_target(pred, mask, target, self.args)
                    percent = percent_error(filter_pred, filter_target)
                    loss_l1 = F.l1_loss(filter_pred, filter_target)
                    if self.args.loss_type == 0:
                        loss = loss_l1
                        if self.args.wandb:
                            wandb.log({"L1 Train Loss": loss_l1, "Train Loss": loss, "Train Percent Err": percent})
                    elif self.args.loss_type == 1:
                        samples, pos_mask, neg_mask, rd = sample(pred, data, self.args)
                        loss_TC = torch.sum(self.TC(samples, pos_mask, neg_mask, rd))
                        loss = loss_l1 +self.args.lamda*loss_TC
                        if self.args.wandb:
                            wandb.log({"L1 Train Loss": loss_l1, "TC Train Loss": loss_TC, "Train Loss": loss, "Train Percent Err": percent})
                    elif self.args.loss_type == 2:
                        samples, pos_mask, neg_mask, _ = sample(pred, data, self.args)
                        loss_C = torch.sum(self.C(samples, pos_mask, neg_mask))
                        loss = loss_l1 + self.args.lamda * loss_C
                        if self.args.wandb:
                            wandb.log({"L1 Train Loss": loss_l1, "C Train Loss": loss_C, "Train Loss": loss, "Train Percent Err": percent})
                elif self.args.model == "NT":
                    pred = self.model(data)
                    loss_l1 = F.l1_loss(pred, target)
                    percent = torch.mean(torch.abs((pred-target)/target)*100)
                    loss = loss_l1
                    if self.args.wandb:
                        wandb.log({"Train Loss": loss, "Train Percent Err": percent})
                else:
                    pred = self.model(data)
                    mask = data.mask  # mask for leaf nodes
                    filter_pred, filter_target= filter_pred_target(pred, mask, target, self.args)
                    percent = percent_error(filter_pred, filter_target)
                    loss_l1 = F.l1_loss(filter_pred, filter_target)
                    loss = loss_l1
                    if self.args.wandb:
                        wandb.log({"Train Loss": loss, "Train Percent Err": percent})
                    
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # self.scheduler.step()
            # if epoch % 10 == 0:
            #     self.visual_sim()
            # mae, mape, critical_mae, critical_mape = self.eval()
            mae, mape = self.eval()
            if self.args.wandb:
                # wandb.log({"Val MAE": mae, "Val Percent Err": mape, "Critical MAE": critical_mae, "Critical MAPE": critical_mape})
                wandb.log({"Val MAE": mae, "Val Percent Err": mape})
            # print(f'Epoch: {epoch:03d}, Val MAE: {mae:.4f}, Percent Err: {mape:.4f}, Critical MAE: {critical_mae:.4f}, Critical MAPE: {critical_mape:.4f}')
            print(f'Epoch: {epoch:03d}, Val MAE: {mae:.4f}, Percent Err: {mape:.4f}')
        # self.show_err_dist() 
        pth = self.args.save_pth + self.args.exp + ".pth"
        torch.save(self.model, pth)
        self.visual_sim()
        if self.args.elmore_delay and self.args.solver == 'Delay':
            self.show_elmore_delay()
        # if self.args.wandb:
        #     wandb.log({"Acc": wandb.Image('imgs/' + self.args.exp + '_acc_' + str(self.args.loss_type) + '.png'), "Per": wandb.Image('imgs/' + self.args.exp + '_per_' + str(self.args.loss_type) + '.png'), "Sim": wandb.Image('imgs/' + self.args.exp + '_sim_' + str(self.args.loss_type) + '.png')})
        

    def eval(self):
        val_loss = []
        percent_err = []
        critical_mae = []
        critical_mape = []
        critical_target_avg = []
        # time_cost = []
        for data in self.va_loader:
            if self.args.solver == 'Delay':
                target = data.delays
            else:
                target = data.slews
            mask = data.mask  # mask for leaf nodes
            if self.args.model == "TREE":
                # start = time.time()
                pred, embs = self.model(data)
                # cost = time.time() - start
                filter_pred, filter_target = filter_pred_target(pred.detach(), mask, target, self.args)
                loss_l1 = F.l1_loss(filter_pred, filter_target)
                percent = percent_error(filter_pred, filter_target)
                
            elif self.args.model == "NT":
                # start = time.time()
                pred = self.model(data).detach()
                # cost = time.time() - start
                loss_l1 = F.l1_loss(pred, target)
                percent = torch.mean(torch.abs((pred-target)/target)*100)
            else:
                # start = time.time()
                pred = self.model(data)
                # cost = time.time() - start
                filter_pred, filter_target = filter_pred_target(pred.detach(), mask, target, self.args)
                loss_l1 = F.l1_loss(filter_pred, filter_target)
                percent = percent_error(filter_pred, filter_target)
                
            # time_cost.append(cost)
            val_loss.append(loss_l1)
            percent_err.append(percent)

            # critical_index = torch.argmax(filter_target, dim=1)
            # critical_target = torch.gather(filter_target, dim=1, index=critical_index.unsqueeze(1))
            # critical_pred = torch.gather(filter_pred, dim=1, index=critical_index.unsqueeze(1))
            # critical_target_avg.append(torch.mean(critical_target))
            # critical_mae.append(F.l1_loss(critical_target, critical_pred))
            # critical_mape.append(self.mape(critical_target, critical_pred))
            
        # print("Time cost for model evaluation:", sum(time_cost)/len(time_cost), len(time_cost))
        # print('Average of critical target:', mean(critical_target_avg))
        # return mean(val_loss), mean(percent_err), mean(critical_mae), mean(critical_mape)
        return mean(val_loss), mean(percent_err)

    def show_err_dist(self):
        self.err_rcsize_pathlen() 
        self.err_rcsize_leafnum()

    def err_rcsize_pathlen(self): # plot acc/per along with circuit size, num of leafs
        acc_total = np.zeros((self.args.max_len+1, self.args.max_len+1))
        per_total = np.zeros((self.args.max_len+1, self.args.max_len+1))
        fre = np.zeros((self.args.max_len+1, self.args.max_len+1))
        for data in self.va_loader:
            if self.args.solver == 'Delay':
                target = data.delays
            else:
                target = data.slews
            if self.args.model == "TREE":
                pred, embs = self.model(data)
            else:
                pred = self.model(data)
            mask = data.mask
            filter_pred, filter_target = filter_pred_target(pred.detach(), mask, target, self.args)
            for i in range(self.args.batch_size): # iterate each circuit
                n = np.size(target[i])  # get circuit size
                dists = data.dist[i][np.where(mask[i]==True)].tolist()  # get RC path length of leafnodes
                pred = filter_pred[i, :]
                label = filter_target[i, :]
                if torch.sum(pred) == 0:
                    continue;
                percent = percent_error_v(filter_pred[i, :], filter_target[i, :])
                loss = F.l1_loss(pred[pred != 0], label[label != 0], reduce=False)
                for j in range(len(dists)):
                    per_total[n, dists[j]] += percent[j]
                    acc_total[n, dists[j]] += loss[j]
                    fre[n,dists[j]] += 1
        fre[fre == 0] = 1
        acc = np.transpose(acc_total / fre)
        per = np.transpose(per_total / fre)
        df_acc = pd.DataFrame(acc)
        df_per = pd.DataFrame(per)
        mask_acc = acc == 0
        mask_per = per == 0
        df_acc.to_csv('csvs/' + self.args.exp + '_acc_size_len_' + str(self.args.loss_type) +'.csv')
        df_per.to_csv('csvs/' + self.args.exp + '_per_size_len_' + str(self.args.loss_type) +'.csv')
        sns.set(rc={'figure.figsize': (16,12)})
        sns.set_theme(style="white", font_scale=3)
        ax1=sns.heatmap(data=df_acc, mask=mask_acc, cmap="viridis", cbar_kws={'label': self.args.solver + ' Mean Average Error(ps)', 'shrink': .8}, square=True)
        ax1.invert_yaxis()
        if self.args.max_len == 52:
            ax1.set(xticks=np.arange(0,self.args.max_len+1,10), xticklabels=np.arange(0,self.args.max_len+1,10),
                    yticks=np.arange(0, 33, 8), yticklabels=np.arange(0, 33, 8))
        else:
            ax1.set(xticks=np.arange(0,self.args.max_len+1,2), xticklabels=np.arange(0,self.args.max_len+1,2),
                    yticks=np.arange(0, 21, 4), yticklabels=np.arange(0, 21, 4))
        plt.xlabel("Circuit Size")
        plt.ylabel("RC Path Length")
        figure1 = ax1.get_figure()
        figure1.savefig('imgs/' + self.args.exp + '_acc_size_len_' + str(self.args.loss_type) + '.png', dpi=200)
        plt.show()
        plt.close()

        ax2 = sns.heatmap(data=df_per, mask=mask_per, cmap="viridis", cbar_kws={'label': self.args.solver + ' Relative Error(%)', 'shrink': .8}, vmax=100, square=True)
        ax2.invert_yaxis()
        if self.args.max_len == 52:
            ax2.set(xticks=np.arange(0,self.args.max_len+1,10), xticklabels=np.arange(0,self.args.max_len+1,10),
                    yticks=np.arange(0, 33, 8), yticklabels=np.arange(0, 33, 8))
        else:
            ax2.set(xticks=np.arange(0,self.args.max_len+1,2), xticklabels=np.arange(0,self.args.max_len+1,2),
                    yticks=np.arange(0, 21, 4), yticklabels=np.arange(0, 21, 4))
        plt.xlabel("Circuit Size")
        plt.ylabel("RC Path Length")
        figure2 = ax2.get_figure()
        figure2.savefig('imgs/' + self.args.exp + '_per_size_len_' + str(self.args.loss_type) + '.png', dpi=200)
        plt.show()
        plt.close()


    def err_rcsize_leafnum(self): # plot acc/per along with circuit size, num of leafs
        acc_total = np.zeros((self.args.max_len+1, self.args.leaf_num+1))
        per_total = np.zeros((self.args.max_len+1, self.args.leaf_num+1))
        fre = np.zeros((self.args.max_len+1, self.args.leaf_num+1))
        for data in self.va_loader:
            if self.args.solver == 'Delay':
                target = data.delays
            else:
                target = data.slews
            if self.args.model == "TREE":
                pred, embs = self.model(data)
            else:
                pred = self.model(data)
            mask = data.mask
            filter_pred, filter_target = filter_pred_target(pred.detach(), mask, target, self.args)
            for i in range(self.args.batch_size):
                n = np.size(target[i])  # get circuit size
                m = np.size(np.where(mask[i] == True))  # get leaf node number
                loss = F.l1_loss(filter_pred[i, :], filter_target[i, :])
                percent = percent_error(filter_pred[i, :], filter_target[i, :])
                per_total[n, m] += percent
                acc_total[n, m] += loss
                fre[n,m] += 1
        fre[fre == 0] = 1
        acc = acc_total / fre
        per = per_total / fre
        df_acc = pd.DataFrame(acc)
        df_per = pd.DataFrame(per)
        mask_acc = acc == 0
        mask_per = per == 0
        df_acc.to_csv('csvs/' + self.args.exp + '_acc_' + str(self.args.loss_type) +'.csv')
        df_per.to_csv('csvs/' + self.args.exp + '_per_' + str(self.args.loss_type) +'.csv')
        sns.set(rc={'figure.figsize': (16,12)})
        sns.set_theme(style="white", font_scale=4)
        ax1=sns.heatmap(data=df_acc, mask=mask_acc, cmap="viridis", cbar_kws={'label': self.args.solver + ' Mean Average Error(ps)', 'shrink': .8})
        ax1.invert_yaxis()
        ax1.set(xticks=np.arange(0,self.args.leaf_num+1,3), xticklabels=np.arange(0,self.args.leaf_num+1,3),
                yticks=np.arange(0,self.args.max_len+1,2), yticklabels=np.arange(0,self.args.max_len+1,2))
        plt.xlabel("Number of Leaf Node")
        plt.ylabel("Circuit Size")
        figure1 = ax1.get_figure()
        figure1.savefig('imgs/' + self.args.exp + '_acc_' + str(self.args.loss_type) + '.png', dpi=200)
        plt.show()
        plt.close()

        ax2 = sns.heatmap(data=df_per, mask=mask_per, cmap="viridis", cbar_kws={'label': self.args.solver + ' Relative Error(%)', 'shrink': .8}, vmax=100)
        ax2.invert_yaxis()
        # ax2.set(xticks=np.arange(0, self.args.leaf_num + 1, 3), xticklabels=np.arange(0, self.args.leaf_num + 1, 3),
        #         yticks=np.arange(0, self.args.max_len + 1, 10), yticklabels=np.arange(0, self.args.max_len + 1, 10))
        ax2.set(xticks=np.arange(0, self.args.leaf_num + 1, 3), xticklabels=np.arange(0, self.args.leaf_num + 1, 3),
                yticks=np.arange(0,self.args.max_len+1,2), yticklabels=np.arange(0,self.args.max_len+1,2))
        
        plt.xlabel("Number of Leaf Node")
        plt.ylabel("Circuit Size")
        figure2 = ax2.get_figure()
        figure2.savefig('imgs/' + self.args.exp + '_per_' + str(self.args.loss_type) + '.png', dpi=200)
        plt.show()
        plt.close()

            
    def show_elmore_delay(self):
        elmore_acc_total = np.zeros((self.args.max_len+1, self.args.leaf_num+1))
        elmore_per_total = np.zeros((self.args.max_len+1, self.args.leaf_num+1))
        fre = np.zeros((self.args.max_len+1, self.args.leaf_num+1))
        for data in self.va_loader_copy:
            target = data.delays
            pred, embs = self.model(data)
            mask = data.mask
            filter_pred, filter_target = filter_pred_target(pred.detach(), mask, target, self.args)
            n = np.size(target[0])  # get circuit size
            m = np.size(np.where(mask[0] == True))  # get leaf node number
            elmore_delays = elmore_delay(data)
            target = filter_target[0, :]
            elmore_mae = F.l1_loss(elmore_delays, target[target != 0])
            elmore_mape = percent_error(elmore_delays, target[target != 0])
            elmore_acc_total[n, m] += elmore_mae
            elmore_per_total[n, m] += elmore_mape
        fre[fre == 0] = 1
        elmore_acc = elmore_acc_total / fre
        elmore_per = elmore_per_total / fre
        elmore_df_acc = pd.DataFrame(elmore_acc)
        elmore_df_per = pd.DataFrame(elmore_per)
        elmore_mask_acc = elmore_acc == 0
        elmore_mask_per = elmore_per == 0
        elmore_df_acc.to_csv('csvs/' + '_elmore_acc_' + '.csv')
        elmore_df_per.to_csv('csvs/' + '_elmore_per_' + '.csv')
        sns.set(rc={'figure.figsize': (16,12)})
        sns.set_theme(style="white", font_scale=4)
        ax1=sns.heatmap(data=elmore_df_acc, mask=elmore_mask_acc, cmap="viridis", cbar_kws={'label': 'Elmore ' + self.args.solver + ' MAE(ps)', 'shrink': .8})
        ax1.invert_yaxis()
        ax1.set(xticks=np.arange(0,self.args.leaf_num+1,3), xticklabels=np.arange(0,self.args.leaf_num+1,3),
                yticks=np.arange(0,self.args.max_len+1,10), yticklabels=np.arange(0,self.args.max_len+1,10))
        plt.xlabel("Number of Leaf Node")
        plt.ylabel("Circuit Size")
        figure1 = ax1.get_figure()
        figure1.savefig('imgs/' + 'elmore_acc.png', dpi=200)
        plt.close()


        ax2 = sns.heatmap(data=elmore_df_per, mask=elmore_mask_per, cmap="viridis", cbar_kws={'label': 'Elmore ' + self.args.solver + ' MAPE(%)', 'shrink': .8}, vmax=100)
        ax2.invert_yaxis()
        ax2.set(xticks=np.arange(0, self.args.leaf_num + 1, 3), xticklabels=np.arange(0, self.args.leaf_num + 1, 3),
                yticks=np.arange(0, self.args.max_len + 1, 10), yticklabels=np.arange(0, self.args.max_len + 1, 10))
        plt.xlabel("Number of Leaf Node")
        plt.ylabel("Circuit Size")
        figure2 = ax2.get_figure()
        figure2.savefig('imgs/' + 'elmore_per.png', dpi=200)
        plt.close()


    def visual_sim(self):
        sim = torch.tensor([], dtype=torch.float32)
        max_rd = 0
        min_rd = 1
        for data in self.va_loader:
            if self.args.model == "TREE":
                pred, embs = self.model(data)
            else:
                pred = self.model(data)
            samples, pos_mask, neg_mask, rd = sample(pred, data, self.args)
            if self.args.loss_type == 1:  # TC
                tmp_max = torch.max(rd)
                tmp_min = torch.min(rd)
                if tmp_max > max_rd:
                    max_rd = tmp_max
                if tmp_min < min_rd:
                    min_rd = tmp_min
            pos_mask_idx = torch.where(pos_mask == 1)
            neg_mask_idx = torch.where(neg_mask == 1)
            sp =samples[pos_mask_idx].reshape(self.args.B, -1)
            sn =samples[neg_mask_idx].reshape(self.args.B, -1)
            N = sp.shape[1]
            M = sn.shape[1]
            for i in range(self.args.B):
                neg = sn[i].repeat(N, 1).flatten()
                pos = sp[i].repeat(M, 1).t().flatten()   # [N, M]
                tmp = torch.cat((neg.view(-1, 1), pos.view(-1, 1)), dim=1)
                sim = torch.cat((sim, tmp), dim=0)
        if self.args.loss_type == 1:
            print(max_rd, min_rd)
        sim = sim.detach().numpy()
        sim = pd.DataFrame(sim)
        sns.set(rc={'figure.figsize': (16, 16)})
        sns.set_theme(style="white", font_scale=2)
        cmap = sns.cubehelix_palette(start=1.3, light=1, as_cmap=True)
        g = sns.JointGrid(data=sim, x=sim[0], y=sim[1], space=0)
        g.plot_joint(sns.kdeplot, fill=True, thresh=0, levels=20, cmap="rocket")
        g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=10)
        g.set_axis_labels(xlabel="$s_{n}$", ylabel="$s_{p}$", size=24)
        plt.savefig('imgs/' + self.args.exp + '_sim_' + str(self.args.loss_type) + '.png', dpi=200)
        plt.close()















