import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv, GCNConv, GraphConv, GENConv
from torch_geometric.nn.models import DeepGCNLayer
from torch_geometric.nn.dense.linear import Linear
from torch.nn import LayerNorm, Linear, ReLU
from utils import zeropad


class TransposeTREE(nn.Module):
    def __init__(self, args):
        super(TransposeTREE, self).__init__()
        self.args = args
        # Conv Net
        self.toup = torch.nn.ModuleList()
        self.toup.append(GATConv(args.conv_in_dim, args.conv_hi_dim, edge_dim=args.edge_dim))
        for i in range(args.n_convs-1):
            self.toup.append(GATConv(args.conv_hi_dim, args.conv_hi_dim, edge_dim=args.edge_dim))

        self.todown = torch.nn.ModuleList()
        for i in range(args.n_convs-1):
            self.todown.append(GATConv(args.conv_hi_dim, args.conv_hi_dim, edge_dim=args.edge_dim, add_self_loops=False))
        self.todown.append(GATConv(args.conv_hi_dim, args.conv_hi_dim, edge_dim=args.edge_dim, add_self_loops=False))

        # Linear Transform to combine updates from toup net and todown net.
        self.lin = torch.nn.ModuleList()
        for i in range(args.n_convs-1):
            self.lin.append(Linear(args.conv_hi_dim, args.conv_hi_dim))
        self.lin.append(Linear(args.conv_hi_dim, args.conv_out_dim))

        # Mlp Layers
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(args.mlp_in_dim, args.mlp_hi_dim), nn.ReLU(True)))
        for i in range(args.n_mlps - 2):
            self.mlps.append(nn.Sequential(nn.Linear(args.mlp_hi_dim, args.mlp_hi_dim), nn.ReLU(True)))
        self.mlps.append(nn.Sequential(nn.Linear(args.mlp_hi_dim, args.mlp_out_dim)))

    def forward(self, data):
        # data processing
        x_to_up, edge_idx_to_up, edge_attr_to_up, delays = data.x, data.edge_index, data.edge_attr, data.delays
        batch_size = len(delays)
        r, c = data.r, data.c
        padded_r = torch.tensor([], dtype=torch.float32)
        padded_c = torch.tensor([], dtype=torch.float32)
        flat_c = torch.tensor([], dtype=torch.float32)
        for i in range(batch_size):
            tmp = torch.tensor(c[i])
            flat_c = torch.cat([flat_c, tmp], dim=0)
        # for i in range(batch_size):
        #     tmp = c[i].T
        #     tmp = np.pad(tmp, (0, self.args.max_len - tmp.size))
        #     tmp = torch.tensor(tmp, dtype=torch.float32).unsqueeze(0)
        #     padded_c = torch.cat([padded_c, tmp], dim=0)
        # for i in range(batch_size):
        #     tmp = r[i].T
        #     tmp = np.pad(tmp, (0, self.args.max_len - tmp.size))
        #     tmp = torch.tensor(tmp, dtype=torch.float32).unsqueeze(0)
        #     padded_r = torch.cat([padded_r, tmp], dim=0)
        # path_emb = torch.tensor([], dtype=torch.float32)
        # for i in range(batch_size):
        #     tmp = path_enc[i]
        #     tmp = np.pad(tmp, ((0, self.args.max_len - tmp.shape[0]), (0, 0)), mode='constant', constant_values=0)
        #     path_emb = torch.cat((path_emb, torch.tensor(tmp).unsqueeze(0)), dim=0)
        delays = np.asarray(delays)
        idx = torch.LongTensor([1, 0])
        edge_idx_to_down, edge_attr_to_down = edge_idx_to_up[idx], edge_attr_to_up

        # get node embeddings.
        x_to_up = flat_c.view(-1, 1)
        x_to_up = torch.tensor(x_to_up, dtype=torch.float32)
        for i in range(self.args.n_convs):
            x_to_up = self.toup[i](x_to_up, edge_idx_to_up, edge_attr_to_up).relu()

        x_to_down = self.todown[0](x_to_up, edge_idx_to_down, edge_attr_to_down).relu()
        x = x_to_down + x_to_up
        for i in range(self.args.n_convs-1):
            x = self.lin[i](x).relu()
            x_to_down = self.todown[i](x, edge_idx_to_down, edge_attr_to_down).relu()
            x = x_to_down + x_to_up
        final_emb = x
        x = self.lin[-1](x)

        # split and pad embeddings into [Batch, Max_len, Out_dim].
        cut = [delays[i].size for i in range(batch_size)]
        padded_batch_out = zeropad(torch.split(x, cut), self.args.max_len, batch_size)
        padded_batch_emb = zeropad(torch.split(final_emb, cut), self.args.max_len, batch_size)
        batch_out = padded_batch_out[0].unsqueeze(0)
        batch_emb = padded_batch_emb[0].unsqueeze(0)
        for i in range(1, batch_size):
            batch_out = torch.vstack((batch_out, padded_batch_out[i].unsqueeze(0)))
            batch_emb = torch.vstack((batch_emb, padded_batch_emb[i].unsqueeze(0)))
        out = batch_out - batch_out[:, 0, :].unsqueeze(1).repeat(1, self.args.max_len, 1)
        # out = torch.cat((batch_out, padded_c.unsqueeze(-1)), dim=-1)
        #
        # out = torch.tensor(out, dtype=torch.float32)
        # for i in range(self.args.n_mlps):
        #     out = self.mlps[i](out)
        # out = out - out[:, 0, :].unsqueeze(1).repeat(1, self.args.max_len, 1)

        return out.squeeze(-1), batch_emb


class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = args
        # Convoluton Layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(args.conv_in_dim, args.conv_hi_dim))
        for i in range(args.n_convs - 2):
            self.convs.append(GATConv(args.conv_hi_dim, args.conv_hi_dim))
        self.convs.append(GATConv(args.conv_hi_dim, args.conv_out_dim))

        # Mlp Layers
        self.mlps = torch.nn.ModuleList()
        for i in range(args.n_mlps):
            self.mlps.append(nn.Sequential(nn.Linear(args.max_len, args.max_len),
                                           nn.ReLU(True)))
        # Readout Layers
        self.readout = nn.Linear(args.max_len, args.max_len)

    def forward(self, data):
        x, edge_index, edge_attr, delays = data.x, data.edge_index, data.edge_attr, data.delays
        batch_size = len(delays)
        c = data.c
        edge_attr = edge_attr.view(-1,1)
        flat_c = torch.tensor([], dtype=torch.float32)
        for i in range(batch_size):
            tmp = torch.tensor(c[i])
            flat_c = torch.cat([flat_c, tmp], dim=0)
        delays = np.asarray(delays)

        x = flat_c.view(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)
        for i in range(self.args.n_convs):
            x = self.convs[i](x, edge_index, edge_attr)
            x.relu()

        cut = [delays[i].size for i in range(batch_size)]
        padded_batch_out = zeropad(torch.split(x, cut), self.args.max_len, batch_size)

        batch_out = padded_batch_out[0].unsqueeze(0)
        for i in range(1, batch_size):
            batch_out = torch.vstack((batch_out, padded_batch_out[i].unsqueeze(0)))
        batch_out = batch_out - batch_out[:, 0, :].unsqueeze(1).repeat(1, self.args.max_len, 1)

        out = batch_out

        return out.squeeze(-1)


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        # Convoluton Layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(args.conv_in_dim, args.conv_hi_dim))
        for i in range(args.n_convs - 2):
            self.convs.append(GCNConv(args.conv_hi_dim, args.conv_hi_dim))
        self.convs.append(GCNConv(args.conv_hi_dim, args.conv_out_dim))

        # Mlp Layers
        self.mlps = torch.nn.ModuleList()
        for i in range(args.n_mlps):
            self.mlps.append(nn.Sequential(nn.Linear(args.max_len, args.max_len),
                                           nn.ReLU(True)))
        # Readout Layers
        self.readout = nn.Linear(args.max_len, args.max_len)

    def forward(self, data):
        x, edge_index, edge_weight, delays = data.x, data.edge_index, data.edge_attr, data.delays
        batch_size = len(delays)
        c = data.c
        flat_c = torch.tensor([], dtype=torch.float32)
        for i in range(batch_size):
            tmp = torch.tensor(c[i])
            flat_c = torch.cat([flat_c, tmp], dim=0)
        delays = np.asarray(delays)

        x = flat_c.view(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)
        for i in range(self.args.n_convs):
            x = self.convs[i](x, edge_index, edge_weight)
            x.relu()

        cut = [delays[i].size for i in range(batch_size)]
        padded_batch_out = zeropad(torch.split(x, cut), self.args.max_len, batch_size)

        batch_out = padded_batch_out[0].unsqueeze(0)
        for i in range(1, batch_size):
            batch_out = torch.vstack((batch_out, padded_batch_out[i].unsqueeze(0)))
        batch_out = batch_out - batch_out[:, 0, :].unsqueeze(1).repeat(1, self.args.max_len, 1)

        out = batch_out

        return out.squeeze(-1)


class GraphSAGE(nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.args = args
        # Convoluton Layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(args.conv_in_dim, args.conv_hi_dim))
        for i in range(args.n_convs - 2):
            self.convs.append(GraphConv(args.conv_hi_dim, args.conv_hi_dim))
        self.convs.append(GraphConv(args.conv_hi_dim, args.conv_out_dim))

        # Mlp Layers
        self.mlps = torch.nn.ModuleList()
        for i in range(args.n_mlps):
            self.mlps.append(nn.Sequential(nn.Linear(args.max_len, args.max_len),
                                           nn.ReLU(True)))
        # Readout Layers
        self.readout = nn.Linear(args.max_len, args.max_len)

    def forward(self, data):
        x, edge_index, edge_weight, delays = data.x, data.edge_index, data.edge_attr, data.delays
        batch_size = len(delays)
        c = data.c
        edge_weight = edge_weight.squeeze()
        flat_c = torch.tensor([], dtype=torch.float32)
        for i in range(batch_size):
            tmp = torch.tensor(c[i])
            flat_c = torch.cat([flat_c, tmp], dim=0)
        delays = np.asarray(delays)

        x = flat_c.view(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)
        for i in range(self.args.n_convs):
            x = self.convs[i](x, edge_index, edge_weight)
            x.relu()

        cut = [delays[i].size for i in range(batch_size)]
        padded_batch_out = zeropad(torch.split(x, cut), self.args.max_len, batch_size)
        batch_out = padded_batch_out[0].unsqueeze(0)
        for i in range(1, batch_size):
            batch_out = torch.vstack((batch_out, padded_batch_out[i].unsqueeze(0)))
        batch_out = batch_out - batch_out[:, 0, :].unsqueeze(1).repeat(1, self.args.max_len, 1)

        out = batch_out

        return out.squeeze(-1)


class DeepGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeepGCN, self).__init__()
        self.args = args

        self.node_encoder = Linear(args.conv_in_dim, args.conv_hi_dim)
        self.edge_encoder = Linear(args.conv_in_dim, args.conv_hi_dim)

        self.convs = torch.nn.ModuleList()
        for i in range(self.args.n_convs):
            conv = GENConv(args.conv_hi_dim, args.conv_hi_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(args.conv_hi_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.convs.append(layer)

        self.lin = Linear(args.conv_hi_dim, args.conv_out_dim)

    def forward(self, data):
        x, edge_index, edge_weight, delays = data.x, data.edge_index, data.edge_attr, data.delays
        batch_size = len(delays)
        c = data.c
        flat_c = torch.tensor([], dtype=torch.float32)
        for i in range(batch_size):
            tmp = torch.tensor(c[i])
            flat_c = torch.cat([flat_c, tmp], dim=0)
        delays = np.asarray(delays)

        x = flat_c.view(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_weight.reshape(-1, 1))

        x = self.convs[0].conv(x, edge_index, edge_attr)

        for i in range(1, self.args.n_convs):
            x = self.convs[i](x, edge_index, edge_attr)

        x = self.convs[0].act(self.convs[0].norm(x))
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        cut = [delays[i].size for i in range(batch_size)]
        padded_batch_out = zeropad(torch.split(x, cut), self.args.max_len, batch_size)
        batch_out = padded_batch_out[0].unsqueeze(0)
        for i in range(1, batch_size):
            batch_out = torch.vstack((batch_out, padded_batch_out[i].unsqueeze(0)))
        batch_out = batch_out - batch_out[:, 0, :].unsqueeze(1).repeat(1, self.args.max_len, 1)

        out = batch_out

        return out.squeeze(-1)


class Transfer_Net(nn.Module):
    def __init__(self, args):
        super(Transfer_Net, self).__init__()
        self.args = args
        self.gnn = TransposeTREE(args)
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(args.transfer_mlp_in_dim, args.transfer_mlp_hi_dim), nn.ReLU(True)))
        for i in range(args.transfer_n_mlps - 2):
            self.mlps.append(nn.Sequential(nn.Linear(args.transfer_mlp_hi_dim, args.transfer_mlp_hi_dim), nn.ReLU(True)))
        self.mlps.append(nn.Sequential(nn.Linear(args.transfer_mlp_hi_dim, args.transfer_mlp_out_dim)))

    def forward(self, data):
        self.gnn.load_state_dict(torch.load(self.args.transfer_gnn_model_pth), strict=False)
        pred, embs = self.gnn(data)
        out = embs
        for i in range(self.args.transfer_n_mlps):
            out = self.mlps[i](out)
        out = out - out[:, 0, :].unsqueeze(1).repeat(1, self.args.max_len, 1)

        return out.squeeze(-1)





