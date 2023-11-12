import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.norm.batch_norm import BatchNorm
from utils import zeropad

# We use the Neural Tree model as one of our baselines. The Neural Tree model is from:
# Rajat Talak, Siyi Hu, Lisa Peng, and Luca Carlone. Neural trees for learning on graphs. Advances in Neural Information Processing Systems, 34:26395–26408, 2021.
# https://github.com/MIT-SPARK/neural_tree.git

def build_conv_layer(conv_block, input_dim, hidden_dim):

    if conv_block == 'GCN':
        return pyg_nn.GCNConv(input_dim, hidden_dim)
    elif conv_block == 'GIN':
        return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim)), eps=0., train_eps=True)
    elif conv_block == 'GraphSAGE':
        return pyg_nn.SAGEConv(input_dim, hidden_dim, normalize=False, bias=True)
    else:
        return NotImplemented


def build_GAT_conv_layers(input_dim, hidden_dims, heads, concats, dropout=0.):

    assert len(hidden_dims) == len(heads)
    assert len(hidden_dims) == len(concats)
    convs = nn.ModuleList()
    convs.append(pyg_nn.GATConv(input_dim, hidden_dims[0], heads=heads[0], concat=concats[0], dropout=dropout))
    for i in range(1, len(hidden_dims)):
        if concats[i - 1]:
            convs.append(pyg_nn.GATConv(hidden_dims[i - 1] * heads[i - 1], hidden_dims[i], heads=heads[i],
                                        concat=concats[i], dropout=dropout))
        else:
            convs.append(pyg_nn.GATConv(hidden_dims[i - 1], hidden_dims[i], heads=heads[i], concat=concats[i],
                                        dropout=dropout))
    return convs


class BasicNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, conv_block='GCN', hidden_dim=None, num_layers=None,
                 GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25):
        super(BasicNetwork, self).__init__()
        self.conv_block = conv_block
        self.num_layers = num_layers if conv_block != 'GAT' else len(GAT_heads)
        self.dropout = dropout
        self.need_postmp = isinstance(output_dim, tuple)

        # message passing
        self.convs = nn.ModuleList()
        if self.conv_block != 'GAT':  # GAT dimensions are different than others
            self.convs.append(build_conv_layer(self.conv_block, input_dim, hidden_dim))
            if self.need_postmp:
                for _ in range(1, self.num_layers):
                    self.convs.append(build_conv_layer(self.conv_block, hidden_dim, hidden_dim))
            else:
                for _ in range(1, self.num_layers - 1):
                    self.convs.append(build_conv_layer(self.conv_block, hidden_dim, hidden_dim))
                self.convs.append(build_conv_layer(self.conv_block, hidden_dim, output_dim))

        else:
            if self.need_postmp:
                self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims, GAT_heads, GAT_concats,
                                                   dropout=dropout)
            else:
                self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims + [output_dim], GAT_heads,
                                                   GAT_concats, dropout=dropout)

        # batch normalization
        if self.conv_block == 'GIN':
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms.append(BatchNorm(hidden_dim))

        # post message passing
        if self.need_postmp:
            if self.conv_block != 'GAT':
                final_hidden_dim = hidden_dim
            else:
                final_hidden_dim = GAT_hidden_dims[-1] * GAT_heads[-1] if GAT_concats[-1] else GAT_hidden_dims[-1]
            self.post_mp = nn.ModuleList()
            for dim in output_dim:
                self.post_mp.append(nn.Linear(final_hidden_dim, dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            raise RuntimeError('No node feature')

        if not self.need_postmp:  # pre-iteration dropout for citation networks (might not be necessary in some case)
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index=edge_index)
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                if self.conv_block == 'GIN':
                    x = self.batch_norms[i](x)
                x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.need_postmp:
            x = F.relu(x) if self.conv_block != 'GAT' else F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            return tuple(self.post_mp[i](x) for i in range(len(self.post_mp)))
        else:
            return x

    def loss(self, pred, label, mask=None):
        if mask is None:
            return F.cross_entropy(pred, label)
        else:
            return sum(F.cross_entropy(pred[i][mask[i], :], label[mask[i]]) for i in range(len(mask))
                       if mask[i].sum().item() > 0)


class NeuralTreeNetwork(BasicNetwork):
    def __init__(self, args):
        self.args = args
        super(NeuralTreeNetwork, self).__init__(self.args.input_dim, self.args.output_dim, self.args.conv_block, self.args.hidden_dim, self.args.num_layers,
                                                self.args.GAT_hidden_dims, self.args.GAT_heads,self.args.GAT_concats, self.args.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_hat = x.view(-1,1)
        x_hat = F.dropout(x_hat, p=self.args.dropout)
        for i in range(self.args.num_layers):
            x_hat = self.convs[i](x_hat, edge_index=edge_index).relu()
            if i != self.args.num_layers - 1:    # activation and dropout, except for the last iteration
                x_hat = F.relu(x_hat)
                x_hat = F.dropout(x_hat, p=self.args.dropout)

        x_hat = x_hat[data.leaf_mask, :]

        return x_hat.squeeze(-1)
        

