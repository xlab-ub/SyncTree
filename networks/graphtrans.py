import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import zeropad
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.utils import degree


# Note: We modify the GraphTrans readout layer as our baseline model, the GraphTrans model is from:
# Zhanghao Wu, et al. Representing long-range context for graph neural networks with global attention. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 
# https://github.com/ucbrise/graphtrans.git

def pad_batch(h_node, batch, max_input_len, get_mask=False):
    num_batch = batch[-1] + 1
    num_nodes = []
    masks = []
    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    # logger.info(max(num_nodes))
    max_num_nodes = min(max(num_nodes), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            num_node = max_num_nodes
        padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]

    if get_mask:
        return padded_h_node, src_padding_mask, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x, attn_mask: torch.Tensor = None, valid_input_mask: torch.Tensor = None, mask_value=-1e6):
        """mask should be a 3D tensor of shape (B, T, T)"""
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if attn_mask is not None:
            att = att.masked_fill(attn_mask.unsqueeze(1) == 0, mask_value)
        if valid_input_mask is not None:
            att = att.masked_fill(valid_input_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_ff, n_head, attn_pdrop, resid_pdrop, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_ff),
            nn.GELU(),
            nn.Linear(n_ff, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, attn_mask=None, valid_input_mask=None):
        if self.prenorm:
            x = x + self.attn(self.ln1(x), attn_mask, valid_input_mask)
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x, attn_mask, valid_input_mask))
            x = self.ln2(x + self.mlp(x))
        return x


class TransformerNodeEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.d_model = args.d_model
        self.num_layer = args.num_encoder_layers
        # Creating Transformer Encoder Model
        encoder_layer = nn.TransformerEncoderLayer(
            args.d_model, args.nhead, args.dim_feedforward, args.transformer_dropout, args.transformer_activation
        )
        encoder_norm = nn.LayerNorm(args.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, args.num_encoder_layers, encoder_norm)

        self.norm_input = None
        if args.transformer_norm_input:
            self.norm_input = nn.LayerNorm(args.d_model)
        self.cls_embedding = None
        if args.graph_pooling == "cls":
            self.cls_embedding = nn.Parameter(torch.randn([1, 1, args.d_model], requires_grad=True))

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """

        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)

            zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask)  # (S, B, h_d)

        return transformer_out, src_padding_mask


class MaskedTransformerBlock(nn.Module):
    def __init__(self, n_layer, n_embd, n_ff, n_head, attn_pdrop, resid_pdrop, prenorm=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_ff, n_head, attn_pdrop, resid_pdrop, prenorm) for _ in range(n_layer)])
        # self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def forward(self, x, attn_mask=None, valid_input_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask, valid_input_mask)
        return x


class MaskedOnlyTransformerEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.masked_transformer = MaskedTransformerBlock(
            args.num_encoder_layers_masked,
            args.d_model,
            args.dim_feedforward,
            args.nhead,
            args.transformer_dropout,
            args.transformer_dropout,
        )

    def forward(self, x, attn_mask=None, valid_input_mask=None):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """
        x = self.masked_transformer(x, attn_mask=attn_mask, valid_input_mask=valid_input_mask)
        return x


# GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim: int, edge_encoder_cls):
        """
        emb_dim (int): node embedding dimensionality
        """
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = edge_encoder_cls(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_encoder_cls):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = edge_encoder_cls(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1.0 / deg.view(
            -1, 1
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, args, num_layer, emb_dim, node_encoder, edge_encoder_cls, drop_ratio=0.5, JK="last", residual=False,
                 gnn_type="gin"):
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers
        """
        self.args = args
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim, edge_encoder_cls))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim, edge_encoder_cls))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data, perturb=None):
        x, edge_index, edge_attr, delays = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.delays
        edge_attr = edge_attr.view(-1,1)
        c = batched_data.c
        flat_c = torch.tensor([], dtype=torch.float32)
        for i in range(self.args.batch_size):
            tmp = torch.tensor(c[i])
            flat_c = torch.cat([flat_c, tmp], dim=0)
        delays = np.asarray(delays)
        cut = [delays[i].size for i in range(self.args.batch_size)]
        x = flat_c.view(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)
        if self.node_encoder is not None:
            encoded_node = (
                self.node_encoder(x)
            )
        else:
            encoded_node = x
        tmp = encoded_node + perturb if perturb is not None else encoded_node
        h_list = [tmp]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.JK == "cat":
            node_representation = torch.cat([h_list[0], h_list[-1]], dim=-1)

        return node_representation, cut


def GNNNodeEmbedding(*args, **kwargs):
    return GNN_node(*args, **kwargs)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batched_data, perturb=None):
        raise NotImplementedError

    def epoch_callback(self, epoch):
        return


class GNNTransformer(BaseModel):

    def __init__(self, args, node_encoder, edge_encoder_cls):
        super().__init__()
        self.gnn_node = GNNNodeEmbedding(
            args,
            args.gnn_num_layer,
            args.gnn_emb_dim,
            node_encoder,
            edge_encoder_cls,
            JK=args.gnn_JK,
            drop_ratio=args.gnn_dropout,
            residual=args.gnn_residual,
            gnn_type=args.gnn_type,
        )
        self.args = args
        self.max_seq_len = None
        gnn_emb_dim = 2 * args.gnn_emb_dim if args.gnn_JK == "cat" else args.gnn_emb_dim
        self.gnn2transformer = nn.Linear(gnn_emb_dim, args.d_model)
        self.pos_encoder = PositionalEncoding(args.d_model, dropout=0) if args.pos_encoder else None
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        self.num_encoder_layers = args.num_encoder_layers
        self.num_encoder_layers_masked = args.num_encoder_layers_masked

        self.pooling = args.graph_pooling
        self.graph_pred_linear_list = torch.nn.ModuleList()

        self.max_seq_len = args.max_seq_len

        # Mlp Layers
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(args.trans_mlp_in_dim, args.trans_mlp_hi_dim), nn.ReLU(True)))
        for i in range(args.trans_n_mlps - 2):
            self.mlps.append(nn.Sequential(nn.Linear(args.trans_mlp_hi_dim, args.trans_mlp_hi_dim), nn.ReLU(True)))
        self.mlps.append(nn.Sequential(nn.Linear(args.trans_mlp_hi_dim, args.trans_mlp_out_dim)))

    def forward(self, batched_data, perturb=None):
        h_node, cut = self.gnn_node(batched_data, perturb)
        h_node = self.gnn2transformer(h_node)  # [s, b, d_model]

        padded_h_node, src_padding_mask, num_nodes, mask, max_num_nodes = pad_batch(
            h_node, batched_data.batch, self.args.max_input_len, get_mask=True
        )  # Pad in the front

        # TODO(paras): implement mask
        transformer_out = padded_h_node
        if self.pos_encoder is not None:
            transformer_out = self.pos_encoder(transformer_out)
        if self.num_encoder_layers_masked > 0:
            adj_list = batched_data.adj_list
            padded_adj_list = torch.zeros((len(adj_list), max_num_nodes, max_num_nodes), device=h_node.device)
            for idx, adj_list_item in enumerate(adj_list):
                N, _ = adj_list_item.shape
                padded_adj_list[idx, 0:N, 0:N] = torch.from_numpy(adj_list_item)
            transformer_out = self.masked_transformer_encoder(
                transformer_out.transpose(0, 1), attn_mask=padded_adj_list, valid_input_mask=src_padding_mask
            ).transpose(0, 1)
        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, b, h], [b, s]

        if self.pooling in ["last", "cls"]:
            h_graph = transformer_out[-1]
        elif self.pooling == "mean":
            h_graph = transformer_out.sum(0) / src_padding_mask.sum(-1, keepdim=True)
        else:
            raise NotImplementedError
        h_graph = torch.where(torch.isnan(h_graph), torch.full_like(h_graph, 0), h_graph)
        h_graph = torch.where(torch.isinf(h_graph), torch.full_like(h_graph, 0), h_graph)

        padded_batch_out = zeropad(torch.split(h_node, cut), self.args.max_len, self.args.batch_size)
        batch_out = padded_batch_out[0].unsqueeze(0)
        for i in range(1, self.args.batch_size):
            batch_out = torch.vstack((batch_out, padded_batch_out[i].unsqueeze(0)))

        h_graph_expand = h_graph.unsqueeze(1).repeat(1, self.args.max_len, 1)
        out = torch.cat((batch_out, h_graph_expand), dim=-1)
        out = torch.tensor(out, dtype=torch.float32)
        for i in range(self.args.trans_n_mlps):
            out = self.mlps[i](out)

        return out.squeeze(-1)

        return pred_list

    def epoch_callback(self, epoch):
        # TODO: maybe unfreeze the gnn at the end.
        if self.freeze_gnn is not None and epoch >= self.freeze_gnn:
            for param in self.gnn_node.parameters():
                param.requires_grad = False

    def gnn_node_state(self, state_dict):
        module_name = "gnn_node"
        new_state_dict = dict()
        for k, v in state_dict.items():
            if module_name in k:
                new_key = k.split(".")
                module_index = new_key.index(module_name)
                new_key = ".".join(new_key[module_index + 1:])
                new_state_dict[new_key] = v
        return new_state_dict


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def graph_trans_preprocess(args):
    node_encoder_cls = lambda: nn.Linear(args.num_features, args.gnn_emb_dim)

    def edge_encoder_cls(_):
        def zero(_):
            return 0

        return zero

    return node_encoder_cls, edge_encoder_cls
