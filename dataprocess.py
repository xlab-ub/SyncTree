import os
import json
import re
import torch
import networkx as nx
import torch_geometric.utils as pyg_utils
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np


# Using V3 RC data.
# Not consider driving voltage since all circuits has same maximum driving voltage.
# x(ones) edge_index(A), edge_attr([R, C]), y([delays]).

def delay_slew_compose():
    json_path = "traindata/delay/raw"
    json_list = os.listdir(json_path)
    json_list.sort()
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    delays_list = []
    slews_list = []
    id_list = []
    lens = []

    for step in range(len(json_list)):
        f = open(os.path.join(json_path, json_list[step]))
        print(os.path.join(json_path, json_list[step]))
        jsondata = json.load(f)

        nodes_list = re.findall(r"'name': '(\w*)'", str(jsondata))
        lens.append(len(nodes_list))
        from_nodes_list = re.findall(r"'from': '(\w*)'", str(jsondata))
        to_nodes_list = re.findall(r"'to': '(\w*)'", str(jsondata))
        froms = [nodes_list.index(x) for x in from_nodes_list]
        tos = [nodes_list.index(x) for x in to_nodes_list]

        # 1.set edges for directed graph (down to up).
        edges = np.vstack((np.array(tos), np.array(froms)))
        edge_index = torch.tensor(edges, dtype=torch.long)

        # undiredted graph.
        # edges = np.hstack((np.vstack((np.array(froms), np.array(tos))),
        #                    np.vstack((np.array(tos), np.array(froms)))))

        # 2.set node attributes as ones.
        x = torch.ones(len(nodes_list), 1) * 1e-6

        # 3.set [r,c] as edge attributes.
        r = []
        c = []
        for edge in jsondata['edges']:
            r.append(edge['r']) if 'r' in edge else r.append(0)
            c.append(edge['c']) if 'c' in edge else c.append(0)

        # for undirected graph.
        # edge_attr = np.hstack((np.vstack((np.array(r), np.array(c))),
        #                        np.vstack((np.array(r), np.array(c)))))

        # for directed graph (down to up).
        edge_attr = np.vstack((np.array(r), np.array(c)))
        edge_attr = torch.tensor(edge_attr.transpose(), dtype=torch.float32)

        # 4.seperately set delays and slews as targets.
        delays = []
        slews = []
        for delay in jsondata['delays']:
            delays.append(delay['delay'])
        delays = [0, 0] + delays
        delays = np.array(delays) * 1e12

        for slew in jsondata['slews']:
            slews.append(slew['slew'])
        slews.insert(1, 0)
        slews = np.array(slews) * 1e12

        print(step, 'delays', delays, 'slews', slews)

        x_list.append(x)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        delays_list.append(delays)
        slews_list.append(slews)
        id_list.append(step)
    print('Total Graphs:', len(delays_list), 'Num Range of Nodes:', max(lens), min(lens))

    # plot ground truth range
    sample = []
    for i in delays_list:
        for j in i:
            sample.append(j)
    sample = np.array(sample)
    x = [0, 0.1, 1, 10, 100]
    y = []
    for idx in range(len(x) - 1):
        tmp = np.sum((sample > x[idx]) & (sample < x[idx + 1]))
        y.append(tmp)
    y.append(np.sum(sample > x[-1]))
    tick_label = ['0~0.1', '0.1~1', '1~10', '10~100', '100~']
    x = range(5)
    # plt.figure('Delays Distribution')
    # plt.bar(x, y, 0.5)
    # plt.xticks(ticks=x, labels=tick_label, rotation=45)
    # plt.show()

    sample = []
    for i in slews_list:
        for j in i:
            sample.append(j)
    sample = np.array(sample)
    x = [0, 0.1, 1, 10, 100]
    y = []
    for idx in range(len(x) - 1):
        tmp = np.sum((sample > x[idx]) & (sample < x[idx + 1]))
        y.append(tmp)
    y.append(np.sum(sample > x[-1]))
    tick_label = ['0~0.1', '0.1~1', '1~10', '10~100', '100~']
    x = range(5)
    # plt.figure('Slews Distribution')
    # plt.bar(x, y, 0.5)
    # plt.xticks(ticks=x, labels=tick_label, rotation=45)
    # plt.show()

    return x_list, edge_index_list, edge_attr_list, delays_list, slews_list, id_list


# Using V6 R uniform data.
# x([n,len(seq)]) edge_index(A), edge_attr([R]), y([n,len(seq)]).
# node attr of driving cell is the voltate injection sequence, node attr of other vertices is initiallized by zeros.

def vseq_compose():
    json_path = "traindata/vseq/raw"
    json_list = os.listdir(json_path)
    json_list.sort()
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    y_list = []
    vseq_drive_list = []
    mask_list = []
    id_list = []
    node_lens = []

    for step in range(len(json_list)):
        f = open(os.path.join(json_path, json_list[step]))
        jsondata = json.load(f)
        if jsondata['vertices'][0]['edge'] == 'rising':
            print(os.path.join(json_path, json_list[step]))
            nodes_list = re.findall(r"'name': '(\w*)'", str(jsondata))
            node_lens.append(len(nodes_list))
            from_nodes_list = re.findall(r"'from': '(\w*)'", str(jsondata))
            to_nodes_list = re.findall(r"'to': '(\w*)'", str(jsondata))
            froms = [nodes_list.index(x) for x in from_nodes_list]
            tos = [nodes_list.index(x) for x in to_nodes_list]

            # 1.set edges for directed graph (up to down).
            edges = np.vstack((np.array(froms), np.array(tos)))
            edge_index = torch.tensor(edges, dtype=torch.long)

            # undiredted graph.
            # edges = np.hstack((np.vstack((np.array(froms), np.array(tos))),
            #                    np.vstack((np.array(tos), np.array(froms)))))

            # 2.set r as edge attributes.
            r = []
            c = []
            for edge in jsondata['edges']:
                r.append(edge['r']) if 'r' in edge else r.append(0)
                c.append(edge['c']) if 'c' in edge else c.append(0)
            edge_attr = np.vstack((np.array(r), np.array(c)))
            edge_attr = torch.tensor(edge_attr.transpose(), dtype=torch.float32)

            # 3.compose node attributes matrix, 1 for driving node, 0 for other nodes
            x = np.zeros(len(nodes_list))
            x[0] = 1

            # 4.split and padding vseq to composes sentence.
            #   the length of setence is set to 10
            #   each word in sentence
            '''
            vseqs, masks = split_vseq(jsondata, 200)
            for idx in range(len(vseqs)):
                y = vseqs[idx]
                vseq_drive = vseqs[idx][0,:]
                mask = masks[idx]
                x_list.append(x)
                edge_index_list.append(edge_index)
                edge_attr_list.append(edge_attr)
                y_list.append(y)
                vseq_drive_list.append(vseq_drive)
                mask_list.append(mask)
                id_list.append(step)
            '''
            vseq, mask = samp_vseq(jsondata, 200)
            y = vseq
            vseq_drive = vseq[0, :]
            mask = mask
            x_list.append(x)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)
            y_list.append(y)
            vseq_drive_list.append(vseq_drive)
            mask_list.append(mask)
            id_list.append(step)

    print('Total Graphs:', len(y_list), 'Num Range of Nodes:', max(node_lens), min(node_lens))
    return x_list, edge_index_list, edge_attr_list, y_list, vseq_drive_list, mask_list, id_list


# def get_delays(jsondata):
#     t_mid = []
#     for wave in jsondata['voltage_waveforms']:
#         t_v = np.array(wave['time_value_data'])
#         t = t_v[:,0]
#         v = t_v[:,1]
#         v_mid = 0.5 * (v[0] + v[-1])
#         a = np.where(v > v_mid)
#         b = np.where(v < v_mid)
#         vmax = min(v[a])
#         vmin = max(v[b])
#         if wave['edge'] == 'falling':
#             tmin = max(t[a])
#             tmax = min(t[b])

#         else:
#             tmin = max(t[b])
#             tmax = min(t[a])

#         # linear regression to get t where 50% vdd.
#         # use ps as time unit.
#         t_mid.append((tmin + (v_mid - vmin) / (vmax - vmin) * (tmax - tmin)) * 1e12)
#     # insert 0 delay for driving node & ground node.
#     delays = [ t_mid[i] - t_mid[0]  for i in range(1, len(t_mid))]
#     delays = [0, 0] + delays
#     delays = torch.tensor(delays, dtype=torch.float32)
#     return delays


def split_vseq(jsondata, length):
    vseq = []
    vseqs = []
    masks = []
    for wave in jsondata['voltage_waveforms']:
        v = np.array(wave['time_value_data'])[:, 1]
        vseq.append(v)
    vseq = np.asarray(vseq)
    print(vseq.shape)
    n = vseq.shape[1] // length
    if n > 0:
        vseqs = np.split(vseq[:, :n * length], n, axis=1)
        for i in range(n):
            mask = np.ones([vseq.shape[0], length], dtype=bool)
            masks.append(mask)
    tmp = np.pad(vseq[:, n * length:], ((0, 0), (0, length - vseq.shape[1] % length)))
    vseqs.append(tmp)
    mask = tmp != 0
    masks.append(mask)
    masks = np.asarray(masks)
    vseqs = np.asarray(vseqs)
    print(len(vseqs))

    return vseqs, masks


def samp_vseq(jsondata, length):
    vseq = []
    for wave in jsondata['voltage_waveforms']:
        v = np.array(wave['time_value_data'])[:, 1]
        vseq.append(v)
    vseq = np.asarray(vseq)
    # insert 0 for ground node
    vseq = np.insert(vseq, 1, 0, axis=0)
    total_len = vseq.shape[1]
    m = total_len // length
    n = total_len % length
    if m == 0:
        vseq = np.pad(vseq, ((0, 0), (0, length - n)))
    else:
        if n > 0.5 * length:
            samp_len = total_len // (m + 1)
            idx = [i * (m + 1) for i in range(samp_len)]
            vseq = vseq[..., idx]
            vseq = np.pad(vseq, ((0, 0), (0, length - samp_len)))
        else:
            vseq = vseq[:, :length]

    mask = vseq != 0
    print(vseq.shape)
    return vseq, mask


#  for first trial, we use RC graphs to predict time delays and use R graphs to predict voltages.
#  in RC graphs, we don't consider voltages since time delays can be represented by R*C, so no node atrrtibutes in these graphs. (1000 graphs)
#  in R graphs, we use distribution parameters of voltage curve as node attributes.

class makegraphs(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        # generate vseq dataset
        x_list, edge_index_list, edge_attr_list, y_list, vseq_drive_list, mask_list, id_list = vseq_compose()
        for i in range(len(y_list)):
            data = Data(x = x_list[i], edge_index=edge_index_list[i], edge_attr=edge_attr_list[i], y=y_list[i], vseq_drive=vseq_drive_list[i], mask = mask_list[i], id = id_list[i])
            data_list.append(data)
        # generate delay dataset
        # x_list, edge_index_list, edge_attr_list, delays_list, slews_list, id_list = delay_slew_compose()
        # for i in range(len(delays_list)):
        #     data = Data(x=x_list[i], edge_index=edge_index_list[i], edge_attr=edge_attr_list[i], delays=delays_list[i], slews=slews_list[i], id=id_list[i])
        #     data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class makejths(InMemoryDataset):

    def __init__(self, args, root, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        data_pth_list = convert_dataset_to_junction_tree_hierarchy(self.args)
        for i in range(len(data_pth_list)):
            data = Data(x=data_pth_list[i].x, edge_index=data_pth_list[i].edge_index, delays=data_pth_list[i].delays, slews=data_pth_list[i].slews, mask=data_pth_list[i].mask, leaf_mask=data_pth_list[i].leaf_mask, classification_node=data_pth_list[i].classification_node)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])
        


def convert_dataset_to_junction_tree_hierarchy(args):
    """
    Convert a torch.dataset object or a list of torch.Data to a junction tree hierarchies.
    """
    dataset = makegraphs(args.data_pth)
    max_diameter = None
    data_jth_list= []
    step = 0
    for data in dataset:
        print("Processing Graph:", step)
        step += 1
        for i in range(np.size(data.mask)): # iterate every node
            if data.mask[i] == True:
                data_jth = convert_object_graph_to_jth(args, data, node_id=i, radius=args.radius)
                if (args.min_diameter is None or data_jth.diameter >= args.min_diameter) and \
                    (max_diameter is None or data_jth.diameter <= max_diameter):
                    data_jth_list.append(data_jth)
    return data_jth_list


def convert_object_graph_to_jth(args, data, node_id=None, radius=None):
    data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, node_id, None)
    # Save leaf_mask
    leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
    for v, attr in G_jth.nodes('type'):
        if attr == 'node' and G_jth.nodes[v]['clique_has'] == data_jth['classification_node']:
            leaf_mask[v] = True
    data_jth['leaf_mask'] = leaf_mask
    
    return data_jth


def convert_to_networkx_jth(data, node_id=None, radius=None):
    """
    Convert a graph or its subgraph given id and radius (an ego graph) to junction tree hierarchy. The node features in
    the input graph will be copied to the corresponding leaf nodes of the output tree decomposition.
    :param data: torch_geometric.data.Data, input graph
    :param node_id: int, node id in the input graph to be classified (only used when task='node')
    :param radius: int, radius of the ego graph around classification node to be converted (only used when task='node')
    :returns: data_jth, G_jth, root_nodes
    """
    # Convert to networkx graph
    G = pyg_utils.to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
    G = nx.to_undirected(G)

    if radius is not None:
        G_subgraph = nx.ego_graph(G, node_id, radius=radius, undirected=False)
        extracted_id = [i for i in G_subgraph.nodes.keys()]
        G_subgraph = nx.relabel_nodes(G_subgraph, dict(zip(extracted_id, list(range(len(G_subgraph))))), copy=True)
        G = generate_node_labels(G_subgraph)
    else:
        extracted_id = [i for i in G.nodes.keys()]
        G = generate_node_labels(G)
    # index of the classification node in the extracted graph, for computing leaf_mask
    classification_node_id = extracted_id.index(node_id)

    is_clique_graph = True if len(list(G.edges)) == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2 else False

    # Create junction tree hierarchy
    G.graph = {'original': True}
    zero_feature = [0.0] * data.num_node_features
    G_jth, root_nodes = generate_jth(G, zero_feature=zero_feature)

    # Convert back to torch Data (change first clique_has value to avoid TypeError when calling pyg_utils.from_networkx
    if is_clique_graph:  # clique graph
        G_jth.nodes[0]['clique_has'] = 0
    else:
        G_jth.nodes[0]['clique_has'] = [0]
    data_jth = pyg_utils.from_networkx(G_jth)

    try:
        data_jth['diameter'] = nx.diameter(G_jth)
    except nx.NetworkXError:
        data_jth['diameter'] = 0
        print('junction tree hierarchy disconnected.')
        return data_jth

    data_jth['classification_node'] = classification_node_id
    data_jth['delays'] = data.delays[node_id]
    data_jth['slews'] = data.slews[node_id]
    data_jth['mask'] = data.mask[node_id]

    return data_jth, G_jth, root_nodes


def generate_jth(G, zero_feature, remove_edges_every_layer=True):
    """
    This function constructs a junction tree hierarchy tree for graph G. Note: when calling this function from outside,
    make sure to set 'original' attribute of graph G to 'True' and run GenNodeLabels(G) to setup 'clique_has' and 'type'
    attributes for each node before calling this function.
    :param G:               nx.Graph, input graph
    :param zero_feature:    list, feature vector of the clique nodes in the junction tree hierarchy
    :param remove_edges_every_layer: bool, if true, remove edges in the same layer of the JT hierarchy
    :return: JTG, root_nodes
    """
    if len(nx.nodes(G)) == 1 and G.graph['original']:
        JTG = G.copy()
        JTG = nx.relabel_nodes(JTG, {list(nx.nodes(G))[0]: 0})
        return JTG, None

    JTG = nx.algorithms.tree.decomposition.junction_tree(G)
    node_list = [node[0] for node in JTG.nodes.data() if node[1]['type'] == 'clique']
    JTG = nx.bipartite.projected_graph(JTG, node_list)

    # RootNodes = JTG.nodes ## PROBLEM!!!

    # Add clique attributes to JTG nodes.   # Old: set node labels/attributes: #WORKS
    node_index_count = 0
    new_index = {}
    clique_has = {}
    feature_vector = {}
    for clique_node in list(nx.nodes(JTG)):
        clique_has[clique_node] = {"clique_has": generate_clique_labels(G, clique_node)}
        feature_vector[clique_node] = {"x": 0.0}
        new_index[clique_node] = node_index_count
        node_index_count += 1

    nx.set_node_attributes(JTG, clique_has)
    nx.set_node_attributes(JTG, feature_vector)
    JTG = nx.relabel_nodes(JTG, new_index)

    root_nodes = list(nx.nodes(JTG))

    # if G is a clique graph and it is not the original loopy graph
    if len(nx.nodes(JTG)) == 1 and G.graph['original'] is False:
        GE = nx.create_empty_copy(G)
        return GE, list(nx.nodes(GE))  # returns G without any links

    # Otherwise, G is not a clique graph or the original loopy graph
    # Construct the JTHierarcyGraph:
    Clique_Nodes = list(nx.nodes(JTG))
    for Anode in Clique_Nodes:

        SG = nx.subgraph(G, JTG.nodes[Anode]["clique_has"])
        SG.graph['original'] = False  # SG is a subgraph, not the original loopy graph

        if len(nx.nodes(SG)) == 1:
            pass

        elif len(nx.nodes(SG)) == 2:
            U = nx.create_empty_copy(SG)
            new_index = {}
            for n in list(nx.nodes(U)):
                new_index[n] = node_index_count
                node_index_count += 1
            U = nx.relabel_nodes(U, new_index)

            for n in list(nx.nodes(U)):
                U.add_edges_from([(n, Anode)])
            JTG.update(U)

        else:
            Subgraph_Tree, Subgraph_Tree_RootNodes = generate_jth(SG, zero_feature)

            # This part removed the tree structure in a given layer
            if remove_edges_every_layer:
                SGTreeTemp = nx.subgraph(Subgraph_Tree, Subgraph_Tree_RootNodes)
                for an_edge in SGTreeTemp.edges():
                    Subgraph_Tree.remove_edge(*an_edge)
            ##############################################################

            new_index = {}
            RootNodesTemp = []
            for n in list(nx.nodes(Subgraph_Tree)):
                new_index[n] = node_index_count
                if n in Subgraph_Tree_RootNodes:
                    RootNodesTemp.append(node_index_count)
                node_index_count += 1
            U = nx.relabel_nodes(Subgraph_Tree, new_index)

            for n in RootNodesTemp:
                U.add_edges_from([(n, Anode)])
            JTG.update(U)

    return JTG, root_nodes


def generate_node_labels(G):
    """ Add node attributes clique_has and type to all nodes in the input graph. """
    clique_has = {}
    type = {}
    for node in list(nx.nodes(G)):
        clique_has[node] = {"clique_has": node}
        type[node] = {"type": "node"}

    nx.set_node_attributes(G, clique_has)
    nx.set_node_attributes(G, type)
    return G


def generate_clique_labels(G, clique_node):
    """ Generate clique label, which is just the list of nodes in G that it contains. """
    SG = nx.subgraph(G, clique_node)
    return list(SG.nodes())


