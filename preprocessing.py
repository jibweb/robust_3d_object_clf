from functools import partial
import numpy as np

from utils.params import params as p
from py_graph_construction import get_graph, get_graph_3d


# Graph structure
p.define("nodes_nb", 128)
p.define("feat_nb", 4)
p.define("edge_feat_nb", 5)
p.define("neigh_size", 0.15)
p.define("neigh_nb", 5)
p.define("gridsize", 64)
p.define("feats_3d", True)
p.define("edge_feats", False)

# Data transformation
p.define("to_remove", 0.)
p.define("to_keep", 5000)
p.define("occl_pct", 0.)
p.define("noise_std", 0.)
p.define("rotation_deg", 0)

p.define("debug", False)
p.define("viz", False)
p.define("viz_small_spheres", False)


def preprocess_dummy(data):
    return data


def preprocess_adj_to_bias(adj):
    """
     Prepare adjacency matrix by converting it to bias vectors.
     Expected shape: [nodes, nodes]
     Originally from github.com/PetarV-/GAT
    """
    # mt = adj + np.eye(adj.shape[1])
    return -1e9 * (1.0 - adj)


# def graph_preprocess_shot(fn, p):
#     feats, adj = get_graph_feats(fn, **p.__dict__)
#     bias = adj_to_bias(adj)

#     # 2-hop adj matrix
#     # adj_2hop = np.matmul(adj, adj)
#     # adj_2hop = (adj_2hop > 0).astype(adj_2hop.dtype)
#     # bias_2hop = adj_to_bias(adj_2hop)

#     return feats, bias


def preprocess_fpfh(feats):
    max_feats = np.max(feats, axis=1) + 1e-6
    feats = feats / np.repeat(max_feats.reshape((p.nodes_nb, 1)), 33, axis=1)
    return feats


def preprocess_esf3d(feats):
    return np.array(feats)[..., np.newaxis]


def graph_preprocess_3d(fn, p, preprocess_feats, preprocess_adj,
                        preprocess_edge_feats):
    feats, adj, edge_feats, valid_indices = get_graph_3d(fn, **p.__dict__)
    feats = preprocess_feats(feats)
    adj = preprocess_adj(adj)
    edge_feats = preprocess_edge_feats(edge_feats)

    return feats, adj, edge_feats, valid_indices


def graph_preprocess(fn, p, preprocess_feats, preprocess_adj,
                     preprocess_edge_feats):
    feats, adj, edge_feats, valid_indices = get_graph(fn, **p.__dict__)
    feats = preprocess_feats(feats)
    adj = preprocess_adj(adj)
    edge_feats = preprocess_edge_feats(edge_feats)

    return feats, adj, edge_feats, valid_indices


def get_graph_preprocessing_fn(p):
    if p.feats_3d:
        return partial(graph_preprocess_3d, p=p,
                       preprocess_feats=preprocess_esf3d,
                       preprocess_adj=preprocess_adj_to_bias,
                       preprocess_edge_feats=preprocess_dummy)
    else:
        if p.feat_nb == 33:
            return partial(graph_preprocess, p=p,
                           preprocess_feats=preprocess_fpfh,
                           preprocess_adj=preprocess_adj_to_bias,
                           preprocess_edge_feats=preprocess_dummy)
        else:
            return partial(graph_preprocess, p=p,
                           preprocess_feats=preprocess_dummy,
                           preprocess_adj=preprocess_adj_to_bias,
                           preprocess_edge_feats=preprocess_dummy)
