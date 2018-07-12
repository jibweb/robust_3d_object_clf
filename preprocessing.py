from functools import partial
import numpy as np
# import os
from utils.params import params as p

# os.system("python setup.py build_ext -i")
from graph_extraction import get_graph_feats, get_graph_feats3d


# Graph structure
p.define("nodes_nb", 64)
p.define("feat_nb", 352)
p.define("neigh_size", 0.1)
p.define("neigh_nb", 5)
p.define("gridsize", 64)
p.define("feats_3d", False)

# Data transformation
p.define("to_remove", 0.)
p.define("to_keep", 10000)
p.define("occl_pct", 0.)
p.define("noise_std", 0.)

p.define("debug", False)
p.define("viz", False)
p.define("viz_small_spheres", False)


def adj_to_bias(adj):
    """
     Prepare adjacency matrix by converting it to bias vectors.
     Expected shape: [nodes, nodes]
     Originally from github.com/PetarV-/GAT
    """
    # mt = adj + np.eye(adj.shape[1])
    return -1e9 * (1.0 - adj)


def graph_preprocess_shot(fn, p):
    feats, adj = get_graph_feats(fn, **p.__dict__)
    bias = adj_to_bias(adj)

    # 2-hop adj matrix
    # adj_2hop = np.matmul(adj, adj)
    # adj_2hop = (adj_2hop > 0).astype(adj_2hop.dtype)
    # bias_2hop = adj_to_bias(adj_2hop)

    return feats, bias


def graph_preprocess_fpfh(fn, p):
    feats, adj = get_graph_feats(fn, **p.__dict__)
    bias = adj_to_bias(adj)
    max_feats = np.max(feats, axis=1) + 1e-6
    feats = feats / np.repeat(max_feats.reshape((p.nodes_nb, 1)), 33, axis=1)
    return feats, bias


def graph_preprocess_esf3d(fn, p):
    feats, adj = get_graph_feats3d(fn, **p.__dict__)
    bias = adj_to_bias(adj)
    return np.array(feats)[..., np.newaxis], bias


def get_graph_preprocessing_fn(p):
    if p.feat_nb == 352:
        return partial(graph_preprocess_shot, p=p)
    elif p.feat_nb == 33:
        return partial(graph_preprocess_fpfh, p=p)
    elif p.feats_3d:
        return partial(graph_preprocess_esf3d, p=p)
