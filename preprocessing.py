from functools import partial
import numpy as np
import os
from utils.params import params as p

os.system("python setup.py build_ext -i")
from graph_extraction import get_graph


# General parameters
p.define("local_feat_num", 3)
p.define("global_feat_p_num", 5)
p.define("global_feat_t_num", 4)

# Data transformation
p.define("to_remove", 0.)
p.define("occl_pct", 0.)
p.define("noise_std", 0.)


def adj_to_bias(adj):
    """
     Prepare adjacency matrix by converting it to bias vectors.
     Expected shape: [nodes, nodes]
     Originally from github.com/PetarV-/GAT
    """
    mt = adj + np.eye(adj.shape[1])
    return -1e9 * (1.0 - mt)


def graph_preprocess_vanilla(fn):
    feats, adj = get_graph(fn)
    bias = adj_to_bias(adj)

    return feats, bias


def get_graph_preprocessing_fn(p):
    return partial(graph_preprocess_vanilla, p=p)
