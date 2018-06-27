from functools import partial
import numpy as np
# import os
from utils.params import params as p

# os.system("python setup.py build_ext -i")
from graph_extraction import get_graph


# Graph structure
p.define("nodes_nb", 64)
p.define("feat_nb", 352)
p.define("neigh_size", 0.2)
p.define("neigh_nb", 8)

# Data transformation
p.define("to_remove", 0.)
p.define("occl_pct", 0.)
p.define("noise_std", 0.)

p.define("debug", False)
p.define("viz", False)


def adj_to_bias(adj):
    """
     Prepare adjacency matrix by converting it to bias vectors.
     Expected shape: [nodes, nodes]
     Originally from github.com/PetarV-/GAT
    """
    mt = adj + np.eye(adj.shape[1])
    return -1e9 * (1.0 - mt)


def graph_preprocess_vanilla(fn, p):
    feats, adj = get_graph(fn, **p.__dict__)
    bias = adj_to_bias(adj)

    return feats, bias


def get_graph_preprocessing_fn(p):
    return partial(graph_preprocess_vanilla, p=p)
