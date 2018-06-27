import cython
from libcpp.string cimport string
# from libc.stdlib cimport malloc, free
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


# declare the interface to the C code
cdef extern from "wrapper_interface.cpp":
    ctypedef struct Parameters:
        # Graph structure
        unsigned int nodes_nb
        unsigned int feat_nb
        float neigh_size
        unsigned int neigh_nb
        # General
        unsigned int gridsize
        bint viz
        bint debug
        # PC transformations
        float to_remove
        float occl_pct
        float noise_std

    int compute_graph_feats(string filename,
                            double* node_feats,
                            double* adj_mat,
                            Parameters params)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_graph(filename, **kwargs):  # p):
    cdef Parameters params
    params.nodes_nb = kwargs.get("nodes_nb", 50)
    params.feat_nb = kwargs.get("feat_nb", 352)
    params.neigh_size = kwargs.get("neigh_size", 0.1)
    params.neigh_nb = kwargs.get("neigh_nb", 8)

    params.gridsize = kwargs.get("gridsize", 64)
    params.viz = kwargs.get("viz", False)
    params.debug = kwargs.get("debug", False)

    params.to_remove = kwargs.get("to_remove", 0.)
    params.occl_pct = kwargs.get("occl_pct", 0.)
    params.noise_std = kwargs.get("noise_std", 0.)

    if params.debug:
        print params

    cdef np.ndarray[double, ndim=2, mode="c"] node_feats = np.zeros([params.nodes_nb,
                                                                     params.feat_nb],
                                                                    dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([params.nodes_nb,
                                                                  params.nodes_nb],
                                                                 dtype=np.float64)

    compute_graph_feats(filename, &node_feats[0, 0], &adj_mat[0, 0], params)

    return node_feats, adj_mat
