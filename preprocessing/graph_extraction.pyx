import cython
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
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
        bint viz_small_spheres
        bint debug
        # PC transformations
        float to_remove
        unsigned int to_keep
        float occl_pct
        float noise_std

    int compute_graph_feats(string filename,
                            double* node_feats,
                            double* adj_mat,
                            Parameters params)

    int compute_graph_feats3d(string filename,
                              double** node_feats,
                              double* adj_mat,
                              Parameters params)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_graph_feats(filename, **kwargs):
    cdef Parameters params
    params.nodes_nb = kwargs.get("nodes_nb")
    params.feat_nb = kwargs.get("feat_nb")
    params.neigh_size = kwargs.get("neigh_size")
    params.neigh_nb = kwargs.get("neigh_nb")

    params.gridsize = kwargs.get("gridsize")
    params.viz = kwargs.get("viz")
    params.viz_small_spheres = kwargs.get("viz_small_spheres")
    params.debug = kwargs.get("debug")

    params.to_remove = kwargs.get("to_remove")
    params.to_keep = kwargs.get("to_keep")
    params.occl_pct = kwargs.get("occl_pct")
    params.noise_std = kwargs.get("noise_std")

    cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([params.nodes_nb,
                                                                  params.nodes_nb],
                                                                 dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] node_feats = np.zeros([params.nodes_nb,
                                                                     params.feat_nb],
                                                                    dtype=np.float64)

    compute_graph_feats(filename, &node_feats[0, 0], &adj_mat[0, 0], params)
    return node_feats, adj_mat


@cython.boundscheck(False)
@cython.wraparound(False)
def get_graph_feats3d(filename, **kwargs):
    cdef Parameters params
    params.nodes_nb = kwargs.get("nodes_nb")
    params.feat_nb = kwargs.get("feat_nb")
    params.neigh_size = kwargs.get("neigh_size")
    params.neigh_nb = kwargs.get("neigh_nb")

    params.gridsize = kwargs.get("gridsize")
    params.viz = kwargs.get("viz")
    params.viz_small_spheres = kwargs.get("viz_small_spheres")
    params.debug = kwargs.get("debug")

    params.to_remove = kwargs.get("to_remove")
    params.to_keep = kwargs.get("to_keep")
    params.occl_pct = kwargs.get("occl_pct")
    params.noise_std = kwargs.get("noise_std")

    cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([params.nodes_nb,
                                                                  params.nodes_nb],
                                                                 dtype=np.float64)

    cdef double **node_feats3d_ptr = <double **> malloc(params.nodes_nb*sizeof(double *))
    node_feats3d = []
    cdef np.ndarray[double, ndim=3, mode="c"] temp

    for i in range(params.nodes_nb):
        temp = np.zeros([params.feat_nb, params.feat_nb, params.feat_nb],
                        dtype=np.float64)
        node_feats3d_ptr[i]  = &temp[0, 0, 0]
        node_feats3d.append(temp)

    compute_graph_feats3d(filename, node_feats3d_ptr, &adj_mat[0, 0], params)
    return node_feats3d, adj_mat
