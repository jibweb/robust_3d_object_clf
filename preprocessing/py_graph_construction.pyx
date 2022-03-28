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
        unsigned int edge_feat_nb
        float min_angle_z_normal
        float neigh_size
        int neigh_nb
        bint feats_3d
        bint edge_feats
        bint mesh
        bint scale
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
        unsigned int rotation_deg

    int construct_graph(string filename,
                        double* node_feats,
                        double* adj_mat,
                        double* edge_feats,
                        int* valid_indices,
                        Parameters params)

    int construct_graph_nd(string filename,
                           double** node_feats,
                           double* adj_mat,
                           double* edge_feats,
                           int* valid_indices,
                           Parameters params)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_graph(filename, **kwargs):
    cdef Parameters params
    params.nodes_nb = kwargs.get("nodes_nb")
    params.feat_nb = kwargs.get("feat_nb")
    params.edge_feat_nb = kwargs.get("edge_feat_nb")
    params.neigh_size = kwargs.get("neigh_size")
    params.neigh_nb = kwargs.get("neigh_nb")
    params.feats_3d = kwargs.get("feats_3d")
    params.edge_feats = kwargs.get("edge_feats")
    params.mesh = kwargs.get("mesh")
    params.scale = kwargs.get("scale")

    params.gridsize = kwargs.get("gridsize")
    params.viz = kwargs.get("viz")
    params.viz_small_spheres = kwargs.get("viz_small_spheres")
    params.debug = kwargs.get("debug")

    params.to_remove = kwargs.get("to_remove")
    params.to_keep = kwargs.get("to_keep")
    params.occl_pct = kwargs.get("occl_pct")
    params.noise_std = kwargs.get("noise_std")
    params.rotation_deg = kwargs.get("rotation_deg")
    params.min_angle_z_normal = kwargs.get("min_angle_z_normal")

    cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([params.nodes_nb,
                                                                  params.nodes_nb],
                                                                 dtype=np.float64)

    cdef np.ndarray[double, ndim=3, mode="c"] edge_feats_mat = np.zeros([params.nodes_nb,
                                                                         params.nodes_nb,
                                                                         params.edge_feat_nb],
                                                                        dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] node_feats = np.zeros([params.nodes_nb,
                                                                     params.feat_nb],
                                                                    dtype=np.float64)

    cdef np.ndarray[int, ndim=1, mode="c"] valid_indices = np.zeros([params.nodes_nb],
                                                                    dtype=np.int32)

    if params.debug:
        print "### File:", filename

    construct_graph(filename, &node_feats[0, 0], &adj_mat[0, 0], &edge_feats_mat[0, 0, 0], &valid_indices[0], params)
    return node_feats, adj_mat, edge_feats_mat, valid_indices


@cython.boundscheck(False)
@cython.wraparound(False)
def get_graph_nd(filename, **kwargs):
    cdef Parameters params
    params.nodes_nb = kwargs.get("nodes_nb")
    params.feat_nb = kwargs.get("feat_nb")
    params.edge_feat_nb = kwargs.get("edge_feat_nb")
    params.neigh_size = kwargs.get("neigh_size")
    params.neigh_nb = kwargs.get("neigh_nb")
    params.feats_3d = kwargs.get("feats_3d")
    params.edge_feats = kwargs.get("edge_feats")
    params.mesh = kwargs.get("mesh")
    params.scale = kwargs.get("scale")

    params.gridsize = kwargs.get("gridsize")
    params.viz = kwargs.get("viz")
    params.viz_small_spheres = kwargs.get("viz_small_spheres")
    params.debug = kwargs.get("debug")

    params.to_remove = kwargs.get("to_remove")
    params.to_keep = kwargs.get("to_keep")
    params.occl_pct = kwargs.get("occl_pct")
    params.noise_std = kwargs.get("noise_std")
    params.rotation_deg = kwargs.get("rotation_deg")
    params.min_angle_z_normal = kwargs.get("min_angle_z_normal")

    cdef np.ndarray[double, ndim=2, mode="c"] adj_mat = np.zeros([params.nodes_nb,
                                                                  params.nodes_nb],
                                                                 dtype=np.float64)

    cdef np.ndarray[double, ndim=3, mode="c"] edge_feats_mat = np.zeros([params.nodes_nb,
                                                                         params.nodes_nb,
                                                                         params.edge_feat_nb],
                                                                        dtype=np.float64)

    if params.debug:
        print "\n###\n File:", filename
        print params

    if params.feat_nb >= 500:
        node_shape = [params.feat_nb, 6, 1]  # TODO TMP CHANGE !!
    else:
        node_shape = [params.feat_nb, params.feat_nb, params.feat_nb]

    cdef double **node_feats3d_ptr = <double **> malloc(params.nodes_nb*sizeof(double *))
    node_feats3d = []
    cdef np.ndarray[double, ndim=3, mode="c"] temp

    for i in range(params.nodes_nb):
        temp = np.zeros(node_shape,
                        dtype=np.float64)
        node_feats3d_ptr[i] = &temp[0, 0, 0]

        node_feats3d.append(temp)

    cdef np.ndarray[int, ndim=1, mode="c"] valid_indices = np.zeros([params.nodes_nb],
                                                                    dtype=np.int32)

    construct_graph_nd(filename, node_feats3d_ptr, &adj_mat[0, 0], &edge_feats_mat[0, 0, 0], &valid_indices[0], params)
    return node_feats3d, adj_mat, edge_feats_mat, valid_indices