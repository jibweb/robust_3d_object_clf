import cython
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


# declare the interface to the C code
cdef extern from "wrapper_interface.cpp":
    ctypedef struct Parameters:
        # Local params
        unsigned int local_feat_num
        unsigned int sal_pt_num
        unsigned int neigh_size
        # Global params
        unsigned int global_feat_p_num
        unsigned int global_feat_t_num
        unsigned int triplet_num
        unsigned int gridsize
        # Generic
        bint viz
        bint debug
        # PC transformations
        float to_remove
        float occl_pct
        float noise_std

    int compute_graph_feats(string filename,
                            double** local_feats,
                            double* global_feats_p,
                            double* global_feats_t,
                            int* valid_sal_pt_num,
                            Parameters params)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_graph(filename, **kwargs):  # p):
    cdef Parameters params
    params.local_feat_num = 3
    params.sal_pt_num = kwargs.get("sal_pt_num", 50)
    params.neigh_size = kwargs.get("neigh_size", 64)

    params.global_feat_p_num = 5
    params.global_feat_t_num = 4
    params.triplet_num = kwargs.get("triplet_num", 2000)
    params.gridsize = kwargs.get("gridsize", 64)

    params.viz = kwargs.get("viz", False)
    params.debug = kwargs.get("debug", False)

    params.to_remove = kwargs.get("to_remove", 0.)
    params.occl_pct = kwargs.get("occl_pct", 0.)
    params.noise_std = kwargs.get("noise_std", 0.)

    # //////////////////////////////
    # params.feat_size = p.feat_size
    # params.to_remove = p.to_remove
    # params.occl_pct = p.occl_pct
    # params.noise_std = p.noise_std
    # //////////////////////////////

    cdef double **local_feats_ptr = <double **> malloc(params.sal_pt_num*sizeof(double *))
    local_feats = []
    cdef np.ndarray[double, ndim=2, mode="c"] temp

    for i in range(params.sal_pt_num):
        temp = np.zeros([params.neigh_size,
                         params.local_feat_num],
                        dtype=np.float64)
        local_feats_ptr[i]  = &temp[0,0]
        local_feats.append(temp)

    cdef np.ndarray[double, ndim=2, mode="c"] global_feats_p = np.zeros([3*params.triplet_num,
                                                                         params.global_feat_p_num],
                                                                        dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] global_feats_t = np.zeros([params.triplet_num,
                                                                         params.global_feat_t_num],
                                                                        dtype=np.float64)

    cdef np.ndarray[int, ndim=1, mode="c"] valid_sal_pt_num = np.zeros([1], dtype=np.int32)

    compute_graph_feats(filename, local_feats_ptr, &global_feats_p[0, 0], &global_feats_t[0,0], &valid_sal_pt_num[0], params)

    return local_feats, global_feats_p, global_feats_t, valid_sal_pt_num[0]
