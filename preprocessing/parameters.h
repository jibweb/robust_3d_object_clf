#ifndef PARAMS_H
#define PARAMS_H


#define GRIDSIZE 64
#define GRIDSIZE_H GRIDSIZE/2


struct Parameters
{
  // Local params
  unsigned int local_feat_num;
  unsigned int sal_pt_num;
  unsigned int neigh_size;
  // Global params
  unsigned int global_feat_p_num;
  unsigned int global_feat_t_num;
  unsigned int triplet_num;
  unsigned int gridsize;
  // Generic
  bool viz;
  bool debug;
  // PC tranformations
  float to_remove;
  float occl_pct;
  float noise_std;
};

#endif