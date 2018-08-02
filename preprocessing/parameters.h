#ifndef PARAMS_H
#define PARAMS_H


// #define GRIDSIZE 64
// #define GRIDSIZE_H GRIDSIZE/2


struct Parameters
{
  // Graph structure
  unsigned int nodes_nb;
  unsigned int feat_nb;
  unsigned int edge_feat_nb;
  float neigh_size;
  int neigh_nb;
  bool feats_3d;
  bool edge_feats;
  // General
  unsigned int gridsize;
  bool viz;
  bool viz_small_spheres;
  bool debug;
  // PC tranformations
  float to_remove;
  unsigned int to_keep;
  float occl_pct;
  float noise_std;
};

#endif