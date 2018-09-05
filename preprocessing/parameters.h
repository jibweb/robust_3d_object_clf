#pragma once


// #define GRIDSIZE 64
// #define GRIDSIZE_H GRIDSIZE/2


struct Parameters
{
  // Graph structure
  unsigned int nodes_nb;
  unsigned int feat_nb;
  unsigned int edge_feat_nb;
  float min_angle_z_normal;
  float neigh_size;
  int neigh_nb; // /!\ Correspond to the number of neighbor of a node when using meshes and the number of points in the neighborhood when using a mesh
  bool feats_3d;
  bool edge_feats;
  bool mesh;
  bool scale;
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
  unsigned int rotation_deg;
};
