#include "point_cloud_graph_construction.cpp"
#include "mesh_graph_construction.cpp"


///////////////////////////////////////////////////////////////////////////////
int construct_graph(std::string filename,
                    double* node_feats,
                    double* adj_mat,
                    double* edge_feats,
                    int* valid_indices,
                    Parameters params) {
  if (params.mesh) {
    MeshGraphConstructor mgc(filename, params);
    // Setup
    mgc.initialize();

    // Construct the graph
    mgc.samplePoints();
    mgc.computeAdjacency(adj_mat);
    mgc.computeEdgeFeatures(edge_feats);
    mgc.computeFeatures1d(node_feats);

    // Post-processing
    mgc.correctAdjacencyForValidity(adj_mat);
    mgc.getValidIndices(valid_indices);

    if (params.viz)
      mgc.viz();
  } else {
    PointCloudGraphConstructor pcgc(filename, params);
    // Setup
    pcgc.initialize();

    // Construct the graph
    pcgc.samplePoints();
    pcgc.computeAdjacency(adj_mat);
    pcgc.computeEdgeFeatures(edge_feats);
    pcgc.computeFeatures1d(node_feats);

    // Post-processing
    pcgc.correctAdjacencyForValidity(adj_mat);
    pcgc.getValidIndices(valid_indices);

    if (params.viz)
      pcgc.viz(adj_mat);
  }

  return 0;
}


///////////////////////////////////////////////////////////////////////////////
int construct_graph_nd(std::string filename,
                       double** node_feats,
                       double* adj_mat,
                       double* edge_feats,
                       int* valid_indices,
                       Parameters params) {
  if (params.mesh) {
    MeshGraphConstructor mgc(filename, params);
    // Setup
    mgc.initialize();

    // Construct the graph
    mgc.samplePoints();
    mgc.computeAdjacency(adj_mat);
    mgc.computeEdgeFeatures(edge_feats);
    mgc.computeFeatures3d(node_feats);

    // Post-processing
    mgc.correctAdjacencyForValidity(adj_mat);
    mgc.getValidIndices(valid_indices);

    if (params.viz)
      mgc.viz();
  } else {
    PointCloudGraphConstructor pcgc(filename, params);
    // Setup
    pcgc.initialize();
    // std::cout << "a" << std::endl;

    // Construct the graph
    pcgc.samplePoints();
    // std::cout << "b" << std::endl;
    pcgc.computeAdjacency(adj_mat);
    // std::cout << "c" << std::endl;
    pcgc.computeEdgeFeatures(edge_feats);
    // std::cout << "d" << std::endl;
    pcgc.computeFeatures3d(node_feats);
    // std::cout << "e" << std::endl;

    // Post-processing
    pcgc.correctAdjacencyForValidity(adj_mat);
    pcgc.getValidIndices(valid_indices);
    // std::cout << "f" << std::endl;

    if (params.viz)
      pcgc.viz(adj_mat);
  }

  return 0;
}


///////////////////////////////////////////////////////////////////////////////
int
main(int argc, char* argv[]) {
  Parameters params;
  params.nodes_nb = 16;
  params.feat_nb = 800;
  params.edge_feat_nb = 3;
  params.min_angle_z_normal = 10;
  params.neigh_size = 0.401;
  params.neigh_nb = 4;
  params.feats_3d = true;
  params.edge_feats = true;
  params.mesh = false;
  // General
  params.gridsize = 64;
  params.viz = false;
  params.viz_small_spheres = true;
  params.debug = true;
  // PC tranformations
  params.to_remove = 0.;
  params.to_keep = 20000;
  params.occl_pct = 0.;
  params.noise_std = 0.;
  params.rotation_deg = 0;

  std::string filename = "/home/jbweibel/dataset/ModelNet/modelnet40_manually_aligned_TrainPc/range_hood/range_hood_0065_dist_2.000000_full_wnormals_wattention.pcd";

  double **node_feats;
  node_feats = new double *[params.nodes_nb];
  for(int i = 0; i <params.nodes_nb; i++)
    node_feats[i] = new double[params.feat_nb*6];

  for (uint i=0; i<params.nodes_nb; i++)
    for (uint j=0; j<params.feat_nb*6; j++)
      node_feats[i][j] = 0;

  double adj_mat[params.nodes_nb*params.nodes_nb];
  for (uint i=0; i<params.nodes_nb*params.nodes_nb; i++)
    adj_mat[i] = 0;

  double edge_feats[params.nodes_nb*params.nodes_nb*params.edge_feat_nb];
  for (uint i=0; i<params.nodes_nb*params.nodes_nb*params.edge_feat_nb; i++)
    edge_feats[i] = 0;

  int valid_indices[params.nodes_nb];
  for (uint i=0; i<params.nodes_nb; i++)
    valid_indices[i] = 0;

  construct_graph_nd(filename, node_feats, adj_mat, edge_feats, valid_indices, params);

  return 0;
}
