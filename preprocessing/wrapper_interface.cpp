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
      pcgc.viz();
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

    // Construct the graph
    pcgc.samplePoints();
    pcgc.computeAdjacency(adj_mat);
    pcgc.computeEdgeFeatures(edge_feats);
    pcgc.computeFeatures3d(node_feats);

    // Post-processing
    pcgc.correctAdjacencyForValidity(adj_mat);
    pcgc.getValidIndices(valid_indices);

    if (params.viz)
      pcgc.viz();
  }

  return 0;
}


///////////////////////////////////////////////////////////////////////////////
// int construct_mesh_graph_3d(std::string filename,
//                        double** node_feats,
//                        double* adj_mat,
//                        double* edge_feats,
//                        int* valid_indices,
//                        Parameters params) {
//   ScopeTime t("Full construction", params.debug);
//   // Setup
//   MeshGraphConstructor mgc(filename, params);
//   mgc.initialize();

//   // Construct the graph
//   mgc.samplePoints();
//   mgc.computeAdjacency(adj_mat);
//   mgc.computeEdgeFeatures(edge_feats);
//   mgc.computeFeatures3d(node_feats);

//   // Post-processing
//   mgc.correctAdjacencyForValidity(adj_mat);
//   mgc.getValidIndices(valid_indices);

//   return 0;
// }
