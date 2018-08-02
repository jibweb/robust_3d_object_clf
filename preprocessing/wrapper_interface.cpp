#include "graph_construction.cpp"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int construct_graph(std::string filename,
                    double* node_feats,
                    double* adj_mat,
                    double* edge_feats,
                    int* valid_indices,
                    Parameters params) {
  // Setup
  GraphConstructor gc(filename, params);
  gc.initialize();

  // Construct the graph
  gc.samplePoints();
  gc.computeAdjacency(adj_mat);
  gc.computeEdgeFeatures(edge_feats);
  gc.computeFeatures1d(node_feats);

  // Post-processing
  gc.correctAdjacencyForValidity(adj_mat);
  gc.getValidIndices(valid_indices);

  return 0;
}

int construct_graph_3d(std::string filename,
                       double** node_feats,
                       double* adj_mat,
                       double* edge_feats,
                       int* valid_indices,
                       Parameters params) {
  // Setup
  GraphConstructor gc(filename, params);
  gc.initialize();

  // Construct the graph
  gc.samplePoints();
  gc.computeAdjacency(adj_mat);
  gc.computeEdgeFeatures(edge_feats);
  gc.computeFeatures3d(node_feats);

  // Post-processing
  gc.correctAdjacencyForValidity(adj_mat);
  gc.getValidIndices(valid_indices);

  return 0;
}
