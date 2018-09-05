#include "mesh_graph_construction.h"

#include <pcl/visualization/pcl_visualizer.h>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INTERFACES /////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MeshGraphConstructor::samplePoints() {
  ScopeTime t("Fake Point sampling computation", params_.debug);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MeshGraphConstructor::computeAdjacency(double* adj_mat) {
  ScopeTime t("Adjacency matrix computation", params_.debug);
  bfsSamplingAdjacency(adj_mat);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADJACENCY ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MeshGraphConstructor::bfsSamplingAdjacency(double* adj_mat) {
  // Prepare the values for the sampling procedure
  srand (static_cast<unsigned int> (time (0)));
  std::vector<bool> visited(pc_->points.size(), false);
  uint nb_visited = 0;
  std::vector<std::vector<int> > neighborhood(pc_->points.size());

  for (uint node_idx=0; node_idx < nodes_elts_.size(); node_idx++) {
    // Setup for BFS
    std::vector<bool> visited_local(pc_->points.size());
    int nb_visited_local = 0;
    std::deque<int> queue;

    if (nb_visited == pc_->points.size())
      break;

    // select a node and enqueue it
    int rdn_idx;
    do {
      rdn_idx = rand() % pc_->points.size();
    } while (visited[rdn_idx]);

    if (!visited[rdn_idx])
      nb_visited++;

    visited[rdn_idx] = true;
    visited_local[rdn_idx] = true;
    neighborhood[rdn_idx].push_back(node_idx);
    queue.push_back(rdn_idx);
    sampled_indices_.push_back(rdn_idx);


    // BFS over the graph to extract the neighborhood
    while(!queue.empty() && nb_visited_local < params_.neigh_nb)
    {
      // Dequeue a vertex from queue and print it
      int s = queue.front();
      queue.pop_front();
      nb_visited_local++;
      if (nb_visited_local < params_.neigh_nb/2)
        nodes_elts_[node_idx].push_back(s);

      int nb_pt_idx;
      for (uint nb_idx=0; nb_idx < adj_list_[s].size(); nb_idx++) {
        nb_pt_idx = adj_list_[s][nb_idx];
        if (!visited[nb_pt_idx])
          nb_visited++;

        if (!visited_local[nb_pt_idx]) {
          visited[nb_pt_idx] = true;
          visited_local[nb_pt_idx] = true;

          // Fill in the adjacency matrix
          for (uint i=0; i < neighborhood[nb_pt_idx].size(); i++) {
            int node_idx2 = neighborhood[nb_pt_idx][i];
            adj_mat[node_idx*params_.nodes_nb + node_idx2] = true;
            adj_mat[node_idx2*params_.nodes_nb + node_idx] = true;
          }
          neighborhood[nb_pt_idx].push_back(node_idx);

          queue.push_back(nb_pt_idx);
        }
      }
    } // while queue not empty
  } // for each node


  // Update the valid indices vector
  for (uint i=0; i < sampled_indices_.size(); i++) {
    valid_indices_[i] = true;
    adj_mat[i*params_.nodes_nb + i] = true;
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// VIZ /////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void MeshGraphConstructor::viz() {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPolygonMesh(*mesh_,"meshes",0);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  while (!viewer->wasStopped ()){
      viewer->spinOnce (100);
  }
}