#pragma once

#include <pcl/io/ply_io.h>

#include "point_cloud_graph_construction.h"

class MeshGraphConstructor : public PointCloudGraphConstructor
{
protected:
  pcl::PolygonMesh::Ptr mesh_;
  std::vector<std::vector<int> > adj_list_;

  // Adjacency matrix construction method
  void bfsSamplingAdjacency(double* adj_mat);

public:
  MeshGraphConstructor(std::string filename, Parameters params) :
    PointCloudGraphConstructor(filename, params), mesh_(new pcl::PolygonMesh) {}


  void initialize() {
    ScopeTime t("Initialization (MeshGraphConstructor)", params_.debug);

    // Read the point cloud
    if (pcl::io::loadPLYFile(filename_.c_str(), *mesh_) == -1) {
      PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
      return;
    }


    std::string pc_filename = filename_.substr(0, filename_.size() - 4) + ".pcd";
    // Read the point cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (pc_filename.c_str(), *pc_) == -1) {
      PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
      return;
    }


    if (params_.debug) {
      std::cout << "PolygonMesh: " << mesh_->polygons.size() << " triangles" << std::endl;
      std::cout << "PC size: " << pc_->points.size() << std::endl;
    }

    // Data augmentation
    // Eigen::Vector4f centroid;
    // scale_points_unit_sphere (*pc_, params_.gridsize/2, centroid);
    // params_.neigh_size = params_.neigh_size * params_.gridsize/2;
    // augment_data(pc_, params_);

    // Initialize the tree
    tree_->setInputCloud (pc_);

    nodes_elts_.resize(params_.nodes_nb);
    for (uint i=0; i < nodes_elts_.size(); i++)
      nodes_elts_[i].reserve(params_.neigh_nb);


    // Allocate space for the adjacency list
    adj_list_.resize(pc_->points.size());
    for (uint i=0; i < adj_list_.size(); i++)
      adj_list_[i].reserve(16);


    // Fill in the adjacency list
    for (uint t=0; t < mesh_->polygons.size(); t++) {
      pcl::Vertices& triangle = mesh_->polygons[t];

      adj_list_[triangle.vertices[0]].push_back(triangle.vertices[1]);
      adj_list_[triangle.vertices[1]].push_back(triangle.vertices[0]);

      adj_list_[triangle.vertices[1]].push_back(triangle.vertices[2]);
      adj_list_[triangle.vertices[2]].push_back(triangle.vertices[1]);

      adj_list_[triangle.vertices[2]].push_back(triangle.vertices[0]);
      adj_list_[triangle.vertices[0]].push_back(triangle.vertices[2]);
    }



    // Initialize the valid indices
    valid_indices_.resize(params_.nodes_nb);
    // for (uint i=0; i<params_.nodes_nb; i++)
    //   valid_indices_.push_back(false);


    // Prepare the voxel grid
    lut_.resize (params_.gridsize);
    for (uint i = 0; i < params_.gridsize; ++i) {
        lut_[i].resize (params_.gridsize);
        for (uint j = 0; j < params_.gridsize; ++j)
          lut_[i][j].resize (params_.gridsize);
    }

    voxelize (*pc_, lut_, params_.gridsize);
  };

  virtual void samplePoints();
  virtual void computeAdjacency(double* adj_mat);
  virtual void viz();
};
