#pragma once

#include <pcl/io/pcd_io.h>

#include "parameters.h"
#include "augmentation_preprocessing.cpp"
#include "occupancy.cpp"
#include "scope_time.h"

class PointCloudGraphConstructor
{
protected:
  std::string filename_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_;
  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree_;
  std::vector<std::vector<int> > nodes_elts_;
  std::vector<int> sampled_indices_;
  std::vector<bool> valid_indices_;
  std::vector<std::vector<std::vector<int> > > lut_;
  double scale_;

  Parameters params_;


  // 1d features
  void shotNodeFeatures(double* result);
  void fpfhNodeFeatures(double* result);
  void pointCoordsNodeFeatures(double* result);
  void dummyNodeFeatures(double* result);

  // nd features
  void esf3dNodeFeatures(double** result);
  void lEsfNodeFeatures(double** result);

  // Adjacency matrix construction method
  void occupancyAdjacency(double* adj_mat);

  // Edge features
  void pointPairEdgeFeatures(double* edge_feats);
  void lrfEdgeFeatures(double* edge_feats);
  void coordsEdgeFeatures(double* edge_feats);
  void rotZEdgeFeatures(double* edge_feats);

public:
  PointCloudGraphConstructor(std::string filename, Parameters params) :
    filename_(filename),
    pc_(new pcl::PointCloud<pcl::PointXYZINormal>),
    tree_(new pcl::search::KdTree<pcl::PointXYZINormal>),
    params_(params) {}

  void initialize() {
    ScopeTime t("Initialization (PointCloudGraphConstructor)", params_.debug);

    // Read the point cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (filename_.c_str(), *pc_) == -1) {
      PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
      return;
    }


    // Data augmentation
    Eigen::Vector4f centroid;
    scale_ = scale_points_unit_sphere (*pc_, params_.gridsize/2, centroid);
    params_.neigh_size = params_.neigh_size * params_.gridsize/2;
    augment_data(pc_, params_);

    if (params_.scale && params_.debug)
      std::cout << "Scale: " << scale_ << std::endl;


    // Initialize the tree
    tree_->setInputCloud (pc_);


    // Initialize the valid indices
    for (uint i=0; i<params_.nodes_nb; i++)
      valid_indices_.push_back(false);


    // Prepare the voxel grid
    lut_.resize (params_.gridsize);
    for (uint i = 0; i < params_.gridsize; ++i) {
        lut_[i].resize (params_.gridsize);
        for (uint j = 0; j < params_.gridsize; ++j)
          lut_[i][j].resize (params_.gridsize);
    }

    voxelize (*pc_, lut_, params_.gridsize);
  };

  virtual void computeEdgeFeatures(double* edge_feats);
  virtual void computeAdjacency(double* adj_mat);
  virtual void correctAdjacencyForValidity(double* adj_mat);
  virtual void getValidIndices(int* valid_indices);
  virtual void computeFeatures1d(double* node_feats);
  virtual void computeFeatures3d(double** node_feats);
  virtual void samplePoints();
  virtual void viz(double* adj_mat);
};
