#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

#include "parameters.h"
#include "augmentation_preprocessing.cpp"
#include "occupancy.cpp"
#include "scope_time.h"

class GraphConstructor
{
private:
  std::string filename_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_;
  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree_;
  std::vector<int> sampled_indices_;
  std::vector<bool> valid_indices_;
  std::vector<std::vector<std::vector<int> > > lut_;

  Parameters params_;

  // 1d features
  void shotNodeFeatures(double* result);
  void fpfhNodeFeatures(double* result);
  void pointCoordsNodeFeatures(double* result);
  void dummyNodeFeatures(double* result);

  // 3d features
  void esf3dNodeFeatures(double** result);

  // Adjacency matrix construction method
  void occupancyAdjacency(double* adj_mat);

  // Edge features
  void pointPairEdgeFeatures(double* edge_feats);
  void lrfEdgeFeatures(double* edge_feats);

public:
  GraphConstructor(std::string filename, Parameters params) :
    filename_(filename),
    pc_(new pcl::PointCloud<pcl::PointXYZINormal>),
    tree_(new pcl::search::KdTree<pcl::PointXYZINormal>),
    params_(params) {}

  void initialize() {
    ScopeTime("Initialization", params_.debug);

    // Read the point cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (filename_.c_str(), *pc_) == -1) {
      PCL_ERROR("Couldn't read %s file \n", filename_.c_str());
      return;
    }


    // Data augmentation
    Eigen::Vector4f centroid;
    scale_points_unit_sphere (*pc_, params_.gridsize/2, centroid);
    params_.neigh_size = params_.neigh_size * params_.gridsize/2;
    augment_data(pc_, params_);


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

  void computeEdgeFeatures(double* edge_feats);
  void computeAdjacency(double* adj_mat);
  void correctAdjacencyForValidity(double* adj_mat);
  void getValidIndices(int* valid_indices);
  void computeFeatures1d(double* node_feats);
  void computeFeatures3d(double** node_feats);
  void samplePoints();
};
