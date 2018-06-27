#include <random>
#include <time.h>        /* clock_t, clock, CLOCKS_PER_SEC */
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "augmentation_preprocessing.cpp"
#include "graph_structure.cpp"
#include "features_computation.cpp"
#include "parameters.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ScopeTime
{
protected:
  boost::posix_time::ptime start_time_;

public:
  inline ScopeTime (std::string title, bool debug) :
    title_ (title), debug_(debug)
  {
    start_time_ = boost::posix_time::microsec_clock::local_time ();
  }

  inline ScopeTime () :
    title_ (std::string (""))
  {
    start_time_ = boost::posix_time::microsec_clock::local_time ();
  }

  inline double
  getTime ()
  {
    boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time ();
    return (static_cast<double> (((end_time - start_time_).total_milliseconds ())));
  }

  inline ~ScopeTime ()
  {
    double val = this->getTime ();
    if (debug_)
      std::cerr << title_ << " took " << val << "ms.\n";
  }

private:
  std::string title_;
  bool debug_;
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int compute_graph_feats(std::string filename,
                        double* node_feats,
                        double* adj_mat,
                        Parameters params){
  ScopeTime total("Total time", params.debug);

  // Point cloud reading
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc(new pcl::PointCloud<pcl::PointXYZINormal>);
  {
    ScopeTime t("Point Cloud reading", params.debug);
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (filename.c_str(), *pc) == -1) {
      PCL_ERROR("Couldn't read %s file \n", filename.c_str());
      return (-1);
    }
  }

  // Data corruption (occlusion, noise, downsampling, ...)
  {
    ScopeTime t("Data preprocessing/augmentation", params.debug);

    Eigen::Vector4f centroid;
    scale_points_unit_sphere (*pc, params.gridsize/2, centroid);
    params.neigh_size = params.neigh_size * params.gridsize/2;
    augment_data(pc, params);
  }


  // Sample graph nodes
  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
  tree->setInputCloud (pc);
  std::vector<int> sampled_indices;
  {
    ScopeTime t("Node sampling", params.debug);
    srand (static_cast<unsigned int> (time (0)));
    sample_local_points(pc, sampled_indices, *tree, params);
  }


  {
    ScopeTime t("Local features computation", params.debug);
    shot_features(pc, node_feats, sampled_indices, tree, params);
  }


  std::vector<std::vector<std::vector<int> > > lut_;
  {
    ScopeTime t("Voxelization computation", params.debug);
    lut_.resize (params.gridsize);
    for (uint i = 0; i < params.gridsize; ++i) {
        lut_[i].resize (params.gridsize);
        for (uint j = 0; j < params.gridsize; ++j)
          lut_[i][j].resize (params.gridsize);
    }
    voxelize (*pc, lut_, params.gridsize);
  }


  {
    ScopeTime t("Graph structure computation", params.debug);
    occupancy_graph_structure(pc, adj_mat, sampled_indices, lut_, params);
  }

  if (params.debug)
    std::cout << "Salient points sampled: " << sampled_indices.size() << std::endl;

  if (params.viz) {

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);

    for (uint i=0; i<sampled_indices.size(); i++) {
      int idx = sampled_indices[i];
      viewer->addSphere<pcl::PointXYZINormal>(pc->points[idx], 0.1, 1., 0., 0., "line_" +std::to_string(idx));
      for (uint i2=0; i2<params.nodes_nb; i2++) {
        if (adj_mat[params.nodes_nb*i + i2] > 0.) {
          int idx2 = sampled_indices[i2];
          viewer->addLine<pcl::PointXYZINormal>(pc->points[idx], pc->points[idx2], 0., 0., 1., "line_" +std::to_string(idx)+std::to_string(idx2));
        }
      }
    }

    // params.to_remove = 0.9;
    // augment_data(pc, params);
    // viewer->addPointCloud<pcl::PointXYZINormal> (pc, "cloud");
    while (!viewer->wasStopped()) {
      viewer->spinOnce(100);
    }
  }

  return 0;
}
