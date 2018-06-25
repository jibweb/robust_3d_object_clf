#include <time.h>        /* clock_t, clock, CLOCKS_PER_SEC */

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
int compute_graph_feats (std::string filename,
    double** local_feats,
    double* global_feats_p,
    double* global_feats_t,
    int* valid_sal_pt_num,
    Parameters params) {
  // pcl::ScopeTime total("Total time");

  // Point cloud reading
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc(new pcl::PointCloud<pcl::PointXYZINormal>);
  {
    // pcl::ScopeTime t("Point Cloud reading");
    if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (filename.c_str(), *pc) == -1) {
      PCL_ERROR("Couldn't read %s file \n", filename.c_str());
      return (-1);
    }
  }

  {
    ScopeTime t("Downsampling", params.debug);
    int pt_to_remove = params.to_remove*pc->points.size();
    int pre_pc_size = pc->points.size();
    for (uint i=0; i<pt_to_remove; i++) {
      int idx = rand()%static_cast<int>(pc->points.size());
      if (std::isnan(pc->points[idx].x))
        i--;
      else
        pc->points[idx].x = pc->points[idx].y = pc->points[idx].z = std::numeric_limits<float>::quiet_NaN();
    }

    std::vector<int> indices;
    pc->is_dense = false;
    pcl::removeNaNFromPointCloud(*pc, *pc, indices);
    if (params.debug)
        std::cout << "Downsampling: pts_to_remove " << pt_to_remove
                  << " / Pct_to_remove " << params.to_remove
                  << " / Pre pc size " << pre_pc_size
                  << " / Post pc size " << pc->points.size() << std::endl;
  }

  if (params.occl_pct != 0.){
    ScopeTime t("Occluding", params.debug);
    int pre_pc_size = pc->points.size();
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
    tree->setInputCloud (pc);

    int occl_neigh = params.occl_pct * pc->points.size();
    int index = rand()%static_cast<int>(pc->points.size());

    std::vector< int > k_indices(occl_neigh, 0);
    std::vector< float > k_sqr_distances(occl_neigh, 0.);
    tree->nearestKSearch(pc->points[index], occl_neigh, k_indices, k_sqr_distances);

    for (uint i=0; i<k_indices.size(); i++) {
      int idx = k_indices[i];
      pc->points[idx].x = pc->points[idx].y = pc->points[idx].z = std::numeric_limits<float>::quiet_NaN();
    }

    std::vector<int> indices;
    pc->is_dense = false;
    pcl::removeNaNFromPointCloud(*pc, *pc, indices);
    if (params.debug)
        std::cout << "Occlusion: occl_neigh " << occl_neigh
                  << " / Pre pc size " << pre_pc_size
                  << " / Post pc size " << pc->points.size() << std::endl;
  }

  if (params.noise_std >= 0.000001) {
    ScopeTime t("Noise", params.debug);

    boost::mt19937 rng; rng.seed (static_cast<unsigned int> (time (0)));
    pcl::PointXYZINormal minPt, maxPt;
    pcl::getMinMax3D (*pc, minPt, maxPt);
    float max_dist = pcl::euclideanDistance (minPt, maxPt);
    double standard_deviation = params.noise_std * max_dist;
    boost::normal_distribution<> nd (0, standard_deviation);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor (rng, nd);

    for (size_t point_i = 0; point_i < pc->points.size (); ++point_i)
    {
      pc->points[point_i].x = pc->points[point_i].x + static_cast<float> (var_nor ());
      pc->points[point_i].y = pc->points[point_i].y + static_cast<float> (var_nor ());
      pc->points[point_i].z = pc->points[point_i].z + static_cast<float> (var_nor ());
    }
    if (params.debug)
      std::cout << "Noise std: " << standard_deviation << std::endl;
  }

  // Scale to unit sphere
  Eigen::Vector4f xyz_centroid;
  scale_points_unit_sphere (*pc, static_cast<float>(GRIDSIZE_H), xyz_centroid);


  std::vector<int> sampled_indices;
  {
    ScopeTime t("Local features computation", params.debug);
    // local_RFfeatures(pc, local_feats, sampled_indices, params);
  }


  srand (static_cast<unsigned int> (time (0)));


  std::vector<std::vector<std::vector<int> > > lut_;
  {
    ScopeTime t("Voxelization computation", params.debug);
    lut_.resize (GRIDSIZE);
    for (int i = 0; i < GRIDSIZE; ++i) {
        lut_[i].resize (GRIDSIZE);
        for (int j = 0; j < GRIDSIZE; ++j)
          lut_[i][j].resize (GRIDSIZE);
    }
    voxelize9 (*pc, lut_);
  }


  {
    ScopeTime t("Global feature computation", params.debug);
    global_features(pc, global_feats_p, global_feats_t, lut_, params);
  }

  if (params.debug)
    std::cout << "Salient points sampled: " << sampled_indices.size() << std::endl;

  *valid_sal_pt_num = sampled_indices.size();

  return 0;
}
