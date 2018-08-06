#include <random>

#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>

#define PI 3.14159265
#include "parameters.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void scale_points_unit_sphere (pcl::PointCloud<pcl::PointXYZINormal> &pc,
                               float scalefactor,
                               Eigen::Vector4f& centroid) {
  pcl::compute3DCentroid (pc, centroid);
  pcl::demeanPointCloud (pc, centroid, pc);

  float max_distance = 0, d;
  pcl::PointXYZINormal cog;
  cog.x = 0;
  cog.y = 0;
  cog.z = 0;

  for (size_t i = 0; i < pc.points.size (); ++i)
  {
    d = pcl::euclideanDistance(cog,pc.points[i]);
    if (d > max_distance)
      max_distance = d;
  }

  float scale_factor = 1.0f / max_distance * scalefactor;

  Eigen::Affine3f matrix = Eigen::Affine3f::Identity();
  matrix.scale (scale_factor);
  pcl::transformPointCloud (pc, pc, matrix);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void augment_data(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                  Parameters & params) {

  if (pc->points.size() > params.to_keep){
    uint pt_to_remove = pc->points.size() - params.to_keep;
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
        std::cout << "Downsampling (#): pts_to_remove " << pt_to_remove
                  << " / Nb_to_keep " << params.to_keep
                  << " / Post pc size " << pc->points.size() << std::endl;
  }


  if (params.to_remove > 0.00000001) {
    // ScopeTime t("Downsampling", params.debug);
    uint pt_to_remove = params.to_remove*pc->points.size();
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
        std::cout << "Downsampling (%): pts_to_remove " << pt_to_remove
                  << " / Pct_to_remove " << params.to_remove
                  << " / Pre pc size " << pre_pc_size
                  << " / Post pc size " << pc->points.size() << std::endl;
  }

  if (params.occl_pct > 0.00000001){
    // ScopeTime t("Occluding", params.debug);
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

  if (params.noise_std >= 0.00000001) {
    // ScopeTime t("Noise", params.debug);

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

    // Scale to unit sphere
    Eigen::Vector4f xyz_centroid;
    scale_points_unit_sphere (*pc, static_cast<float>(params.gridsize/2), xyz_centroid);
  }

  if (params.rotation_deg > 0) {
    int roll = rand() % params.rotation_deg;
    int pitch = rand() % params.rotation_deg;
    int yaw = rand() % params.rotation_deg;

    if (params.debug)
      std::cout << "Rotating the point cloud with angles roll: " << roll
                << " / pitch: " << pitch
                << " / yaw: " << yaw
                << std::endl;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 0.0, 0.0, 0.0;
    transform.rotate(Eigen::AngleAxisf((roll*M_PI) / 180, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf((pitch*M_PI) / 180, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf((yaw*M_PI) / 180, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*pc, *pc, transform); //Only rotate target cloud
  }
}
