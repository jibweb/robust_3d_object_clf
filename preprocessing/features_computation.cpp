#include <pcl/features/shot_lrf.h>

#include "parameters.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void local_RFfeatures(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                    double** result,
                    std::vector<int> & sampled_indices,
                    Parameters & p) {
  if (p.debug) {
    if (p.sal_pt_num*2*p.neigh_size > pc->points.size()) {
      std::cout << "Trying to sample too many points OR the neighborhood is too large. "
                   "Reduce either of those !!!" << std::endl;
    }
  }

  int index;
  std::vector< int > k_indices(p.neigh_size, 0);
  std::vector< float > k_sqr_distances(p.neigh_size, 0.);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr viz_local_pc(new pcl::PointCloud<pcl::PointXYZINormal>);

  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
  tree->setInputCloud (pc);

  sample_local_points(pc, sampled_indices, *tree, p);

  pcl::PointCloud<pcl::ReferenceFrame> lrf_pc;
  boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices));

  pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZINormal, pcl::ReferenceFrame> lrf_estimator;
  lrf_estimator.setRadiusSearch (8.);
  lrf_estimator.setInputCloud (pc->makeShared ());
  lrf_estimator.setSearchMethod (tree);
  lrf_estimator.setIndices (sampled_indices_ptr);
  lrf_estimator.compute (lrf_pc);

  for (uint sal_pt_idx=0; sal_pt_idx < sampled_indices.size(); sal_pt_idx++) {
    index = sampled_indices[sal_pt_idx];
    tree->nearestKSearch(pc->points[index], p.neigh_size, k_indices, k_sqr_distances);

    Eigen::Matrix3f local_rf = lrf_pc.points[sal_pt_idx].getMatrix3fMap();
    Eigen::Vector3f v1 = pc->points[index].getVector3fMap();

    if (std::isnan(local_rf(0)) ||
        std::isnan(local_rf(1)) ||
        std::isnan(local_rf(2))){
      std::cout << "Garbage LRF: " << sal_pt_idx << std::endl;
      continue;
    }

    for (uint nn_idx=0; nn_idx < k_indices.size(); nn_idx++) {
      if (p.viz)
        viz_local_pc->points.push_back(pc->points[k_indices[nn_idx]]);

      Eigen::Vector3f v2 = pc->points[k_indices[nn_idx]].getVector3fMap();
      Eigen::Vector3f new_coords = local_rf*(v2 - v1);
      result[sal_pt_idx][p.local_feat_num*nn_idx + 0] = new_coords(0);
      result[sal_pt_idx][p.local_feat_num*nn_idx + 1] = new_coords(1);
      result[sal_pt_idx][p.local_feat_num*nn_idx + 2] = new_coords(2);
    }
  }

  if (p.viz)
    pcl::io::savePCDFile ("local_test.pcd", *viz_local_pc, true);
}
