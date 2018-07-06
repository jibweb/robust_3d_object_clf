// #include <pcl/features/shot_lrf.h>
#include <pcl/features/shot.h>
#include <pcl/features/fpfh.h>

#include "parameters.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void shot_features(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                   double* result,
                   std::vector<int> & sampled_indices,
                   pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
                   Parameters & p) {
  pcl::PointCloud<pcl::SHOT352> shot_pc;
  boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices));

  // Compute the SHOT descriptor
  pcl::SHOTEstimation<pcl::PointXYZINormal, pcl::PointXYZINormal, pcl::SHOT352> shot_estimator;
  shot_estimator.setRadiusSearch (p.neigh_size);
  shot_estimator.setInputCloud (pc->makeShared ());
  shot_estimator.setInputNormals (pc->makeShared ());
  shot_estimator.setSearchMethod (tree);
  shot_estimator.setIndices (sampled_indices_ptr);
  shot_estimator.compute (shot_pc);

  for (uint pt_idx=0; pt_idx<shot_pc.points.size(); pt_idx++) {
    for (uint i=0; i<p.feat_nb; i++)
      if (!std::isnan(shot_pc.points[pt_idx].descriptor[i]))
        result[p.feat_nb*pt_idx + i] = shot_pc.points[pt_idx].descriptor[i];
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void fpfh_features(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                   double* result,
                   std::vector<int> & sampled_indices,
                   pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
                   Parameters & p) {
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
  boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices));

  pcl::FPFHEstimation<pcl::PointXYZINormal, pcl::PointXYZINormal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (pc);
  fpfh.setInputNormals (pc);
  fpfh.setSearchMethod (tree);
  fpfh.setIndices (sampled_indices_ptr);
  fpfh.setRadiusSearch (p.neigh_size / 2.); // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  fpfh.compute (*fpfhs);

  for (uint pt_idx=0; pt_idx<fpfhs->points.size(); pt_idx++) {
    for (uint i=0; i<p.feat_nb; i++)
      if (!std::isnan(fpfhs->points[pt_idx].histogram[i]))
        result[p.feat_nb*pt_idx + i] = fpfhs->points[pt_idx].histogram[i];
  }
}

// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void local_RFfeatures(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
//                     double** result,
//                     std::vector<int> & sampled_indices,
//                     pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
//                     Parameters & p) {
//   int index;
//   std::vector< int > k_indices;
//   std::vector< float > k_sqr_distances;
//   pcl::PointCloud<pcl::PointXYZINormal>::Ptr viz_local_pc(new pcl::PointCloud<pcl::PointXYZINormal>);

//   pcl::PointCloud<pcl::ReferenceFrame> lrf_pc;
//   boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices));

//   // Compute the SHOT LRF
//   pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZINormal, pcl::ReferenceFrame> lrf_estimator;
//   lrf_estimator.setRadiusSearch (8.);
//   lrf_estimator.setInputCloud (pc->makeShared ());
//   lrf_estimator.setSearchMethod (tree);
//   lrf_estimator.setIndices (sampled_indices_ptr);
//   lrf_estimator.compute (lrf_pc);

//   for (uint pt_idx=0; pt_idx < sampled_indices.size(); pt_idx++) {
//     index = sampled_indices[pt_idx];
//     tree->nearestKSearch(pc->points[index], p.neigh_size, k_indices, k_sqr_distances);

//     Eigen::Matrix3f local_rf = lrf_pc.points[pt_idx].getMatrix3fMap();
//     Eigen::Vector3f v1 = pc->points[index].getVector3fMap();

//     if (std::isnan(local_rf(0)) ||
//         std::isnan(local_rf(1)) ||
//         std::isnan(local_rf(2))){
//       std::cout << "Garbage LRF: " << pt_idx << std::endl;
//       continue;
//     }

//     for (uint nn_idx=0; nn_idx < k_indices.size(); nn_idx++) {
//       if (p.viz)
//         viz_local_pc->points.push_back(pc->points[k_indices[nn_idx]]);

//       Eigen::Vector3f v2 = pc->points[k_indices[nn_idx]].getVector3fMap();
//       Eigen::Vector3f new_coords = local_rf*(v2 - v1);
//       result[pt_idx][p.feat_nb*nn_idx + 0] = new_coords(0);
//       result[pt_idx][p.feat_nb*nn_idx + 1] = new_coords(1);
//       result[pt_idx][p.feat_nb*nn_idx + 2] = new_coords(2);
//     }
//   }

//   if (p.viz)
//     pcl::io::savePCDFile ("local_test.pcd", *viz_local_pc, true);
// }
