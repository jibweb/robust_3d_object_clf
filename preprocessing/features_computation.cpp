// #include <pcl/features/shot_lrf.h>
#include <math.h>
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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void points_coords_features(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                   double* result,
                   std::vector<int> & sampled_indices,
                   pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
                   Parameters & p) {
  for (uint pt_idx=0; pt_idx<sampled_indices.size(); pt_idx++) {
    result[p.feat_nb*pt_idx + 0] = pc->points[sampled_indices[pt_idx]].x *2/p.gridsize;
    result[p.feat_nb*pt_idx + 1] = pc->points[sampled_indices[pt_idx]].y *2/p.gridsize;
    result[p.feat_nb*pt_idx + 2] = pc->points[sampled_indices[pt_idx]].z *2/p.gridsize;
  }
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void esf3d_features(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                    double** result,
                    std::vector<int> & sampled_indices,
                    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
                    Parameters & p) {
  uint p1_idx, p2_idx, rdn_weight;
  float pair_nb=0.;
  uint max_pair_nb;
  const int sample_pair_nb = 500;

  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;

  Eigen::Vector4f v1, v2, n1, n2, v12;
  uint d_idx, na_idx, va_idx;
  float max_dist = 2*p.neigh_size;

  for (uint pt_idx=0; pt_idx<sampled_indices.size(); pt_idx++) {
    // std::cout << "Point " << pt_idx << " / index: " << sampled_indices[pt_idx] <<std::endl;
    tree->radiusSearch(pc->points[sampled_indices[pt_idx]], p.neigh_size, k_indices, k_sqr_distances);

    // std::cout << "neighborhood: " << k_indices.size() << std::endl;
    pair_nb = 0.;
    max_pair_nb = k_indices.size() * (k_indices.size() - 1) / 2;

    if (p.debug && max_pair_nb < sample_pair_nb)
      std::cout << "Max pair nb: " << max_pair_nb << std::endl;

    for (uint index1=0; index1<k_indices.size(); index1++) {
      for (uint index2=index1+1; index2<k_indices.size(); index2++) {
        rdn_weight = rand() % max_pair_nb;
        if (rdn_weight > sample_pair_nb)
          continue;

        // std::cout << index1 << " " << index2;
        p1_idx = k_indices[index1];
        p2_idx = k_indices[index2];

        if (std::isnan(pc->points[p1_idx].normal_x) || std::isnan(pc->points[p2_idx].normal_x))
          continue;

        // Get the vectors
        v1 = pc->points[p1_idx].getVector4fMap ();
        v2 = pc->points[p2_idx].getVector4fMap ();
        n1 = pc->points[p1_idx].getNormalVector4fMap ();
        n2 = pc->points[p2_idx].getNormalVector4fMap ();

        v12 = v1 - v2;

        // Get the indices
        d_idx = static_cast<uint>(std::min(std::max(ceil(4*(v12.norm() / max_dist)) - 1, 0.), 3.));
        na_idx = static_cast<uint>(std::min(std::max(ceil(2*(n1.dot(n2) + 1)) - 1, 0.), 3.));
        v12.normalize();
        va_idx = static_cast<uint>(std::min(std::max(ceil(4*std::max(fabs(v12.dot(n1)), fabs(v12.dot(n2)))) - 1, 0.), 3.));



        if (na_idx > 3 || d_idx > 3 || va_idx > 3) {
          std::cout << d_idx << " " << na_idx << " " << va_idx << std::endl;
          std::cout << " " << n1 <<  "\n --- \n " << n2 << std::endl;
          std::cout << "._._._.\n" << v12 << "\n._._._." << std::endl;
          std::cout << 4*4*d_idx + 4*na_idx + va_idx << std::endl;
        }

        result[pt_idx][4*4*d_idx + 4*na_idx + va_idx] += 1.;
        pair_nb += 1;

        if (pair_nb == sample_pair_nb)
          goto norm_hist;
      }
    }

    norm_hist:
    // std::cout << "pair: " << pair_nb << " / " << max_pair_nb << std::endl;

    // Normalize
    for (uint i=0; i<64; i++) {
      result[pt_idx][i] /= pair_nb + 1e-6;
      // result[pt_idx][i] -= 0.5;
    }

    // std::cout << "Normalized happily" << std::endl;
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
