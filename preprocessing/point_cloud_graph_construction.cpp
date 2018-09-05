#include <random>
#include <math.h>
// #include <time>

#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


#include "point_cloud_graph_construction.h"



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INTERFACES /////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::computeFeatures1d(double* node_feats) {
  ScopeTime t("Nodes features computation", params_.debug);
  if (params_.feat_nb == 352)
    shotNodeFeatures(node_feats);
  else if (params_.feat_nb == 33)
    fpfhNodeFeatures(node_feats);
  else if (params_.feat_nb == 3)
    pointCoordsNodeFeatures(node_feats);
  else if (params_.feat_nb == 1)
    dummyNodeFeatures(node_feats);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::computeFeatures3d(double** node_feats) {
  ScopeTime t("Nodes features computation", params_.debug);
  if (params_.feat_nb == 4)
    esf3dNodeFeatures(node_feats);
  else if (params_.feat_nb >= 500)
    lEsfNodeFeatures(node_feats);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::samplePoints() {
  ScopeTime t("Point sampling computation", params_.debug);
  // Prepare the values for the sampling procedure
  srand (static_cast<unsigned int> (time (0)));
  int rdn_weight, index;
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;
  int total_weight = pc_->points.size();
  std::vector<bool> probs(pc_->points.size(), true);

  ////////////////////////////
  if (params_.min_angle_z_normal > 0.) {
    Eigen::Vector3f z(0., 0., 1.);
    z.normalize();
    for (uint i=0; i<pc_->points.size(); i++) {
      Eigen::Vector3f n1 = pc_->points[i].getNormalVector3fMap();
      n1.normalize();

      if (acos(fabs(n1.dot(z))) < params_.min_angle_z_normal*M_PI/180.) {
        probs[i] = false;
        total_weight -= 1;
      }
    }
  }
  ////////////////////////////

  for (uint pt_idx=0; pt_idx < params_.nodes_nb; pt_idx++) {
    // Sample a new point
    if (total_weight > 0) {
      rdn_weight = rand() % total_weight;
      index = -1;
    } else {
      break;
    }

    for (uint i=0; i<pc_->points.size(); i++){
      if (!probs[i])
        continue;

      if (rdn_weight == 0) {
        index = i;
        break;
      }

      rdn_weight -= 1;
    }

    if (index == -1) {
      // There is no point left to sample !
      if (params_.debug)
        std::cout << "Couldn't sample " << params_.nodes_nb - pt_idx << " salient points" << std::endl;
      break;
    }

    // Check if the sampled point is usable
    if (std::isnan(pc_->points[index].normal_x)) {
      probs[index] = false;
      total_weight -= 1;
      pt_idx--;
      continue;
    }

    if (params_.neigh_size > 0.) {
      // Extract the sampled point neighborhood
      tree_->radiusSearch(pc_->points[index], params_.neigh_size, k_indices, k_sqr_distances);

      // Update the sampling probability
      for (uint i=0; i < k_indices.size(); i++) {
        if (probs[k_indices[i]])
          total_weight -= 1;

        probs[k_indices[i]] = false;
      }
    } else {
      probs[index] = false;
      total_weight -= 1;
    }

    sampled_indices_.push_back(index);
  }

  // Update the valid indices vector
  for (uint i=0; i < sampled_indices_.size(); i++)
    valid_indices_[i] = true;

  if (params_.debug)
    std::cout << "Sampled points: " << sampled_indices_.size() << std::endl;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::computeEdgeFeatures(double* edge_feats) {
  ScopeTime t("Edge features computation", params_.debug);

  if (!params_.edge_feats)
    return;

  if (params_.edge_feat_nb == 5)
    pointPairEdgeFeatures(edge_feats);
  else if (params_.edge_feat_nb == 6)
    lrfEdgeFeatures(edge_feats);
  else if (params_.edge_feat_nb == 3) {
    if (params_.min_angle_z_normal > 0.) {
      rotZEdgeFeatures(edge_feats);
    } else {
      coordsEdgeFeatures(edge_feats);
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::computeAdjacency(double* adj_mat) {
  ScopeTime t("Adjacency matrix computation", params_.debug);
  if (params_.neigh_nb <= 0) {
    for (uint index1=0; index1 < params_.nodes_nb; index1++) {
      for (uint index2=0; index2 < params_.nodes_nb; index2++) {
        adj_mat[index1*params_.nodes_nb + index2] = 1.;
      }
    }
  } else
    occupancyAdjacency(adj_mat);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::getValidIndices(int* valid_indices) {
  for (uint i=0; i < params_.nodes_nb; i++) {
    if (valid_indices_[i])
      valid_indices[i] = 1;
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADJACENCY ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::occupancyAdjacency(double* adj_mat) {
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (uint pt_idx=0; pt_idx < sampled_indices_.size(); pt_idx++) {
    local_cloud->points.push_back(pc_->points[sampled_indices_[pt_idx]]);
  }

  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr local_tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
  local_tree->setInputCloud(local_cloud);
  std::vector<int> k_indices;
  std::vector<float> k_sqr_distances;
  float occ_ratio;

  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
    pcl::PointXYZINormal pt = local_cloud->points[pt_idx];
    int k_elts = params_.neigh_nb < local_cloud->points.size() ? params_.neigh_nb : local_cloud->points.size();
    local_tree->nearestKSearch(pt, k_elts, k_indices, k_sqr_distances);

    Eigen::Vector4f v1 = local_cloud->points[pt_idx].getVector4fMap ();
    adj_mat[params_.nodes_nb*pt_idx + pt_idx] = 1.;
    for (uint i=0; i < k_elts; i++) {
      if (k_sqr_distances[i] == 0.)
        continue;

      Eigen::Vector4f v2 = local_cloud->points[k_indices[i]].getVector4fMap ();
      occ_ratio = occupancy_ratio(v1, v2, lut_, params_.gridsize/2);
      if (occ_ratio > 0.0) {
        adj_mat[params_.nodes_nb*pt_idx + k_indices[i]] = 1.;
        adj_mat[params_.nodes_nb*k_indices[i] + pt_idx] = 1.;
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::correctAdjacencyForValidity(double* adj_mat) {
  for (uint index1=0; index1 < params_.nodes_nb; index1++) {
    for (uint index2=0; index2 < params_.nodes_nb; index2++) {
      if (!valid_indices_[index1] || !valid_indices_[index2])
        adj_mat[index1*params_.nodes_nb + index2] = 0.;
    }
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// EDGE FEATURES ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::pointPairEdgeFeatures(double* edge_feats) {
  int index1, index2;
  float d, occ_r, a12, a12n1, a12n2;
  float feat_min = -0.5;
  float feat_max =  0.5;

  for (uint pt1=0; pt1<sampled_indices_.size(); pt1++) {
    for (uint pt2=pt1; pt2<sampled_indices_.size(); pt2++) {
      if (pt1 == pt2) {
        edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 0] = -0.5;
        edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 1] =  0.5;
        edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 2] =  0.5;
        edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 3] = -0.5;
        edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 4] = -0.5;
        continue;
      }

      index1 = sampled_indices_[pt1];
      index2 = sampled_indices_[pt2];

      // Vector setup
      Eigen::Vector4f v1 = pc_->points[index1].getVector4fMap ();
      Eigen::Vector4f v2 = pc_->points[index2].getVector4fMap ();
      Eigen::Vector4f n1 = pc_->points[index1].getNormalVector4fMap ();
      Eigen::Vector4f n2 = pc_->points[index2].getNormalVector4fMap ();
      Eigen::Vector4f v12 = v1 - v2;

      // Feature computation
      d = v12.norm() / (params_.gridsize) - 0.5;
      v12.normalize();
      occ_r = occupancy_ratio(v1, v2, lut_, params_.gridsize/2) - 0.5;
      a12 = n1.dot(n2)/2;
      a12n1 = fabs(v12.dot(n1)) - 0.5;
      a12n2 = fabs(v12.dot(n2)) - 0.5;

      // Saturate the features
      d = std::min(std::max(d, feat_min), feat_max);
      a12 = std::min(std::max(a12, feat_min), feat_max);
      a12n1 = std::min(std::max(a12n1, feat_min), feat_max);
      a12n2 = std::min(std::max(a12n2, feat_min), feat_max);

      // Fill in the matrix
      edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 0] = d;
      edge_feats[5*(params_.nodes_nb*pt2 + pt1) + 0] = d;
      edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 1] = occ_r;
      edge_feats[5*(params_.nodes_nb*pt2 + pt1) + 1] = occ_r;
      edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 2] = a12;
      edge_feats[5*(params_.nodes_nb*pt2 + pt1) + 2] = a12;
      edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 3] = a12n1;
      edge_feats[5*(params_.nodes_nb*pt2 + pt1) + 3] = a12n1;
      edge_feats[5*(params_.nodes_nb*pt1 + pt2) + 4] = a12n2;
      edge_feats[5*(params_.nodes_nb*pt2 + pt1) + 4] = a12n2;
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::lrfEdgeFeatures(double* edge_feats) {
  int index1, index2;
  float d, a12, occ_r;
  pcl::PointCloud<pcl::ReferenceFrame> lrf_pc;
  boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices_));

  pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZINormal, pcl::ReferenceFrame> lrf_estimator;
  lrf_estimator.setRadiusSearch (0.35*params_.gridsize);
  lrf_estimator.setInputCloud (pc_);
  lrf_estimator.setSearchMethod (tree_);
  lrf_estimator.setIndices (sampled_indices_ptr);
  lrf_estimator.compute (lrf_pc);

  for (uint pt1_idx=0; pt1_idx < sampled_indices_.size(); pt1_idx++) {
    index1 = sampled_indices_[pt1_idx];
    Eigen::Matrix3f local_rf = lrf_pc.points[pt1_idx].getMatrix3fMap();
    local_rf.normalize();
    Eigen::Vector3f v1 = pc_->points[index1].getVector3fMap();
    Eigen::Vector4f n1 = pc_->points[index1].getNormalVector4fMap ();

    if (std::isnan(local_rf(0)) ||
        std::isnan(local_rf(1)) ||
        std::isnan(local_rf(2))){
      if (params_.debug)
        std::cout << "Garbage LRF: " << pt1_idx << std::endl;
      valid_indices_[pt1_idx] = false;
      continue;
    }

    for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
      if (pt1_idx == pt2_idx) {
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 0] = -0.5;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 1] =  0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 2] =  0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 3] =  0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 4] =  0.5;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 5] =  0.5;
        continue;
      }
      index2 = sampled_indices_[pt2_idx];
      Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
      Eigen::Vector4f n2 = pc_->points[index2].getNormalVector4fMap ();
      Eigen::Vector3f v21 = v2 - v1;
      Eigen::Vector3f new_coords = local_rf * v21;
      new_coords.normalize();

      d = v21.norm() / (params_.gridsize) - 0.5;
      v21.normalize();
      occ_r = occupancy_ratio(v1, v2, lut_, params_.gridsize/2) - 0.5;
      a12 = n1.dot(n2)/2;

      edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 0] = d;
      edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 1] = new_coords(0); // / params_.gridsize;
      edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 2] = new_coords(1); // / params_.gridsize;
      edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 3] = new_coords(2); // / params_.gridsize;
      edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 4] = a12;
      edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 5] = occ_r;

      if (std::isnan(new_coords(0)) || std::isnan(new_coords(1)) || std::isnan(new_coords(2)))
        std::cout << new_coords << "\n----\n"
                  << local_rf << "\n---\n"
                  << v21 << "\n---\n"
                  << v1 << "\n---\n"
                  << v2 << "\n***\n"
                  << std::endl;

      if (std::isnan(new_coords(0)) || std::isnan(new_coords(1)) || std::isnan(new_coords(2)) ||
          std::isnan(d) || std::isnan(a12) || std::isnan(occ_r)) {
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 0] = 0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 1] = 0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 2] = 0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 3] = 0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 4] = 0.;
        edge_feats[6*(params_.nodes_nb*pt1_idx + pt2_idx) + 5] = 0.;
      }


    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::coordsEdgeFeatures(double* edge_feats) {
  int index1, index2;

  for (uint pt1_idx=0; pt1_idx < sampled_indices_.size(); pt1_idx++) {
    index1 = sampled_indices_[pt1_idx];
    Eigen::Vector3f v1 = pc_->points[index1].getVector3fMap();

    for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
      index2 = sampled_indices_[pt2_idx];
      Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
      Eigen::Vector3f v21 = v2 - v1;
      // v21.normalize();

      edge_feats[3*(params_.nodes_nb*pt1_idx + pt2_idx) + 0] = v21(0);
      edge_feats[3*(params_.nodes_nb*pt1_idx + pt2_idx) + 1] = v21(1);
      edge_feats[3*(params_.nodes_nb*pt1_idx + pt2_idx) + 2] = v21(2);
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::rotZEdgeFeatures(double* edge_feats) {
  int index1, index2;
  Eigen::Vector3f z(0., 0., 1.);
  z.normalize();

  for (uint pt1_idx=0; pt1_idx < sampled_indices_.size(); pt1_idx++) {
    index1 = sampled_indices_[pt1_idx];
    Eigen::Vector3f v1 = pc_->points[index1].getVector3fMap();
    Eigen::Vector3f n1 = pc_->points[index1].getNormalVector3fMap();
    n1.normalize();

    if (acos(fabs(n1.dot(z))) < params_.min_angle_z_normal*M_PI/180.) {
      valid_indices_[pt1_idx] = false;
      continue;
    }


    Eigen::Vector3f n_axis;
    n_axis(0) = n1(0);
    n_axis(1) = n1(1);
    n_axis(2) = 0.;
    n_axis.normalize();
    Eigen::Vector3f w = n_axis.cross(z);

    Eigen::Matrix3f lrf;
    lrf(0,0) = n_axis(0);
    lrf(0,1) = n_axis(1);
    lrf(0,2) = n_axis(2);

    lrf(1,0) = w(0);
    lrf(1,1) = w(1);
    lrf(1,2) = w(2);

    lrf(2,0) = z(0);
    lrf(2,1) = z(1);
    lrf(2,2) = z(2);

    // lrf.row(0) << n1;
    // lrf.row(1) << w;
    // lrf.row(2) << z;

    for (uint pt2_idx=0; pt2_idx < sampled_indices_.size(); pt2_idx++) {
      index2 = sampled_indices_[pt2_idx];
      Eigen::Vector3f v2 = pc_->points[index2].getVector3fMap();
      Eigen::Vector3f v21 = v2 - v1;
      Eigen::Vector3f local_coords = lrf * v21;
      // v21.normalize();

      if (std::isnan(local_coords(0)) || std::isnan(local_coords(1)) || std::isnan(local_coords(2))) {
        std::cout << local_coords << "\n----\n"
                  << lrf << "\n---\n"
                  << n1 << "\n---\n"
                  << n_axis << "\n---\n"
                  << w << "\n---\n"
                  << z << "\n***\n"
                  << std::endl;
      }

      edge_feats[3*(params_.nodes_nb*pt1_idx + pt2_idx) + 0] = local_coords(0);
      edge_feats[3*(params_.nodes_nb*pt1_idx + pt2_idx) + 1] = local_coords(1);
      edge_feats[3*(params_.nodes_nb*pt1_idx + pt2_idx) + 2] = local_coords(2);
    }
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// NODE FEATURES ///////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::dummyNodeFeatures(double* result) {
  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++)
    result[params_.feat_nb*pt_idx + 0] = 1.;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::shotNodeFeatures(double* result) {
  pcl::PointCloud<pcl::SHOT352> shot_pc;
  boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices_));

  // Compute the SHOT descriptor
  pcl::SHOTEstimation<pcl::PointXYZINormal, pcl::PointXYZINormal, pcl::SHOT352> shot_estimator;
  shot_estimator.setRadiusSearch (params_.neigh_size);
  shot_estimator.setInputCloud (pc_);
  shot_estimator.setInputNormals (pc_);
  shot_estimator.setSearchMethod (tree_);
  shot_estimator.setIndices (sampled_indices_ptr);
  shot_estimator.compute (shot_pc);

  for (uint pt_idx=0; pt_idx<shot_pc.points.size(); pt_idx++) {
    for (uint i=0; i<params_.feat_nb; i++) {
      if (!std::isnan(shot_pc.points[pt_idx].descriptor[i]))
        result[params_.feat_nb*pt_idx + i] = shot_pc.points[pt_idx].descriptor[i];
      else {
        valid_indices_[pt_idx] = false;
        break;
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::fpfhNodeFeatures(double* result) {
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
  boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices_));

  pcl::FPFHEstimation<pcl::PointXYZINormal, pcl::PointXYZINormal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud (pc_);
  fpfh.setInputNormals (pc_);
  fpfh.setSearchMethod (tree_);
  fpfh.setIndices (sampled_indices_ptr);
  fpfh.setRadiusSearch (params_.neigh_size / 2.); // IMPORTANT: the radius used here has to be larger than the
  // radius used to estimate the surface normals!!!
  fpfh.compute (*fpfhs);

  for (uint pt_idx=0; pt_idx<fpfhs->points.size(); pt_idx++) {
    for (uint i=0; i<params_.feat_nb; i++) {
      if (!std::isnan(fpfhs->points[pt_idx].histogram[i]))
        result[params_.feat_nb*pt_idx + i] = fpfhs->points[pt_idx].histogram[i];
      else {
        valid_indices_[pt_idx] = false;
        break;
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::pointCoordsNodeFeatures(double* result) {
  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
    result[params_.feat_nb*pt_idx + 0] = pc_->points[sampled_indices_[pt_idx]].x *2/params_.gridsize;
    result[params_.feat_nb*pt_idx + 1] = pc_->points[sampled_indices_[pt_idx]].y *2/params_.gridsize;
    result[params_.feat_nb*pt_idx + 2] = pc_->points[sampled_indices_[pt_idx]].z *2/params_.gridsize;
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::esf3dNodeFeatures(double** result) {
  uint p1_idx, p2_idx, rdn_weight;
  float pair_nb=0.;
  uint max_pair_nb;
  const int sample_pair_nb = 1000;
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;
  Eigen::Vector4f v1, v2, n1, n2, v12;
  double dist, na, va;
  uint d_idx, na_idx, va_idx;
  float max_dist = 0.;
  if (params_.mesh) {
    for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
      v1 = pc_->points[nodes_elts_[pt_idx][0]].getVector4fMap ();
      for (uint i=nodes_elts_[pt_idx].size()/2; i < nodes_elts_[pt_idx].size(); i++) {
        v2 = pc_->points[nodes_elts_[pt_idx][i]].getVector4fMap ();
        v12 = v1-v2;
        float dist = 2.*v12.norm();
        if (dist > max_dist)
          max_dist = dist;
      }
    }
  } else
    max_dist = 2*params_.neigh_size;

  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
    if (params_.mesh)
      k_indices = nodes_elts_[pt_idx];
    else
      tree_->radiusSearch(pc_->points[sampled_indices_[pt_idx]], params_.neigh_size, k_indices, k_sqr_distances);

    pair_nb = 0.;
    max_pair_nb = k_indices.size() * (k_indices.size() - 1) / 2;

    if (max_pair_nb < sample_pair_nb / 2) {
      if (params_.debug)
        std::cout << "Node " << pt_idx << " max pair nb: " << max_pair_nb << std::endl;

      valid_indices_[pt_idx] = false;
    }

    for (uint index1=0; index1 < k_indices.size(); index1++) {
      for (uint index2=index1+1; index2 < k_indices.size(); index2++) {
        if (max_pair_nb < 10*sample_pair_nb) {
          rdn_weight = rand() % max_pair_nb;
          if (rdn_weight > sample_pair_nb)
            continue;

          p1_idx = k_indices[index1];
          p2_idx = k_indices[index2];
        } else {
          rdn_weight = rand() % k_indices.size();
          p1_idx = k_indices[rdn_weight];
          rdn_weight = rand() % k_indices.size();
          p2_idx = k_indices[rdn_weight];
        }

        if (std::isnan(pc_->points[p1_idx].normal_x) || std::isnan(pc_->points[p2_idx].normal_x))
          continue;

        // Get the vectors
        v1 = pc_->points[p1_idx].getVector4fMap ();
        v2 = pc_->points[p2_idx].getVector4fMap ();
        n1 = pc_->points[p1_idx].getNormalVector4fMap ();
        n2 = pc_->points[p2_idx].getNormalVector4fMap ();

        v12 = v1 - v2;

        // Get the indices
        dist = ceil(4*(v12.norm() / max_dist)) - 1;
        d_idx = static_cast<uint>(std::min(std::max(dist, 0.), 3.));
        na = ceil(2*(n1.dot(n2) + 1)) - 1;
        na_idx = static_cast<uint>(std::min(std::max(na, 0.), 3.));
        v12.normalize();
        va = ceil(4*std::max(fabs(v12.dot(n1)), fabs(v12.dot(n2)))) - 1;
        va_idx = static_cast<uint>(std::min(std::max(va, 0.), 3.));

        if (na_idx > 3 || d_idx > 3 || va_idx > 3) {
          std::cout << d_idx << " " << na_idx << " " << va_idx << std::endl;
          std::cout << " " << n1 <<  "\n --- \n " << n2 << std::endl;
          std::cout << "._._._.\n" << v12 << "\n._._._." << std::endl;
          std::cout << 4*4*d_idx + 4*na_idx + va_idx << std::endl;
        }

        result[pt_idx][4*4*d_idx + 4*na_idx + va_idx] += 1.;
        pair_nb += 1;

        if (pair_nb >= sample_pair_nb) {
          // Break out of the two loops
          index1 = k_indices.size();
          index2 = k_indices.size();
        }
      }
    }

    // Normalize
    for (uint i=0; i<64; i++) {
      result[pt_idx][i] /= pair_nb + 1e-6;
      // result[pt_idx][i] -= 0.5;
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::lEsfNodeFeatures(double** result) {
  uint p1_idx, p2_idx, rdn_weight;
  uint pair_nb=0;
  uint max_pair_nb;
  const uint sample_pair_nb = params_.feat_nb;
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;
  Eigen::Vector4f v1, v2, n1, n2, v12;
  double dist, a12, a12n1, a12n2, occ_r, z;
  double feat_min = -0.5;
  double feat_max =  0.5;
  double max_dist = 0.;

  // if (params_.mesh) {
  //   for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
  //     v1 = pc_->points[nodes_elts_[pt_idx][0]].getVector4fMap ();
  //     for (uint i=nodes_elts_[pt_idx].size()/2; i < nodes_elts_[pt_idx].size(); i++) {
  //       v2 = pc_->points[nodes_elts_[pt_idx][i]].getVector4fMap ();
  //       v12 = v1-v2;
  //       double dist = 2.*v12.norm();
  //       if (dist > max_dist)
  //         max_dist = dist;
  //     }
  //   }
  // } else
  //   max_dist = 2*params_.neigh_size;
  max_dist = params_.gridsize;

  for (uint pt_idx=0; pt_idx<sampled_indices_.size(); pt_idx++) {
    if (params_.mesh)
      k_indices = nodes_elts_[pt_idx];
    else
      tree_->radiusSearch(pc_->points[sampled_indices_[pt_idx]], params_.neigh_size, k_indices, k_sqr_distances);

    pair_nb = 0;
    max_pair_nb = k_indices.size() * (k_indices.size() - 1) / 2;

    if (max_pair_nb < sample_pair_nb / 2) {
      if (params_.debug)
        std::cout << "Node " << pt_idx << " max pair nb: " << max_pair_nb << std::endl;

      valid_indices_[pt_idx] = false;
    }

    for (uint index1=0; index1 < k_indices.size(); index1++) {
      for (uint index2=index1+1; index2 < k_indices.size(); index2++) {
        if (max_pair_nb < 10*sample_pair_nb) {
          rdn_weight = rand() % max_pair_nb;
          if (rdn_weight > sample_pair_nb)
            continue;

          p1_idx = k_indices[index1];
          p2_idx = k_indices[index2];
        } else {
          int max_sampling = std::min(static_cast<int>(k_indices.size()), 500);
          rdn_weight = rand() % max_sampling;
          p1_idx = k_indices[rdn_weight];
          // rdn_weight = rand() % k_indices.size();
          // p2_idx = k_indices[rdn_weight];
          p2_idx = rand() % pc_->points.size();
        }

        if (std::isnan(pc_->points[p1_idx].normal_x) || std::isnan(pc_->points[p2_idx].normal_x))
          continue;

        // Get the vectors
        v1 = pc_->points[p1_idx].getVector4fMap ();
        v2 = pc_->points[p2_idx].getVector4fMap ();
        n1 = pc_->points[p1_idx].getNormalVector4fMap ();
        n2 = pc_->points[p2_idx].getNormalVector4fMap ();

        v12 = v1 - v2;


        // Feature computation
        dist = v12.norm() / (params_.gridsize) - 0.5;
        z = v12(2) / params_.gridsize - 0.5;
        v12.normalize();
        occ_r = occupancy_ratio(v1, v2, lut_, params_.gridsize/2) - 0.5;
        a12 = n1.dot(n2)/2.;
        a12n1 = fabs(v12.dot(n1)) - 0.5;
        a12n2 = fabs(v12.dot(n2)) - 0.5;


        // Saturate the features
        dist = std::min(std::max(dist, feat_min), feat_max);
        a12 = std::min(std::max(a12, feat_min), feat_max);
        a12n1 = std::min(std::max(a12n1, feat_min), feat_max);
        a12n2 = std::min(std::max(a12n2, feat_min), feat_max);

        if (std::isnan(dist) || std::isnan(a12) || std::isnan(a12n1) ||
            std::isnan(a12n2) || std::isnan(occ_r) || std::isnan(z))
          continue;


        // Fill in the matrix
        result[pt_idx][6*pair_nb + 0] = dist;
        result[pt_idx][6*pair_nb + 1] = occ_r;
        result[pt_idx][6*pair_nb + 2] = a12;
        result[pt_idx][6*pair_nb + 3] = a12n1;
        result[pt_idx][6*pair_nb + 4] = a12n2;
        result[pt_idx][6*pair_nb + 5] = z;

        pair_nb += 1;

        if (pair_nb >= sample_pair_nb) {
          // Break out of the two loops
          index1 = k_indices.size();
          index2 = k_indices.size();
        }
      }
    }

    // // Normalize
    // for (uint i=0; i<64; i++) {
    //   result[pt_idx][i] /= pair_nb + 1e-6;
    //   // result[pt_idx][i] -= 0.5;
    // }
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// VIZ //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void PointCloudGraphConstructor::viz(double* adj_mat) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1., "coords", 0);

  for (uint i=0; i<sampled_indices_.size(); i++) {
    int idx = sampled_indices_[i];

    if (params_.viz_small_spheres)
      viewer->addSphere<pcl::PointXYZINormal>(pc_->points[idx], 0.05, 1., 0., 0., "sphere_" +std::to_string(idx));
    else
      viewer->addSphere<pcl::PointXYZINormal>(pc_->points[idx], params_.neigh_size, 1., 0., 0., "sphere_" +std::to_string(idx));

    for (uint i2=0; i2<params_.nodes_nb; i2++) {
      if (adj_mat[params_.nodes_nb*i + i2] > 0.) {
        int idx2 = sampled_indices_[i2];
        if (idx != idx2)
          viewer->addLine<pcl::PointXYZINormal>(pc_->points[idx], pc_->points[idx2], 0., 0., 1., "line_" +std::to_string(idx)+std::to_string(idx2));
      }
    }
  }

  viewer->addPointCloud<pcl::PointXYZINormal> (pc_, "cloud");
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}
