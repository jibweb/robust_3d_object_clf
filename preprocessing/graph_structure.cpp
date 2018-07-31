#include <algorithm>
#include <math.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/point_types.h>

#include "parameters.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void voxelize (pcl::PointCloud<pcl::PointXYZINormal> &cluster, std::vector<std::vector<std::vector<int> > > &lut_, int gridsize)
{
  int xi,yi,zi,xx,yy,zz;
  int gridsize_h = gridsize/2;
  for (size_t i = 0; i < cluster.points.size (); ++i)
  {
    xx = cluster.points[i].x<0.0? static_cast<int>(floor(cluster.points[i].x)+gridsize_h) : static_cast<int>(ceil(cluster.points[i].x)+gridsize_h-1);
    yy = cluster.points[i].y<0.0? static_cast<int>(floor(cluster.points[i].y)+gridsize_h) : static_cast<int>(ceil(cluster.points[i].y)+gridsize_h-1);
    zz = cluster.points[i].z<0.0? static_cast<int>(floor(cluster.points[i].z)+gridsize_h) : static_cast<int>(ceil(cluster.points[i].z)+gridsize_h-1);

    for (int x = -1; x < 2; x++)
      for (int y = -1; y < 2; y++)
        for (int z = -1; z < 2; z++)
        {
          xi = xx + x;
          yi = yy + y;
          zi = zz + z;

          if (yi >= gridsize || xi >= gridsize || zi>=gridsize || yi < 0 || xi < 0 || zi < 0)
          {
            ;
          }
          else
            lut_[xi][yi][zi] = 1;
        }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int lci (std::vector<std::vector<std::vector<int> > > &lut_,
    const int x1, const int y1, const int z1,
    const int x2, const int y2, const int z2,
    float &ratio, int &incnt, int &pointcount)
{
  int voxelcount = 0;
  int voxel_in = 0;
  int act_voxel[3];
  act_voxel[0] = x1;
  act_voxel[1] = y1;
  act_voxel[2] = z1;
  int x_inc, y_inc, z_inc;
  int dx = x2 - x1;
  int dy = y2 - y1;
  int dz = z2 - z1;
  if (dx < 0)
    x_inc = -1;
  else
    x_inc = 1;
  int l = abs (dx);
  if (dy < 0)
    y_inc = -1 ;
  else
    y_inc = 1;
  int m = abs (dy);
  if (dz < 0)
    z_inc = -1 ;
  else
    z_inc = 1;
  int n = abs (dz);
  int dx2 = 2 * l;
  int dy2 = 2 * m;
  int dz2 = 2 * n;
  if ((l >= m) & (l >= n))
  {
    int err_1 = dy2 - l;
    int err_2 = dz2 - l;
    for (int i = 1; i<l; i++)
    {
      voxelcount++;;
      voxel_in +=  static_cast<int>(lut_[act_voxel[0]][act_voxel[1]][act_voxel[2]] == 1);
      if (err_1 > 0)
      {
        act_voxel[1] += y_inc;
        err_1 -=  dx2;
      }
      if (err_2 > 0)
      {
        act_voxel[2] += z_inc;
        err_2 -= dx2;
      }
      err_1 += dy2;
      err_2 += dz2;
      act_voxel[0] += x_inc;
    }
  }
  else if ((m >= l) & (m >= n))
  {
    int err_1 = dx2 - m;
    int err_2 = dz2 - m;
    for (int i=1; i<m; i++)
    {
      voxelcount++;
      voxel_in +=  static_cast<int>(lut_[act_voxel[0]][act_voxel[1]][act_voxel[2]] == 1);
      if (err_1 > 0)
      {
        act_voxel[0] +=  x_inc;
        err_1 -= dy2;
      }
      if (err_2 > 0)
      {
        act_voxel[2] += z_inc;
        err_2 -= dy2;
      }
      err_1 += dx2;
      err_2 += dz2;
      act_voxel[1] += y_inc;
    }
  }
  else
  {
    int err_1 = dy2 - n;
    int err_2 = dx2 - n;
    for (int i=1; i<n; i++)
    {
      voxelcount++;
      voxel_in +=  static_cast<int>(lut_[act_voxel[0]][act_voxel[1]][act_voxel[2]] == 1);
      if (err_1 > 0)
      {
        act_voxel[1] += y_inc;
        err_1 -= dz2;
      }
      if (err_2 > 0)
      {
        act_voxel[0] += x_inc;
        err_2 -= dz2;
      }
      err_1 += dy2;
      err_2 += dx2;
      act_voxel[2] += z_inc;
    }
  }
  voxelcount++;
  voxel_in +=  static_cast<int>(lut_[act_voxel[0]][act_voxel[1]][act_voxel[2]] == 1);
  incnt = voxel_in;
  pointcount = voxelcount;

  ratio = static_cast<float>(voxel_in) / static_cast<float>(voxelcount);
  return (2);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float inline occupancy_ratio(Eigen::Vector4f & v1, Eigen::Vector4f & v2,
                           std::vector<std::vector<std::vector<int> > > &lut_, int gridsize_h) {
  int vxlcnt, pcnt;
  float ratio;

  const int xs = v1[0] < 0.0? static_cast<int>(floor(v1[0])+gridsize_h): static_cast<int>(ceil(v1[0])+gridsize_h-1);
  const int ys = v1[1] < 0.0? static_cast<int>(floor(v1[1])+gridsize_h): static_cast<int>(ceil(v1[1])+gridsize_h-1);
  const int zs = v1[2] < 0.0? static_cast<int>(floor(v1[2])+gridsize_h): static_cast<int>(ceil(v1[2])+gridsize_h-1);
  const int xt = v2[0] < 0.0? static_cast<int>(floor(v2[0])+gridsize_h): static_cast<int>(ceil(v2[0])+gridsize_h-1);
  const int yt = v2[1] < 0.0? static_cast<int>(floor(v2[1])+gridsize_h): static_cast<int>(ceil(v2[1])+gridsize_h-1);
  const int zt = v2[2] < 0.0? static_cast<int>(floor(v2[2])+gridsize_h): static_cast<int>(ceil(v2[2])+gridsize_h-1);
  lci (lut_, xs, ys, zs, xt, yt, zt, ratio, vxlcnt, pcnt);
  return ratio;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float inline occupancy_ratio(Eigen::Vector3f & v1, Eigen::Vector3f & v2,
                           std::vector<std::vector<std::vector<int> > > &lut_, int gridsize_h) {
  int vxlcnt, pcnt;
  float ratio;

  const int xs = v1[0] < 0.0? static_cast<int>(floor(v1[0])+gridsize_h): static_cast<int>(ceil(v1[0])+gridsize_h-1);
  const int ys = v1[1] < 0.0? static_cast<int>(floor(v1[1])+gridsize_h): static_cast<int>(ceil(v1[1])+gridsize_h-1);
  const int zs = v1[2] < 0.0? static_cast<int>(floor(v1[2])+gridsize_h): static_cast<int>(ceil(v1[2])+gridsize_h-1);
  const int xt = v2[0] < 0.0? static_cast<int>(floor(v2[0])+gridsize_h): static_cast<int>(ceil(v2[0])+gridsize_h-1);
  const int yt = v2[1] < 0.0? static_cast<int>(floor(v2[1])+gridsize_h): static_cast<int>(ceil(v2[1])+gridsize_h-1);
  const int zt = v2[2] < 0.0? static_cast<int>(floor(v2[2])+gridsize_h): static_cast<int>(ceil(v2[2])+gridsize_h-1);
  lci (lut_, xs, ys, zs, xt, yt, zt, ratio, vxlcnt, pcnt);
  return ratio;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void sample_local_points(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                         std::vector<int> & sampled_indices,
                         pcl::search::KdTree<pcl::PointXYZINormal> & tree,
                         Parameters & p) {
  // Prepare the values for the sampling procedure
  int rdn_weight, index;
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;

  int total_weight = pc->points.size();
  std::vector<bool> probs(pc->points.size(), true);

  for (uint pt_idx=0; pt_idx < p.nodes_nb; pt_idx++) {
    // Sample a new point
    if (total_weight > 0) {
      rdn_weight = rand() % total_weight;
      index = -1;
    } else {
      break;
    }

    for (uint i=0; i<pc->points.size(); i++){
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
      // std::cout << "Couldn't sample " << p.nodes_nb - pt_idx << " salient points" << std::endl;
      break;
    }

    // Check if the sampled point is usable
    if (std::isnan(pc->points[index].normal_x)) {
      probs[index] = false;
      total_weight -= 1;
      pt_idx--;
      continue;
    }

    if (p.neigh_size > 0.) {
      // Extract the sampled point neighborhood
      tree.radiusSearch(pc->points[index], p.neigh_size, k_indices, k_sqr_distances);

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

    sampled_indices.push_back(index);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void occupancy_graph_structure(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                               double* adj_mat,
                               std::vector<int> & sampled_indices,
                               std::vector<std::vector<std::vector<int> > > &lut_,
                               Parameters & p) {
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (uint pt_idx=0; pt_idx < sampled_indices.size(); pt_idx++) {
    local_cloud->points.push_back(pc->points[sampled_indices[pt_idx]]);
  }

  if (p.viz) {
    pcl::io::savePCDFile ("local_cloud.pcd", *local_cloud, true);
  }

  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr local_tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
  local_tree->setInputCloud (local_cloud);
  std::vector< int > k_indices;
  std::vector< float > k_sqr_distances;

  float occ_ratio;

  for (uint pt_idx=0; pt_idx<sampled_indices.size(); pt_idx++) {
    pcl::PointXYZINormal pt = local_cloud->points[pt_idx];
    local_tree->nearestKSearch(pt, p.neigh_nb, k_indices, k_sqr_distances);

    // float mean_dist = 0.;
    // for (uint i=1; i<p.neigh_nb; i++) {
    //   mean_dist += k_sqr_distances[i];
    // }
    // mean_dist /= p.neigh_nb;

    Eigen::Vector4f v1 = local_cloud->points[pt_idx].getVector4fMap ();
    adj_mat[p.nodes_nb*pt_idx + pt_idx] = 1.;
    for (uint i=0; i<p.neigh_nb; i++) {
      if (k_sqr_distances[i] == 0.)
        continue;

      Eigen::Vector4f v2 = pc->points[k_indices[i]].getVector4fMap ();
      occ_ratio = occupancy_ratio(v1, v2, lut_, p.gridsize/2);
      if (occ_ratio > 0.0) { //} && k_sqr_distances[i] < 1.1*mean_dist) {
        adj_mat[p.nodes_nb*pt_idx + k_indices[i]] = 1.;
        adj_mat[p.nodes_nb*k_indices[i] + pt_idx] = 1.;
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void edge_feats_adjacency(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                          double* adj_mat,
                          double* edge_feats_mat,
                          std::vector<int> & sampled_indices,
                          std::vector<std::vector<std::vector<int> > > &lut_,
                          Parameters & p) {
  int index1, index2;
  float d, occ_r, a12, a12n1, a12n2;
  float feat_min = -0.5;
  float feat_max =  0.5;


  for (uint pt1=0; pt1<sampled_indices.size(); pt1++) {
    for (uint pt2=pt1; pt2<sampled_indices.size(); pt2++) {

      // if (!adj_mat[p.nodes_nb*pt1 + pt2])
      //   continue;

      if (pt1 == pt2) {
        edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 0] = -0.5;
        edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 1] =  0.5;
        edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 2] =  0.5;
        edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 3] = -0.5;
        edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 4] = -0.5;
        continue;
      }

      index1 = sampled_indices[pt1];
      index2 = sampled_indices[pt2];

      // Vector setup
      Eigen::Vector4f v1 = pc->points[index1].getVector4fMap ();
      Eigen::Vector4f v2 = pc->points[index2].getVector4fMap ();
      Eigen::Vector4f n1 = pc->points[index1].getNormalVector4fMap ();
      Eigen::Vector4f n2 = pc->points[index2].getNormalVector4fMap ();
      Eigen::Vector4f v12 = v1 - v2;

      // Feature computation
      d = v12.norm() / (p.gridsize) - 0.5;
      v12.normalize();
      occ_r = occupancy_ratio(v1, v2, lut_, p.gridsize/2) - 0.5;
      a12 = n1.dot(n2)/2;
      a12n1 = fabs(v12.dot(n1)) - 0.5;
      a12n2 = fabs(v12.dot(n2)) - 0.5;

      // Saturate the features

      d = std::min(std::max(d, feat_min), feat_max);
      a12 = std::min(std::max(a12, feat_min), feat_max);
      a12n1 = std::min(std::max(a12n1, feat_min), feat_max);
      a12n2 = std::min(std::max(a12n2, feat_min), feat_max);

      // Fill in the matrix
      edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 0] = d;
      edge_feats_mat[5*(p.nodes_nb*pt2 + pt1) + 0] = d;
      edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 1] = occ_r;
      edge_feats_mat[5*(p.nodes_nb*pt2 + pt1) + 1] = occ_r;
      edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 2] = a12;
      edge_feats_mat[5*(p.nodes_nb*pt2 + pt1) + 2] = a12;
      edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 3] = a12n1;
      edge_feats_mat[5*(p.nodes_nb*pt2 + pt1) + 3] = a12n1;
      edge_feats_mat[5*(p.nodes_nb*pt1 + pt2) + 4] = a12n2;
      edge_feats_mat[5*(p.nodes_nb*pt2 + pt1) + 4] = a12n2;
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void coords_feats_adjacency(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                            double* adj_mat,
                            double* edge_feats_mat,
                            pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
                            std::vector<int> & sampled_indices,
                            std::vector<std::vector<std::vector<int> > > &lut_,
                            Parameters & p) {

  int index1, index2;
  float d, a12, occ_r;
  pcl::PointCloud<pcl::ReferenceFrame> lrf_pc;
  boost::shared_ptr<std::vector<int> > sampled_indices_ptr(new std::vector<int> (sampled_indices));

  pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZINormal, pcl::ReferenceFrame> lrf_estimator;
  lrf_estimator.setRadiusSearch (2*p.neigh_size);
  lrf_estimator.setInputCloud (pc);
  lrf_estimator.setSearchMethod (tree);
  lrf_estimator.setIndices (sampled_indices_ptr);
  lrf_estimator.compute (lrf_pc);

  for (uint pt1_idx=0; pt1_idx < sampled_indices.size(); pt1_idx++) {
    index1 = sampled_indices[pt1_idx];
    Eigen::Matrix3f local_rf = lrf_pc.points[pt1_idx].getMatrix3fMap();
    local_rf.normalize();
    Eigen::Vector3f v1 = pc->points[index1].getVector3fMap();
    Eigen::Vector4f n1 = pc->points[index1].getNormalVector4fMap ();

    if (std::isnan(local_rf(0)) ||
        std::isnan(local_rf(1)) ||
        std::isnan(local_rf(2))){
      if (p.debug)
        std::cout << "Garbage LRF: " << pt1_idx << std::endl;
      for (uint i=0; i<p.nodes_nb; i++) {
        edge_feats_mat[6*(p.nodes_nb*pt1_idx + i) + 0] = 2.;
        edge_feats_mat[6*(p.nodes_nb*pt1_idx + i) + 5] = -1.;
      }
      continue;
    }

    for (uint pt2_idx=0; pt2_idx < sampled_indices.size(); pt2_idx++) {
      index2 = sampled_indices[pt2_idx];
      Eigen::Vector3f v2 = pc->points[index2].getVector3fMap();
      Eigen::Vector4f n2 = pc->points[index2].getNormalVector4fMap ();
      Eigen::Vector3f v21 = v2 - v1;
      Eigen::Vector3f new_coords = local_rf * v21;
      d = v21.norm() / (p.gridsize) - 0.5;
      v21.normalize();
      occ_r = occupancy_ratio(v1, v2, lut_, p.gridsize/2);
      a12 = n1.dot(n2)/2;

      edge_feats_mat[6*(p.nodes_nb*pt1_idx + pt2_idx) + 0] = d;
      edge_feats_mat[6*(p.nodes_nb*pt1_idx + pt2_idx) + 1] = new_coords(0) / p.gridsize;
      edge_feats_mat[6*(p.nodes_nb*pt1_idx + pt2_idx) + 2] = new_coords(1) / p.gridsize;
      edge_feats_mat[6*(p.nodes_nb*pt1_idx + pt2_idx) + 3] = new_coords(2) / p.gridsize;
      edge_feats_mat[6*(p.nodes_nb*pt1_idx + pt2_idx) + 4] = a12;
      edge_feats_mat[6*(p.nodes_nb*pt1_idx + pt2_idx) + 5] = occ_r;
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_graph(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                   pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
                   std::vector<int> & sampled_indices,
                   double* adj_mat,
                   Parameters params) {
  // Sample graph nodes
  srand (static_cast<unsigned int> (time (0)));
  sample_local_points(pc, sampled_indices, *tree, params);

  if (params.debug)
    std::cout << "Salient points sampled: " << sampled_indices.size() << std::endl;


  // Voxelization
  std::vector<std::vector<std::vector<int> > > lut_;
  lut_.resize (params.gridsize);
  for (uint i = 0; i < params.gridsize; ++i) {
      lut_[i].resize (params.gridsize);
      for (uint j = 0; j < params.gridsize; ++j)
        lut_[i][j].resize (params.gridsize);
  }
  voxelize (*pc, lut_, params.gridsize);


  // Graph structure
  occupancy_graph_structure(pc, adj_mat, sampled_indices, lut_, params);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void compute_edge_graph(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                   pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree,
                   std::vector<int> & sampled_indices,
                   double* adj_mat,
                   double* edge_feats_mat,
                   Parameters params) {
  // Sample graph nodes
  srand (static_cast<unsigned int> (time (0)));
  sample_local_points(pc, sampled_indices, *tree, params);

  if (params.debug)
    std::cout << "Salient points sampled: " << sampled_indices.size() << std::endl;


  // Voxelization
  std::vector<std::vector<std::vector<int> > > lut_;
  lut_.resize (params.gridsize);
  for (uint i = 0; i < params.gridsize; ++i) {
      lut_[i].resize (params.gridsize);
      for (uint j = 0; j < params.gridsize; ++j)
        lut_[i][j].resize (params.gridsize);
  }
  voxelize (*pc, lut_, params.gridsize);


  // Graph structure
  // occupancy_graph_structure(pc, adj_mat, sampled_indices, lut_, params);
  if (params.edge_feat_nb == 5)
    edge_feats_adjacency(pc, adj_mat, edge_feats_mat, sampled_indices, lut_, params);
  else if (params.edge_feat_nb == 6)
    coords_feats_adjacency(pc, adj_mat, edge_feats_mat, tree, sampled_indices, lut_, params);

}