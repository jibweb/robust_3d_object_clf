#include <algorithm>
#include <math.h>
#include <random>
#include <vector>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "parameters.h"


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
void voxelize9 (pcl::PointCloud<pcl::PointXYZINormal> &cluster, std::vector<std::vector<std::vector<int> > > &lut_)
{
  int xi,yi,zi,xx,yy,zz;
  for (size_t i = 0; i < cluster.points.size (); ++i)
  {
    xx = cluster.points[i].x<0.0? static_cast<int>(floor(cluster.points[i].x)+GRIDSIZE_H) : static_cast<int>(ceil(cluster.points[i].x)+GRIDSIZE_H-1);
    yy = cluster.points[i].y<0.0? static_cast<int>(floor(cluster.points[i].y)+GRIDSIZE_H) : static_cast<int>(ceil(cluster.points[i].y)+GRIDSIZE_H-1);
    zz = cluster.points[i].z<0.0? static_cast<int>(floor(cluster.points[i].z)+GRIDSIZE_H) : static_cast<int>(ceil(cluster.points[i].z)+GRIDSIZE_H-1);

    for (int x = -1; x < 2; x++)
      for (int y = -1; y < 2; y++)
        for (int z = -1; z < 2; z++)
        {
          xi = xx + x;
          yi = yy + y;
          zi = zz + z;

          if (yi >= GRIDSIZE || xi >= GRIDSIZE || zi>=GRIDSIZE || yi < 0 || xi < 0 || zi < 0)
          {
            ;
          }
          else
            lut_[xi][yi][zi] = 1;
        }
  }
}


float inline surface_ratio(Eigen::Vector4f & v1, Eigen::Vector4f & v2,
                           std::vector<std::vector<std::vector<int> > > &lut_) {
  int vxlcnt, pcnt;
  float ratio;
  const int xs = v1[0] < 0.0? static_cast<int>(floor(v1[0])+GRIDSIZE_H): static_cast<int>(ceil(v1[0])+GRIDSIZE_H-1);
  const int ys = v1[1] < 0.0? static_cast<int>(floor(v1[1])+GRIDSIZE_H): static_cast<int>(ceil(v1[1])+GRIDSIZE_H-1);
  const int zs = v1[2] < 0.0? static_cast<int>(floor(v1[2])+GRIDSIZE_H): static_cast<int>(ceil(v1[2])+GRIDSIZE_H-1);
  const int xt = v2[0] < 0.0? static_cast<int>(floor(v2[0])+GRIDSIZE_H): static_cast<int>(ceil(v2[0])+GRIDSIZE_H-1);
  const int yt = v2[1] < 0.0? static_cast<int>(floor(v2[1])+GRIDSIZE_H): static_cast<int>(ceil(v2[1])+GRIDSIZE_H-1);
  const int zt = v2[2] < 0.0? static_cast<int>(floor(v2[2])+GRIDSIZE_H): static_cast<int>(ceil(v2[2])+GRIDSIZE_H-1);
  lci (lut_, xs, ys, zs, xt, yt, zt, ratio, vxlcnt, pcnt);
  return ratio;
}


void swap_if_greater(float& a, float& b) {
  if (a > b) {
    float tmp = a;
    a = b;
    b = tmp;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void global_features(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                     double* result_p,
                     double* result_t,
                     std::vector<std::vector<std::vector<int> > > &lut_,
                     Parameters & p) {

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr viz_global_pc(new pcl::PointCloud<pcl::PointXYZINormal>);
  // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  // viewer->setBackgroundColor (0, 0, 0);

  float a12, a13, a23;
  float a12n1, a12n2, a13n1, a13n3, a23n2, a23n3;
  float d12, d13, d23, r12, r13, r23;
  float s, area;
  int index1, index2, index3;
  float th1, th2, th3;
  int maxindex = static_cast<int> (pc->points.size());
  float pih = static_cast<float>(M_PI) / 2.0f;

  int pair_idx = 0;

  for (uint triplet_idx=0; triplet_idx < p.triplet_num; triplet_idx++) {
    // Get a new random triplet
    index1 = rand()%maxindex;
    index2 = rand()%maxindex;
    index3 = rand()%maxindex;

    if (index1==index2 ||
        index1==index3 ||
        index2==index3 ||
        std::isnan(pc->points[index1].normal_x) ||
        std::isnan(pc->points[index2].normal_x) ||
        std::isnan(pc->points[index3].normal_x)) {
      triplet_idx--;
      continue;
    }

    if (p.viz) {
      viz_global_pc->points.push_back(pc->points[index1]);
      viz_global_pc->points.push_back(pc->points[index2]);
      viz_global_pc->points.push_back(pc->points[index3]);
      // viewer->addLine<pcl::PointXYZINormal>(pc->points[index1], pc->points[index2], "line_" +std::to_string(index1)+std::to_string(index2));
    }

    // Getting the vectors
    Eigen::Vector4f v1 = pc->points[index1].getVector4fMap ();
    Eigen::Vector4f v2 = pc->points[index2].getVector4fMap ();
    Eigen::Vector4f v3 = pc->points[index3].getVector4fMap ();

    Eigen::Vector4f v12 = v1 - v2;
    Eigen::Vector4f v13 = v1 - v3;
    Eigen::Vector4f v23 = v2 - v3;

    // D2
    d12 = v12.norm();
    d13 = v13.norm();
    d23 = v23.norm();

    s = (d12 + d13 + d23) * 0.5f;

    if (s * (s-d12) * (s-d13) * (s-d23) <= 0.001f) {
      triplet_idx--;
      continue;
    }

    v12.normalize();
    v13.normalize();
    v23.normalize();

    // Triangle angles
    th1 = acos(fabs(v12.dot(v13))) / pih;
    th2 = acos(fabs(v12.dot(v23))) / pih;
    th3 = acos(fabs(v13.dot(v23))) / pih;

    if (th1 < 0. || th1 > 1. || std::isnan(th1)) {
      triplet_idx--;
      continue;
    }

    if (th2 < 0. || th2 > 1. || std::isnan(th2)) {
      triplet_idx--;
      continue;
    }

    if (th3 < 0. || th3 > 1. || std::isnan(th3)) {
      triplet_idx--;
      continue;
    }

    if (std::isnan(th1) || std::isnan(th2) || std::isnan(th3)) {
      std::cout << "th1: " << th1
                << "th2: " << th2
                << "th3: " << th3
                << std::endl;
      std::cout << v12 << v13 << v23 << std::endl;
    }

    swap_if_greater(th1, th2);
    swap_if_greater(th1, th3);
    swap_if_greater(th2, th3);

    // surface ratio
    r12 = surface_ratio(v1, v2, lut_);
    r13 = surface_ratio(v1, v3, lut_);
    r23 = surface_ratio(v2, v3, lut_);

    // Cosine similarity
    Eigen::Vector4f n1 = pc->points[index1].getNormalVector4fMap ();
    Eigen::Vector4f n2 = pc->points[index2].getNormalVector4fMap ();
    Eigen::Vector4f n3 = pc->points[index3].getNormalVector4fMap ();


    // cos sim between normals
    a12 = n1.dot(n2);
    a13 = n1.dot(n3);
    a23 = n2.dot(n3);

    // cos sim between vector and normals
    a12n1 = fabs(v12.dot(n1));
    a12n2 = fabs(v12.dot(n2));

    a13n1 = fabs(v13.dot(n1));
    a13n3 = fabs(v13.dot(n3));

    a23n2 = fabs(v23.dot(n2));
    a23n3 = fabs(v23.dot(n3));

    // Area according to Heron's formula
    area = std::sqrt( std::sqrt(s * (s-d12) * (s-d13) * (s-d23)) );


    result_p[p.global_feat_p_num*pair_idx + 0] = d12/static_cast<float>(GRIDSIZE);
    result_p[p.global_feat_p_num*pair_idx + 1] = a12;
    result_p[p.global_feat_p_num*pair_idx + 2] = a12n1;
    result_p[p.global_feat_p_num*pair_idx + 3] = a12n2;
    result_p[p.global_feat_p_num*pair_idx + 4] = r12;
    pair_idx++;

    result_p[p.global_feat_p_num*pair_idx + 0] = d13/static_cast<float>(GRIDSIZE);
    result_p[p.global_feat_p_num*pair_idx + 1] = a13;
    result_p[p.global_feat_p_num*pair_idx + 2] = a13n1;
    result_p[p.global_feat_p_num*pair_idx + 3] = a13n3;
    result_p[p.global_feat_p_num*pair_idx + 4] = r13;
    pair_idx++;

    result_p[p.global_feat_p_num*pair_idx + 0] = d23/static_cast<float>(GRIDSIZE);
    result_p[p.global_feat_p_num*pair_idx + 1] = a23;
    result_p[p.global_feat_p_num*pair_idx + 2] = a23n2;
    result_p[p.global_feat_p_num*pair_idx + 3] = a23n3;
    result_p[p.global_feat_p_num*pair_idx + 4] = r23;
    pair_idx++;

    result_t[p.global_feat_t_num*triplet_idx + 0] = area;
    result_t[p.global_feat_t_num*triplet_idx + 1] = th1;
    result_t[p.global_feat_t_num*triplet_idx + 2] = th2;
    result_t[p.global_feat_t_num*triplet_idx + 3] = th3;
  }

  float max_area = 0.;
  for (uint i=0; i<p.triplet_num; i++) {
    if (result_t[p.global_feat_t_num*i] > max_area)
      max_area = result_t[p.global_feat_t_num*i];
  }

  for (uint i=0; i<p.triplet_num; i++) {
    result_t[p.global_feat_t_num*i] /= max_area;
  }

  if (p.viz) {
    // while (!viewer->wasStopped()) {
    //   viewer->spinOnce(100);
    // }
    pcl::io::savePCDFile ("global_test.pcd", *viz_global_pc, true);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void sample_local_points(pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc,
                         std::vector<int> & sampled_indices,
                         pcl::search::KdTree<pcl::PointXYZINormal> & tree,
                         Parameters & p) {
  // Prepare the values for the sampling procedure
  double total_weight = 0.;
  for (uint i=0; i < pc->points.size(); i++)
    total_weight += pc->points[i].intensity;

  double uniform_prob = total_weight / static_cast<float>(pc->points.size());
  for (uint i=0; i < pc->points.size(); i++)
    pc->points[i].intensity += uniform_prob;

  total_weight *= 2;
  double rdn_weight;
  int index;
  std::vector< int > k_indices(p.neigh_size, 0);
  std::vector< float > k_sqr_distances(p.neigh_size, 0.);

  for (uint sal_pt_idx=0; sal_pt_idx < p.sal_pt_num; sal_pt_idx++) {
    // Sample a new salient point
    /*if (total_weight < 5.*uniform_prob) {
      std::cout << "Couldn't sample " << p.sal_pt_num - sal_pt_idx << " salient points" << std::endl;
      break;
    }*/
    rdn_weight = rand()%static_cast<int>(floor(std::max(total_weight, 1.)));

    for (uint i=0; i<pc->points.size(); i++){
      rdn_weight -= pc->points[i].intensity;
      if (rdn_weight < 0.) {
        index = i;
        break;
      }
    }

    if (rdn_weight >= 0.) {
      std::cout << "Couldn't sample " << p.sal_pt_num - sal_pt_idx << " salient points" << std::endl;
      break;
    }

    // Check if the sampled point is usable
    if (std::isnan(pc->points[index].normal_x)) {
      sal_pt_idx--;
      continue;
    }

    // Extract the sampled point neighborhood
    tree.nearestKSearch(pc->points[index], 2*p.neigh_size+1, k_indices, k_sqr_distances);

    // Update the attention values
    for (uint i=0; i < k_indices.size(); i++) {
      total_weight -= pc->points[k_indices[i]].intensity;
      pc->points[k_indices[i]].intensity = 0.;
    }

    sampled_indices.push_back(index);
  }
}

