#pragma once

#include <algorithm>
#include <math.h>

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
