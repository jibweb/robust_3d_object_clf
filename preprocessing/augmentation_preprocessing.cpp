#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>

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
