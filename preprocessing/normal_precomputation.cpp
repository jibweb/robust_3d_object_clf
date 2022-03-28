#include <algorithm>
#include <math.h>
#include <vector>

#include <boost/program_options.hpp>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>

#define GRIDSIZE 64
#define GRIDSIZE_H GRIDSIZE/2
#define PI 3.14159265

namespace po = boost::program_options;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int pre_computation (pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc,
                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud) {

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*pc, *pc, indices);

  // Scale to unit sphere
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid (*pc, centroid);
  pcl::demeanPointCloud (*pc, centroid, *pc);

   // Estimate the normals
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setInputCloud (pc);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  ne.setSearchMethod (tree);
  pcl::PointCloud<pcl::Normal>::Ptr pc_normals (new pcl::PointCloud<pcl::Normal>);
  ne.setKSearch(9);
  ne.compute (*pc_normals);


  // Copy into a PointXYZINormal
  for (uint i=0; i < pc->points.size(); i++) {
    pcl::PointXYZINormal pt;
    pt.x = pc->points[i].x;
    pt.y = pc->points[i].y;
    pt.z = pc->points[i].z;

    pt.normal_x = pc_normals->points[i].normal_x;
    pt.normal_y = pc_normals->points[i].normal_y;
    pt.normal_z = pc_normals->points[i].normal_z;
    pt.curvature = pc_normals->points[i].curvature;
    cloud->points.push_back(pt);
  }

  return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
  /***********************************
  * Args processing
  ************************************/
  po::options_description desc("Allowed options: ");

  std::string input = "";
  std::string output = "";
  desc.add_options()
      ("help,h", "produce this help message")
      ("input,i", po::value<std::string>(&input)->default_value(input), "Point cloud to load")
      ("output,o", po::value<std::string>(&output)->default_value(output), "Folder in which to save the point cloud");

  po::variables_map vm;
  po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
  std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
  po::store(parsed, vm);
  if (vm.count("help")) {
      std::cout << desc << std::endl;
      return false;
  }

  try {po::notify(vm);}
  catch(std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
      return false;
  }

  if ((output.size() > 0) && !(output[output.size()-1] == '/'))
      output += "/";

  /***********************************
  * Cloud Computation
  ************************************/
  // Point cloud reading
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (input.c_str(), *pc) == -1) {
    PCL_ERROR("Couldn't read %s file \n", input.c_str());
    return (-1);
  }

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  pre_computation(pc, cloud);



  /***********************************
  * Saving the point cloud
  ************************************/
  std::string filename;
  for (uint i=input.size()-1; i > 0; i--)
  {
      if (input[i] == '/')
      {
          filename = input.substr(i+1, input.size() - i - 5);
          break;
      }
  }

  std::string save_filename = output + filename + "_wnormals_wattention.pcd";
  pcl::io::savePCDFile (save_filename, *cloud, true);

  return 0;
}
