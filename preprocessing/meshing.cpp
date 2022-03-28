#include <random>

#include <boost/program_options.hpp>
// #include <boost/graph/adjacency_list.hpp>
// #include <boost/property_map/property_map.hpp>
// #include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>
// #include <sys/time.h>

#include "augmentation_preprocessing.cpp"
#include "parameters.h"

namespace po = boost::program_options;

int main (int argc, char** argv) {
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

  if ((output.size() > 0) && (output[output.size()-1] != '/'))
      output += "/";

  // Load input file into a PointCloud<T> with an appropriate type
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZINormal>);
  // std::string filename = "/home/jbweibel/dataset/ModelNet/ModelNet10_TrainPc/bathtub/bathtub_0001_dist_2.000000_full_wnormals_wattention.pcd";
  if (pcl::io::loadPCDFile<pcl::PointXYZINormal> (input.c_str(), *cloud) == -1) {
    PCL_ERROR("Couldn't read %s file \n", input.c_str());
    return -1;
  }

  /***********************************
  * Meshing
  ************************************/
  // Setup
  pcl::PolygonMesh triangles;
  pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZINormal>);
  tree2->setInputCloud (cloud);
  pcl::GreedyProjectionTriangulation<pcl::PointXYZINormal> gp3;

  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (0.025 * 64);
  // Set typical values for the parameters
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (110);
  gp3.setMaximumSurfaceAngle(60*M_PI/180);
  gp3.setMinimumAngle(5*M_PI/180);
  gp3.setMaximumAngle(135*M_PI/180);
  gp3.setNormalConsistency(false);

  // Get result
  gp3.setInputCloud (cloud);
  gp3.setSearchMethod (tree2);
  gp3.reconstruct (triangles);


  /***********************************
  * Saving the point cloud
  ************************************/
  std::string filename;
  for (uint i=input.size()-1; i > 0; i--) {
    if (input[i] == '/') {
      filename = input.substr(i+1, input.size() - i - 5);
      break;
    }
  }

  std::string save_filename = output + filename + ".ply";
  pcl::io::savePLYFileBinary (save_filename , triangles);

  // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  // viewer->setBackgroundColor (0, 0, 0);
  // viewer->addPolygonMesh(triangles,"meshes",0);
  // viewer->addCoordinateSystem (1.0);
  // viewer->initCameraParameters ();
  // while (!viewer->wasStopped ()){
  //     viewer->spinOnce (100);
  // }

  // Finish
  return 0;
}
