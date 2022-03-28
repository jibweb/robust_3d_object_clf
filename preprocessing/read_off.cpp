#include <fstream>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>


int main(int argc, char** argv) {
  std::ifstream infile("/home/jbweibel/dataset/ModelNet/ModelNet10/bathtub/train/bathtub_0001.off");

  std::string off;
  infile >> off;
  if (!off.compare("OFF\n"))
    std::cout << "Invalid OFF file. Received 1st line: " << off << std::endl;

  int numVertices, numFaces, numEdges;
  infile >> numVertices >> numFaces >> numEdges;

  float x, y, z;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (uint vert_idx=0; vert_idx<numVertices; vert_idx++) {
    infile >> x >> y >> z;
    pcl::PointXYZ p(x, y, z);
    cloud->points.push_back(p);
  }

  std::cout << "Cloud size: " << cloud->points.size() << std::endl;

  std::vector<std::pair<int, int> > edges;
  int numFace, idx1, idx2, idx3;
  while (infile >> numFace >> idx1 >> idx2 >> idx3) {
    std::pair<int, int> pair12(idx1, idx2);
    std::pair<int, int> pair13(idx1, idx3);
    std::pair<int, int> pair23(idx2, idx3);
    edges.push_back(pair12);
    edges.push_back(pair13);
    edges.push_back(pair23);
  }

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);

  for (uint edge_idx=0; edge_idx < edges.size(); edge_idx++) {
    std::pair<int, int> pair = edges[edge_idx];
    viewer->addLine<pcl::PointXYZ>(cloud->points[pair.first], cloud->points[pair.second], 0., 0., 1., "line_" +std::to_string(pair.first)+std::to_string(pair.second));
  }

  viewer->addPointCloud<pcl::PointXYZ> (cloud, "cloud");
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }

}