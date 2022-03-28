#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <pcl/filters/filter.h>
#include <pcl/visualization/cloud_viewer.h>
#include <v4r/common/camera.h>
// #include <v4r/common/pcl_opencv.h>
#include <v4r/rendering/depthmapRenderer.h>
#include <v4r/rendering/dmRenderObject.h>

#include "scope_time.h"

namespace po = boost::program_options;

int
main(int argc, char* argv[]) {

    ScopeTime t_total("Total computation", false);
    /***********************************
    * Args processing
    ************************************/
    po::options_description desc("Generate depth map centered around the closest object in the scene\n");

    std::string input = "";
    std::string output = "";
    bool visualize_ = false;

    desc.add_options()
        ("help,h", "produce this help message")
        ("input,i", po::value<std::string>(&input)->default_value(input), "Mesh to render")
        ("output,o", po::value<std::string>(&output)->default_value(output), "Folder in which to save the point clouds")
        ("visualize,v", po::bool_switch(&visualize_), "visualize results");

    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try {po::notify(vm);}
    catch(std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; return false;
    }

    if ((output.size() > 0) && !(output[output.size()-1] == '/'))
        output += "/";


    /***********************************
    * CAD Rendering setup
    ************************************/
    v4r::Camera::Ptr cam(new v4r::Camera(525.5f, 525.f, 640, 480, 319.5f, 239.5f, 58.f));
    v4r::DepthmapRendererModel model = v4r::DepthmapRendererModel(input);
    v4r::DepthmapRenderer dmr = v4r::DepthmapRenderer(cam->getWidth(), cam->getHeight());
    float sphereDistance = 2.f;
    std::vector<Eigen::Vector3f> spherePositions = dmr.createSphere(sphereDistance, 0);
    dmr.setIntrinsics(cam->getFocalLengthX(), cam->getFocalLengthY(), cam->getCx(), cam->getCy());
    dmr.setModel(&model);

    for (int sp = 0; sp < spherePositions.size(); sp++) {
        Eigen::Vector3f position = spherePositions[sp];
        Eigen::Matrix4f pose = dmr.getPoseLookingToCenterFrom(position);
        dmr.setCamPose(pose);

        float visibleSurfaceArea;
        // std::cout << " --- \n" << std::endl;
        // std::cout << "POSE: \n" << pose << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        // cv::Mat dImg;

        {
            ScopeTime t_view("Single view", false);
            /***********************************
            * Rendering of a point cloud
            ************************************/
            std::vector<int> mapping;
            pcl::removeNaNFromPointCloud(dmr.renderPointcloudColor(visibleSurfaceArea), *cloud, mapping);

            // Turn the object upside down
            // Eigen::Matrix3f rotation = getRotation(0., 0., 180.);
            // Eigen::Vector3f translation;
            // translation << 0., 0., 0.;
            // transfoCloud(cloud, cloud,  rotation, translation);

            if (visualize_)
            {
                pcl::visualization::CloudViewer viewer ("Seg viewer");
                viewer.showCloud(cloud);
                while (!viewer.wasStopped ())
                {}
            }



            // /***********************************
            // * Projection in the image plane
            // ************************************/
            // v4r::PCLOpenCVConverter<pcl::PointXYZRGB> converter = v4r::PCLOpenCVConverter<pcl::PointXYZRGB>();
            // converter.setBackgroundColor(0,0,0);
            // converter.setCamera(cam);
            // converter.setInputCloud(cloud);
            // dImg = converter.extractDepth();
        }

        // // Transform the depth into a proper depth map from the xtion
        // // (mm encoded as uint16 instead of m encoded as float)
        // cv::Mat depthImg =  cv::Mat_<uint16_t>(cam->getHeight(), cam->getWidth());
        // depthImg.setTo( static_cast<uint16_t>(0));

        // for (uint h=0; h < cam->getHeight(); h++)
        //     for(uint w=0; w < cam->getWidth(); w++)
        //         depthImg.at<uint16_t>(h, w) = static_cast<uint16_t>(1000. * dImg.at<float>(h, w));

        // if (visualize_)
        // {
        //     cv::imshow("Depthmap", depthImg);
        //     cv::waitKey(0);
        // }

        std::string filename;
        for (uint i=input.size()-1; i > 0; i--)
        {
            if (input[i] == '/')
            {
                filename = input.substr(i+1, input.size() - i - 5);
                break;
            }
        }
        filename += "_view_";
        if (sp < 10)
          filename += "0";
        filename += std::to_string(sp) + ".pcd";
        pcl::io::savePCDFileBinary<pcl::PointXYZRGB>(output + filename, *cloud);
        // std::cout << filename << " " << sp << " / " << spherePositions.size() << std::endl;
        // cv::imwrite(output + filename + "_depth_div_dist" + std::to_string(sphereDistance) + "_" + std::to_string(sp) + ".png",
        //             depthImg, compression_params);
    }

    return 0;
}
