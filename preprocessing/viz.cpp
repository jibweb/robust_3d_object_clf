 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // VIZ
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // viewer->addSphere<pcl::PointXYZINormal>(cloud->points[rdn_idx], 0.08, 0., 0., 1., "sphere_orig");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_to_view(new pcl::PointCloud<pcl::PointXYZRGB>);

  for (uint clust_idx=0; clust_idx < clust_elts.size(); clust_idx++) {
    int r = rand() % 255, g = rand() % 255, b = rand() % 255;
    if (clust_elts[clust_idx].size() < elt_in_clust/2)
      continue;

    for (uint i=0; i < clust_elts[clust_idx].size(); i++) {
      pcl::PointXYZINormal& po = cloud->points[clust_elts[clust_idx][i]];
      pcl::PointXYZRGB p(r, g, b);
      p.x = po.x;
      p.y = po.y;
      p.z = po.z;

      cloud_to_view->points.push_back(p);
      // viewer->addSphere<pcl::PointXYZINormal>(cloud->points[clust_elts[clust_idx][i]],
      //                                         0.09, r/255., g/255., b/255., "sphere_" + std::to_string(clust_idx) + "_" + std::to_string(i));
    }
  }


  std::cout << "Cloud to view: " << cloud_to_view->points.size() << " / " << cloud->points.size() << std::endl;
  // std::cout << "Cloud to view pt 0: " << cloud_to_view->points[0] << std::endl;


  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  // viewer->addPolygonMesh(triangles,"meshes",0);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud_to_view, "cloud");

  for (uint i1=0; i1 < nb_clust; i1++) {
    viewer->addSphere<pcl::PointXYZINormal>(cloud->points[centroids[i1]],
                                            0.09, 1., 0., 0., "sphere_" + std::to_string(i1));

    for (uint i2=i1+1; i2< nb_clust; i2++) {
      if (adj_mat[i1][i2])
        viewer->addLine<pcl::PointXYZINormal>(cloud->points[centroids[i1]],
                                              cloud->points[centroids[i2]], 0., 0., 1., "line_" +std::to_string(i1)+std::to_string(i2));

    }
  }

  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  while (!viewer->wasStopped ()){
      viewer->spinOnce (100);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // END VIZ
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

