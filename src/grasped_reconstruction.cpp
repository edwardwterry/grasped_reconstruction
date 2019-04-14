#include <grasped_reconstruction/grasped_reconstruction.h>

const float PI_F = 3.14159265358979f;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
class GraspedReconstruction
{
public:
  GraspedReconstruction(ros::NodeHandle &n) : _n(n)
  {
    _n = n;
    // pc_sub = _n.subscribe("/camera/depth/points", 1, &GraspedReconstruction::pcClbk, this);
    // ch_sub = _n.subscribe("/extract_plane_indices/output", 1, &GraspedReconstruction::convexHullClbk, this);
    // hm_sub = _n.subscribe("/camera/depth/points", 1, &GraspedReconstruction::heightMapClbk, this);
    pc_sub = _n.subscribe("/camera/depth/points", 1, &GraspedReconstruction::pcClbk, this);
    occ_sub = _n.subscribe("/camera/depth/points", 1, &GraspedReconstruction::occClbk, this);
    gm_sub = _n.subscribe("/elevation_mapping/elevation_map", 1, &GraspedReconstruction::gmClbk, this);
    occ_pub = n.advertise<sensor_msgs::PointCloud2>("occluded_voxels", 1);
    // ch_pub = n.advertise<pcl_msgs::PolygonMesh>("convex_hull_mesh", 1);
    // hm_pub = n.advertise<sensor_msgs::PointCloud2>("object_without_table", 1);
    coeff_pub = n.advertise<pcl_msgs::ModelCoefficients>("output", 1);
    combo_pub = n.advertise<sensor_msgs::PointCloud2>("combo", 1);
    object_pub = n.advertise<sensor_msgs::PointCloud2>("segmented_object", 1);
    tabletop_pub = n.advertise<sensor_msgs::PointCloud2>("tabletop", 1);
    bb_pub = n.advertise<visualization_msgs::Marker>("bbox", 1);
    cf_pub = n.advertise<sensor_msgs::PointCloud2>("color_filtered", 1);
    image_transport::ImageTransport it(n);
    hm_im_pub = it.advertise("height_map_image", 1);
    try
    {
      ros::Time now = ros::Time::now();
      listener.waitForTransform("/robot_base", "/lens_link",
                                now, ros::Duration(3.0));
      listener.lookupTransform("/robot_base", "/lens_link",
                               now, world_T_lens_link_tf);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }
    std::cout << "robot_base to lens_link received" << std::endl;

    _n.getParam("/bMax", bMax);
    _n.getParam("/gMax", gMax);
    _n.getParam("/rMax", rMax);
    _n.getParam("/bMin", bMin);
    _n.getParam("/gMin", gMin);
    _n.getParam("/rMin", rMin);
  }
  ros::NodeHandle _n;
  ros::Subscriber pc_sub, gm_sub, occ_sub;
  ros::Publisher coeff_pub, object_pub, tabletop_pub, bb_pub, cf_pub, occ_pub, combo_pub;
  image_transport::Publisher hm_im_pub;
  tf::TransformListener listener;
  tf::StampedTransform world_T_lens_link_tf;
  int rMax, rMin, gMax, gMin, bMax, bMin;
  PointCloud combo_orig, orig_observed, orig_unobserved;
  bool orig_observed_set = false;
  bool orig_unobserved_set = false;

  // canonical bounding box
  pcl::PointXYZ orig_bb_min_, orig_bb_max_;
  std::vector<float> leaf_size_;
  int nr_, nc_, nl_;

  void make_combo()
  {
    if (orig_observed_set && orig_unobserved_set)
    {
      // std::cout << "orig_observed: " << std::endl;
      // for (size_t i = 0; i < orig_observed.points.size(); ++i)
      //   std::cout << "    " << orig_observed.points[i].x << " " << orig_observed.points[i].y << " " << orig_observed.points[i].z << std::endl;

      // std::cout << "orig_unobserved: " << std::endl;
      // for (size_t i = 0; i < orig_unobserved.points.size(); ++i)
      //   std::cout << "    " << orig_unobserved.points[i].x << " " << orig_unobserved.points[i].y << " " << orig_unobserved.points[i].z << std::endl;
      combo_orig = orig_observed;
      combo_orig += orig_unobserved;

      // std::cout << "combo_orig: " << std::endl;
      // for (size_t i = 0; i < combo_orig.points.size(); ++i)
      //   std::cout << "    " << combo_orig.points[i].x << " " << combo_orig.points[i].y << " " << combo_orig.points[i].z << std::endl;
      PointCloud::Ptr cl(new PointCloud);
      *cl = combo_orig;
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(cl);
      sor.setMeanK(50);
      sor.setStddevMulThresh(1.0);
      sor.filter(*cl);
      // pcl::getMinMax3D(*cl, orig_bb_min_, orig_bb_max_);
      // pcl::VoxelGrid<pcl::PointXYZ> vg;
      // vg.setInputCloud(cl); // will this go out of scope!?!??!
      // vg.setLeafSize(0.01, 0.01, 0.01);
      // vg.filter(*cl);

      //publish
      sensor_msgs::PointCloud2 out;
      pcl::toROSMsg(*cl, out);
      combo_pub.publish(out);

      // http://pointclouds.org/documentation/tutorials/kdtree_search.php
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_obs;
      PointCloud::Ptr cl_orig_obs(new PointCloud);
      *cl_orig_obs = orig_observed;
      kdtree_obs.setInputCloud(cl_orig_obs);
      std::cout << "\n\nLooking for associations in observed pc" << std::endl;
      for (PointCloud::iterator it = cl->begin(); it != cl->end(); it++) // for each in the combined cloud
      {
        std::cout << "Querying point: " << *it << std::endl;
        int K = 1; // only look for the closest
        std::vector<int> pointIdxNKNSearch;
        pointIdxNKNSearch.resize(K);
        std::vector<float> pointNKNSquaredDistance;
        pointNKNSquaredDistance.resize(K);
        if (kdtree_obs.nearestKSearch(*it, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) // search in the observed tree
        {
          for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
            std::cout << "    " << cl_orig_obs->points[pointIdxNKNSearch[i]].x
                      << " " << cl_orig_obs->points[pointIdxNKNSearch[i]].y
                      << " " << cl_orig_obs->points[pointIdxNKNSearch[i]].z
                      << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
        }
      }
      std::cout << "\n\nLooking for associations in unobserved pc" << std::endl;

      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_unobs;
      PointCloud::Ptr cl_orig_unobs(new PointCloud);
      *cl_orig_unobs = orig_unobserved;
      kdtree_unobs.setInputCloud(cl_orig_unobs);
      for (PointCloud::iterator it = cl->begin(); it != cl->end(); it++) // for each in the combined cloud
      {
        std::cout << "Querying point: " << *it << std::endl;
        int K = 3; // only look for the closest
        std::vector<int> pointIdxNKNSearch;
        pointIdxNKNSearch.resize(K);
        std::vector<float> pointNKNSquaredDistance;
        pointNKNSquaredDistance.resize(K);
        if (kdtree_unobs.nearestKSearch(*it, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) // search in the observed tree
        {
          for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
            std::cout << "    " << cl_orig_unobs->points[pointIdxNKNSearch[i]].x
                      << " " << cl_orig_unobs->points[pointIdxNKNSearch[i]].y
                      << " " << cl_orig_unobs->points[pointIdxNKNSearch[i]].z
                      << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
        }
      }
      orig_observed_set = false;
    }
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time();
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0.5f * (orig_bb_max_.x + orig_bb_min_.x);
    marker.pose.position.y = 0.5f * (orig_bb_max_.y + orig_bb_min_.y);
    marker.pose.position.z = 0.5f * (orig_bb_max_.z + orig_bb_min_.z);
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = orig_bb_max_.x - orig_bb_min_.x;
    marker.scale.y = orig_bb_max_.y - orig_bb_min_.y;
    marker.scale.z = orig_bb_max_.z - orig_bb_min_.z;
    marker.color.a = 0.5; // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 0.5;
    marker.color.b = 0.0;
    bb_pub.publish(marker);
  }

  void pcClbk(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    std::cout << "Received pc message" << std::endl;
    // http://wiki.ros.org/pcl/Tutorials#pcl.2BAC8-Tutorials.2BAC8-hydro.sensor_msgs.2BAC8-PointCloud2
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    sensor_msgs::PointCloud2Ptr msg_transformed(new sensor_msgs::PointCloud2());
    std::string target_frame("world");
    pcl_ros::transformPointCloud(target_frame, *msg, *msg_transformed, listener);
    PointCloud::Ptr cloud(new PointCloud());
    pcl::fromROSMsg(*msg_transformed, *cloud);

    // remove the ground plane
    // http://pointclouds.org/documentation/tutorials/passthrough.php
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.5, 0.5);
    pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
    pass.filter(*cloud);

    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.1, 0.3);
    pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
    pass.filter(*cloud);

    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.1, 0.1);
    pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
    pass.filter(*cloud);
    std::cout << "Removed floor" << std::endl;

    // Downsample this pc
    pcl::VoxelGrid<pcl::PointXYZ> downsample;
    downsample.setInputCloud(cloud);
    downsample.setLeafSize(0.01f, 0.01f, 0.01f);
    downsample.filter(*cloud);

    sensor_msgs::PointCloud2 tabletop_output;
    pcl::toROSMsg(*cloud, tabletop_output);
    tabletop_pub.publish(tabletop_output);

    // // obsolete --> coefficients for plane segmentation
    // pcl::ModelCoefficients coefficients;
    // pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    // // Create the segmentation object
    // pcl::SACSegmentation<pcl::PointXYZ> seg;
    // // Optional
    // seg.setOptimizeCoefficients(true);
    // // Mandatory
    // seg.setModelType(pcl::SACMODEL_PLANE);
    // seg.setMethodType(pcl::SAC_RANSAC);
    // seg.setDistanceThreshold(0.01);
    // // remove the tabletop
    // seg.setInputCloud(cloud);
    // seg.segment(*inliers, coefficients);
    // std::cout << "Removed tabletop" << std::endl;

    // // Publish the model coefficients
    // pcl_msgs::ModelCoefficients ros_coefficients;
    // pcl_conversions::fromPCL(coefficients, ros_coefficients);
    // coeff_pub.publish(ros_coefficients);

    // // end obsolete

    // pcl::ExtractIndices<pcl::PointXYZ> extract;
    // extract.setInputCloud(cloud);
    // extract.setIndices(inliers);
    // extract.setNegative(true);
    // extract.filter(*cloud);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.5, 0.735);
    pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
    pass.filter(*cloud);

    // PointCloud::Ptr cloud_tabletop(new PointCloud());
    // pcl::ExtractIndices<pcl::PointXYZ> extract;
    // extract.setInputCloud(cloud);
    // extract.setIndices(inliers);
    // extract.setNegative(false);
    // extract.filter(*cloud_tabletop);

    // pass.setFilterFieldName("x");
    // pass.setFilterLimits(-3, -0.2);
    // pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
    // pass.filter(*cloud);
    // std::cout << "Removed notch" << std::endl;
    orig_observed = *cloud;
    orig_observed_set = true;
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    object_pub.publish(output);
    std::cout << "Published cloud" << std::endl;

    pcl::PointXYZ min, max;
    pcl::getMinMax3D(*cloud, min, max);
    std::cout << "Got bounding box" << std::endl;

    // ////// occlusion begin
    // pcl::PCLPointCloud2 cloud_filtered2;
    // std::string lens_frame("lens_link");
    // pcl_ros::transformPointCloud(lens_frame, *cloud, *cloud, listener);

    // pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

    // occ.setInputCloud(cloud);
    // occ.setLeafSize(0.01, 0.01, 0.01);
    // occ.initializeVoxelGrid();
    // // occ.filter(cloud_filtered);
    // // Eigen::Vector3f out = occ.getMaxBoundCoordinates();
    // // std::cout<<"getMaxBoundCoordinates: "<<out.matrix()<<std::endl;
    // // // occ.initializeVoxelGrid();
    // pcl_ros::transformPointCloud(target_frame, *cloud, *cloud, listener);
    // Eigen::Vector3i box = occ.getMaxBoxCoordinates();

    // PointCloud cloud_filtered = occ.getFilteredPointCloud();
    // std::vector<Eigen::Vector3i> occluded_voxels;
    // occ.occlusionEstimationAll(occluded_voxels);
    // std::cout << "Proportion occluded: " << (float)occluded_voxels.size() / (float)(box(0) * box(1) * box(2)) << std::endl;
    // PointCloud cloud_occluded;
    // for (const auto &voxel : occluded_voxels)
    // {
    //   // std::cout<<"voxel(0): "<<voxel(0)<<std::endl;
    //   Eigen::Vector4f coord = occ.getCentroidCoordinate(voxel);
    //   cloud_occluded.push_back(pcl::PointXYZ(coord(0), coord(1), coord(2)));
    // }
    // // pcl::toPCLPointCloud2(cloud_filtered, cloud_filtered2);
    // // // Convert to ROS data type
    // pcl::toPCLPointCloud2(cloud_occluded, cloud_filtered2);

    // sensor_msgs::PointCloud2Ptr output2(new sensor_msgs::PointCloud2());

    // pcl_conversions::fromPCL(cloud_filtered2, *output2);
    // output2->header.frame_id = std::string("lens_link");
    // sensor_msgs::PointCloud2Ptr msg_transformed2(new sensor_msgs::PointCloud2());
    // std::string target_frame2("world");
    // pcl_ros::transformPointCloud(target_frame2, *output2, *msg_transformed2, listener);
    // msg_transformed2->header.frame_id = "world";
    // // output.height = 640;
    // // output.width = 480;

    // // Publish the data
    // occ_pub.publish(*msg_transformed2);

    // ////// occlusion end

    // // pcl::PointCloud<pcl::PointXYZHSV>::Ptr hsv_cloud(new pcl::PointCloud<pcl::PointXYZHSV>);

    // // http://www.pcl-users.org/How-to-filter-based-on-color-using-PCL-td2791524.html
    // // Filter for color
    // // build the condition
    // pcl::ConditionAnd<pcl::PointXYZ>::Ptr color_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
    // if (_n.hasParam("/rMax"))
    // {
    //   color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZ>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZ>("h", pcl::ComparisonOps::LT, rMax)));
    // }
    // if (_n.hasParam("/rMin"))
    // {
    //   color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZ>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZ>("h", pcl::ComparisonOps::GT, rMin)));
    // }
    // if (_n.hasParam("/gMax"))
    // {
    //   color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZ>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZ>("s", pcl::ComparisonOps::LT, gMax)));
    // }
    // if (_n.hasParam("/gMin"))
    // {
    //   color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZ>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZ>("s", pcl::ComparisonOps::GT, gMin)));
    // }
    // if (_n.hasParam("/bMax"))
    // {
    //   color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZ>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZ>("i", pcl::ComparisonOps::LT, bMax)));
    // }
    // if (_n.hasParam("/bMin"))
    // {
    //   color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZ>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZ>("i", pcl::ComparisonOps::GT, bMin)));
    // }

    // // build the filter
    // pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
    // condrem.setCondition(color_cond);
    // condrem.setInputCloud(cloud);
    // condrem.setKeepOrganized(false);

    // // apply filter
    // condrem.filter(*cloud);
    // sensor_msgs::PointCloud2 cloud_by_color_sm;
    // pcl::toROSMsg(*cloud, cloud_by_color_sm);
    // cf_pub.publish(cloud_by_color_sm);
    // http://www.pointclouds.org/documentation/tutorials/region_growing_rgb_segmentation.php
    // pcl::search::Search <pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    // pcl::RegionGrowingRGB<pcl::PointXYZ> reg;
    // reg.setInputCloud(cloud);
    // reg.setIndices(inliers);
    // reg.setSearchMethod(tree);
    // reg.setDistanceThreshold(10);
    // reg.setPointColorThreshold(6);
    // reg.setRegionColorThreshold(5);
    // reg.setMinClusterSize(10);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZ>); // = reg.getColoredCloud();
    // colored_cloud = reg.getColoredCloud();
    // sensor_msgs::PointCloud2 cloud_by_color_sm;
    // pcl::toROSMsg(*colored_cloud, cloud_by_color_sm);
    // cf_pub.publish(cloud_by_color_sm);

    // if (coefficients.values.size() > 0)
    // {
    //   visualization_msgs::Marker marker;
    //   marker.header.frame_id = "world";
    //   marker.header.stamp = ros::Time();
    //   marker.type = visualization_msgs::Marker::CUBE;
    //   marker.action = visualization_msgs::Marker::ADD;
    //   marker.pose.position.x = 0.5f * (max.x + min.x);
    //   marker.pose.position.y = 0.5f * (max.y + min.y);
    //   marker.pose.position.z = 0.5f * (max.z - coefficients.values[3]);
    //   marker.pose.orientation.x = 0.0;
    //   marker.pose.orientation.y = 0.0;
    //   marker.pose.orientation.z = 0.0;
    //   marker.pose.orientation.w = 1.0;
    //   marker.scale.x = max.x - min.x;
    //   marker.scale.y = max.y - min.y;
    //   marker.scale.z = max.z + coefficients.values[3];
    //   marker.color.a = 0.5; // Don't forget to set the alpha!
    //   marker.color.r = 1.0;
    //   marker.color.g = 0.5;
    //   marker.color.b = 0.0;
    //   bb_pub.publish(marker);
    // }
  }

  void occClbk(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {

    std::cout << "Received pc message" << std::endl;
    // http://wiki.ros.org/pcl/Tutorials#pcl.2BAC8-Tutorials.2BAC8-hydro.sensor_msgs.2BAC8-PointCloud2
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    sensor_msgs::PointCloud2Ptr msg_transformed(new sensor_msgs::PointCloud2());
    std::string target_frame("world");
    std::string lens_frame("lens_link");
    pcl_ros::transformPointCloud(target_frame, *cloud_msg, *msg_transformed, listener);
    PointCloud::Ptr cloud(new PointCloud());
    pcl::fromROSMsg(*msg_transformed, *cloud);

    // remove the ground plane
    // http://pointclouds.org/documentation/tutorials/passthrough.php
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.5, 0.73);
    pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
    pass.filter(*cloud);

    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.1, 0.3);
    pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
    pass.filter(*cloud);

    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.1, 0.1);
    pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
    pass.filter(*cloud);
    std::cout << "Removed floor" << std::endl;

    // Downsample this pc
    pcl::VoxelGrid<pcl::PointXYZ> downsample;
    downsample.setInputCloud(cloud);
    downsample.setLeafSize(0.01f, 0.01f, 0.01f);
    downsample.filter(*cloud);

    // pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2;
    // pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::PCLPointCloud2 cloud_filtered2;

    // Convert to PCL data type
    // pcl_conversions::toPCL(*cloud_msg, *cloud);

    // PointCloud *pc = new pcl::PointCloud<pcl::PointXYZ>;
    // // https://stackoverflow.com/questions/10644429/create-a-pclpointcloudptr-from-a-pclpointcloud
    // PointCloud::Ptr pcPtr(pc);

    // pcl::fromPCLPointCloud2(*cloud, *pc);
    pcl_ros::transformPointCloud(lens_frame, *cloud, *cloud, listener);

    pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

    occ.setInputCloud(cloud);
    occ.setLeafSize(0.01, 0.01, 0.01);
    occ.initializeVoxelGrid();
    // occ.filter(cloud_filtered);
    // Eigen::Vector3f out = occ.getMaxBoundCoordinates();
    // std::cout<<"getMaxBoundCoordinates: "<<out.matrix()<<std::endl;
    // // occ.initializeVoxelGrid();
    Eigen::Vector3i box = occ.getMaxBoxCoordinates();

    PointCloud cloud_filtered = occ.getFilteredPointCloud();
    std::vector<Eigen::Vector3i> occluded_voxels;
    occ.occlusionEstimationAll(occluded_voxels);
    std::cout << "Proportion occluded: " << (float)occluded_voxels.size() / (float)(box(0) * box(1) * box(2)) << std::endl;
    PointCloud::Ptr cloud_occluded(new PointCloud);
    for (const auto &voxel : occluded_voxels)
    {
      // std::cout<<"voxel(0): "<<voxel(0)<<std::endl;
      Eigen::Vector4f coord = occ.getCentroidCoordinate(voxel);
      cloud_occluded->push_back(pcl::PointXYZ(coord(0), coord(1), coord(2)));
    }
    // pcl::toPCLPointCloud2(cloud_filtered, cloud_filtered2);
    // // Convert to ROS data type
    // pcl::PassThrough<pcl::PointXYZ> pass;

    //convert to world
    cloud_occluded->header.frame_id = "lens_link";
    pcl_ros::transformPointCloud(target_frame, *cloud_occluded, *cloud_occluded, listener);

    pass.setInputCloud(cloud_occluded);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.5, 0.73);
    pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
    pass.filter(*cloud_occluded);
    orig_unobserved = *cloud_occluded;
    orig_unobserved_set = true;

    pcl::toPCLPointCloud2(*cloud_occluded, cloud_filtered2);

    sensor_msgs::PointCloud2Ptr output(new sensor_msgs::PointCloud2());

    pcl_conversions::fromPCL(cloud_filtered2, *output);
    // output->header.frame_id = std::string("lens_link");
    // sensor_msgs::PointCloud2Ptr msg_transformed2(new sensor_msgs::PointCloud2());
    // std::string target_frame2("world");
    // pcl_ros::transformPointCloud(target_frame2, *output, *msg_transformed2, listener);
    output->header.frame_id = "world";
    // output.height = 640;
    // output.width = 480;

    // Publish the data
    occ_pub.publish(*output);
  }

  void gmClbk(const grid_map_msgs::GridMap::ConstPtr &msg)
  {
    cv_bridge::CvImagePtr cvImage(new cv_bridge::CvImage);
    grid_map::GridMap gridMap;
    ROS_INFO("%s", "here1");
    grid_map::GridMapRosConverter::fromMessage(*msg, gridMap);
    ROS_INFO("%s", "here2");

    // std::string layer("elevation");
    // std::cout<<msg->layers[0]<<std::endl;
    std::string layer = msg->layers[0];
    grid_map::GridMapRosConverter::toCvImage(gridMap, layer, "mono8", *cvImage);
    ROS_INFO("%s", "here3");

    sensor_msgs::Image ros_image;
    cvImage->toImageMsg(ros_image);
    ros_image.encoding = "mono8";
    ROS_INFO("%s", "here4");

    ros_image.header = std_msgs::Header();
    ros_image.header.stamp = ros::Time::now();
    hm_im_pub.publish(ros_image);
    ROS_INFO("%s", "here5");
  }

  void divideBoundingBoxIntoVoxels()
  {
    // row
    nr_ = floor((orig_bb_max_.x - orig_bb_min_.x) / leaf_size_[0]);
    leaf_size_[0] = (orig_bb_max_.x - orig_bb_min_.x) / nr_;
    // col
    nc_ = floor((orig_bb_max_.y - orig_bb_min_.y) / leaf_size_[1]);
    leaf_size_[1] = (orig_bb_max_.y - orig_bb_min_.y) / nc_;
    // level
    nl_ = floor((orig_bb_max_.z - orig_bb_min_.z) / leaf_size_[2]);
    leaf_size_[2] = (orig_bb_max_.z - orig_bb_min_.z) / nl_;
  }

  float calculateViewEntropy(const Eigen::Vector4f &origin, const PointCloud &cloud)
  {
    std::unordered_map<int, float> cell_occupancy_prob;
    for (PointCloud::iterator it = cloud.begin(); it != cloud.end(); it++)
    {
      // find out what grid coord it belongs in
      Eigen::Vector3i grid_coord = worldCoordToGridCoord(it->x, it->y, it->z);
      // convert this to an index
      int index = gridCoordToVoxelIndex(grid_coord);
      // see whether it's in the map, add if it isn't
      auto it_prob = cell_occupancy_prob.find(index);
      if (it_prob == cell_occupancy_prob.end()) // couldn't find it
      {
        cell_occupancy_prob.insert(std::make_pair<int, float>(index, prob)); // TODO include initial probability
      }
      else // found it
      {
        // float prob_current // TODO
      }
    }
    std::unordered_set<int> cell_visited;
    for (PointCloud::iterator it = cloud.begin(); it != cloud.end(); it++)
    {
      std::vector<Eigen::Vector3i> out_ray;
      Eigen::Vector3i target_voxel = worldCoordToGridCoord(it->x, it->y, it->z);
      Eigen::Vector4f direction;
      direction << it->x - origin[0], it->y - origin[1], it->z - origin[2], 0.0f;
      direction.normalize();
      rayTraversal(out_ray, target_voxel, origin, direction, t_min); // TODO t_min
      for (size_t i = 0; i < out_ray.size(); i++)
      {
        int index = gridCoordToVoxelIndex(out_ray[i]);
        auto it_cell = cell_visited.find(index);
        if (it_cell == cell_visited.end())
        {
          cell_visited.insert(index);
        }
      }
    }
  }

  float calculateEntropyAlongRay(const std::vector<Eigen::Vector3i> &ray)
  {
    
  }

  Eigen::Vector3i worldCoordToGridCoord(const float x, const float y, const float z)
  {
    Eigen::Vector3i p;
    int r, c, l;
    r = (x - orig_bb_min_.x) / leaf_size_[0];
    c = (y - orig_bb_min_.y) / leaf_size_[1];
    l = (z - orig_bb_min_.z) / leaf_size_[2];
    p << r, c, l;
    return p;
  }

  int worldCoordToVoxelIndex(const Eigen::Vector4f &coord)
  {
    int index = gridCoordToVoxelIndex(worldCoordToGridCoord(coord[0], coord[1], coord[2]));
    return index;
  }

  Eigen::Vector3i voxelIndexToGridCoord(const int index)
  {
    int r, c, l;
    int temp;
    l = floor(index / (nr_ * nc_));
    temp = index % (nr_ * nc_);
    c = temp % nc_;
    r = floor(temp % nc_);
    Eigen::Vector3i v;
    v << r, c, l;
    return v;
  }

  int gridCoordToVoxelIndex(const Eigen::Vector3i &coord)
  {
    return coord[2] * (nr_ * nc_) + coord[1] * nr_ + coord[0];
  }

  Eigen::Vector4f gridCoordToWorldCoord(const Eigen::Vector3i &grid_coord)
  {
    Eigen::Vector4f v;
    v[0] = (grid_coord(0) + 0.5f) * leaf_size_[0];
    v[1] = (grid_coord(1) + 0.5f) * leaf_size_[1];
    v[2] = (grid_coord(2) + 0.5f) * leaf_size_[2];
    return v;
  }

  Eigen::Vector4f voxelIndexToWorldCoord(const int index)
  {
    Eigen::Vector4f p = gridCoordToWorldCoord(voxelIndexToGridCoord(index));
    return p;
  }

  void rayTraversal(std::vector<Eigen::Vector3i> &out_ray,
                    const Eigen::Vector3i &target_voxel,
                    const Eigen::Vector4f &origin,
                    const Eigen::Vector4f &direction,
                    const float t_min)
  {
    // coordinate of the boundary of the voxel grid
    Eigen::Vector4f start = origin + t_min * direction;

    // i,j,k coordinate of the voxel were the ray enters the voxel grid
    Eigen::Vector3i ijk = worldCoordToGridCoord(start[0], start[1], start[2]);
    //Eigen::Vector3i ijk = this->getGridCoordinates (start_x, start_y, start_z);

    // steps in which direction we have to travel in the voxel grid
    int step_x, step_y, step_z;

    // centroid coordinate of the entry voxel
    Eigen::Vector4f voxel_max = gridCoordToWorldCoord(ijk);

    if (direction[0] >= 0)
    {
      voxel_max[0] += leaf_size_[0] * 0.5f;
      step_x = 1;
    }
    else
    {
      voxel_max[0] -= leaf_size_[0] * 0.5f;
      step_x = -1;
    }
    if (direction[1] >= 0)
    {
      voxel_max[1] += leaf_size_[1] * 0.5f;
      step_y = 1;
    }
    else
    {
      voxel_max[1] -= leaf_size_[1] * 0.5f;
      step_y = -1;
    }
    if (direction[2] >= 0)
    {
      voxel_max[2] += leaf_size_[2] * 0.5f;
      step_z = 1;
    }
    else
    {
      voxel_max[2] -= leaf_size_[2] * 0.5f;
      step_z = -1;
    }

    float t_max_x = t_min + (voxel_max[0] - start[0]) / direction[0];
    float t_max_y = t_min + (voxel_max[1] - start[1]) / direction[1];
    float t_max_z = t_min + (voxel_max[2] - start[2]) / direction[2];

    float t_delta_x = leaf_size_[0] / static_cast<float>(fabs(direction[0]));
    float t_delta_y = leaf_size_[1] / static_cast<float>(fabs(direction[1]));
    float t_delta_z = leaf_size_[2] / static_cast<float>(fabs(direction[2]));

    while ((ijk[0] < max_b_[0] + 1) && (ijk[0] >= min_b_[0]) &&
           (ijk[1] < max_b_[1] + 1) && (ijk[1] >= min_b_[1]) &&
           (ijk[2] < max_b_[2] + 1) && (ijk[2] >= min_b_[2]))
    {
      // add voxel to ray
      out_ray.push_back(ijk);

      // check if we reached target voxel
      if (ijk[0] == target_voxel[0] && ijk[1] == target_voxel[1] && ijk[2] == target_voxel[2])
        break;

      // estimate next voxel
      if (t_max_x <= t_max_y && t_max_x <= t_max_z)
      {
        t_max_x += t_delta_x;
        ijk[0] += step_x;
      }
      else if (t_max_y <= t_max_z && t_max_y <= t_max_x)
      {
        t_max_y += t_delta_y;
        ijk[1] += step_y;
      }
      else
      {
        t_max_z += t_delta_z;
        ijk[2] += step_z;
      }
    }
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "grasped_reconstruction");

  ros::NodeHandle n;
  ros::Rate loop_rate(10);
  GraspedReconstruction gr(n);
  while (ros::ok())
  {
    gr.make_combo();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}