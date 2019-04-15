#include <grasped_reconstruction/grasped_reconstruction.h>

const float PI_F = 3.14159265358979f;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef std::unordered_map<int, std::pair<pcl::PointXYZ, float>> IndexedPointsWithProb;
class GraspedReconstruction
{
public:
  GraspedReconstruction(ros::NodeHandle &n) : _n(n)
  {
    _n = n;
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
    entropy_arrow_pub = n.advertise<visualization_msgs::MarkerArray>("entropy_arrows", 1);
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
  ros::Publisher coeff_pub, object_pub, tabletop_pub, bb_pub, cf_pub, occ_pub, combo_pub, entropy_arrow_pub;
  image_transport::Publisher hm_im_pub;
  tf::TransformListener listener;
  tf::StampedTransform world_T_lens_link_tf, world_T_object_tf;
  int rMax, rMin, gMax, gMin, bMax, bMin;
  PointCloud combo_orig, orig_observed, orig_unobserved;
  bool orig_observed_set = false;
  bool orig_unobserved_set = false;
  int NUM_AZIMUTH_POINTS = 6;
  int NUM_ELEVATION_POINTS = 6;
  float VIEW_RADIUS = 0.2f;
  float TABLETOP_HEIGHT = 0.735f;
  float P_OCC = 0.9f;
  float P_UNOBS = 0.5f;
  float P_FREE = 0.1f;
  bool combo_made = false;

  // canonical bounding box
  pcl::PointXYZ orig_bb_min_, orig_bb_max_;
  std::vector<float> leaf_size_;
  int nr_, nc_, nl_;
  float LEAF_SIZE = 0.01f;

  void make_combo()
  {
    // std::cout << orig_observed_set << orig_unobserved_set << combo_made << std::endl;
    if (orig_observed_set && orig_unobserved_set && !combo_made)
    {
      PointCloud::Ptr cl(new PointCloud);
      *cl = orig_observed;
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud(cl);
      sor.setMeanK(50);
      sor.setStddevMulThresh(1.0);
      sor.filter(*cl);
      pcl::PointXYZ orig_bb_obs_min, orig_bb_obs_max;
      pcl::getMinMax3D(*cl, orig_bb_obs_min, orig_bb_obs_max);
      std::cout<<"obs min/max: "<<orig_bb_obs_min<<" "<<orig_bb_obs_max<<std::endl;
      *cl = orig_unobserved;
      sor.filter(*cl);
      pcl::PointXYZ orig_bb_unobs_min, orig_bb_unobs_max;
      pcl::getMinMax3D(*cl, orig_bb_unobs_min, orig_bb_unobs_max);
      std::cout<<"unobs min/max: "<<orig_bb_unobs_min<<" "<<orig_bb_unobs_max<<std::endl;

      orig_bb_min_.x = std::min(orig_bb_unobs_min.x, orig_bb_obs_min.x);
      orig_bb_min_.y = std::min(orig_bb_unobs_min.y, orig_bb_obs_min.y);
      orig_bb_min_.z = std::min(orig_bb_unobs_min.z, orig_bb_obs_min.z);

      orig_bb_max_.x = std::max(orig_bb_unobs_max.x, orig_bb_obs_max.x);
      orig_bb_max_.y = std::max(orig_bb_unobs_max.y, orig_bb_obs_max.y);
      orig_bb_max_.z = std::max(orig_bb_unobs_max.z, orig_bb_obs_max.z);

      std::cout << "Got bounding box" << std::endl;

      IndexedPointsWithProb ipp;
      appendAndIncludePointCloudProb(orig_observed, P_OCC, ipp);
      appendAndIncludePointCloudProb(orig_unobserved, P_UNOBS, ipp);
      // for (auto it = ipp.begin(); it != ipp.end(); it++)
      // {
      //   std::cout << "Point #: " << it->first << " Coord: " << it->second.first.x << " " << it->second.first.y << " " << it->second.first.z << " Prob: " << it->second.second << std::endl;
      // }
      try
      {
        ros::Time now = ros::Time::now();
        listener.waitForTransform("/world", "/cube1",
                                  now, ros::Duration(3.0));
        listener.lookupTransform("/world", "/cube1",
                                 now, world_T_object_tf);
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("%s", ex.what());
      }
      std::cout << "world to object transform received" << std::endl;

      std::cout << "Bounding box min: " << orig_bb_min_.x << " " << orig_bb_min_.y << " " << orig_bb_min_.z << std::endl;
      std::cout << "Bounding box max: " << orig_bb_max_.x << " " << orig_bb_max_.y << " " << orig_bb_max_.z << std::endl;

      //publish
      sensor_msgs::PointCloud2 out;
      pcl::toROSMsg(*cl, out);
      combo_pub.publish(out);
      std::cout << "Publishing the combined observed and unobserved point cloud" << std::endl;

      publishBoundingBoxMarker();
      Eigen::Vector4f best_view = calculateNextBestView(ipp);
      // find transform to get to this view

      combo_made = true;
    }
  }

  void publishBoundingBoxMarker()
  {
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
    std::cout << "Published bounding box marker!" << std::endl;
  }

  void pcClbk(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    if (!orig_observed_set)
    {
      std::cout << "Processing point cloud callback" << std::endl;
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
      // std::cout << "Removed floor" << std::endl;

      // Downsample this pc
      pcl::VoxelGrid<pcl::PointXYZ> downsample;
      downsample.setInputCloud(cloud);
      downsample.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
      downsample.filter(*cloud);

      sensor_msgs::PointCloud2 tabletop_output;
      pcl::toROSMsg(*cloud, tabletop_output);
      tabletop_pub.publish(tabletop_output); // publish the object and a bit of the tabletop to assist height map
      std::cout << "Published object on table" << std::endl;

      pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(-0.5, TABLETOP_HEIGHT);
      pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
      pass.filter(*cloud);

      orig_observed = *cloud;
      orig_observed_set = true;
      sensor_msgs::PointCloud2 output;
      pcl::toROSMsg(*cloud, output);
      object_pub.publish(output);
      std::cout << "Published object with table removed" << std::endl;
    }
  }

  void occClbk(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    if (!orig_unobserved_set)
    {
      std::cout << "Processing occlusion callback" << std::endl;
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
      downsample.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
      downsample.filter(*cloud);

      pcl::PCLPointCloud2 cloud_filtered2;

      pcl_ros::transformPointCloud(lens_frame, *cloud, *cloud, listener);

      pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

      occ.setInputCloud(cloud);
      occ.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
      occ.initializeVoxelGrid();

      Eigen::Vector3i box = occ.getMaxBoxCoordinates();

      PointCloud cloud_filtered = occ.getFilteredPointCloud();
      std::vector<Eigen::Vector3i> occluded_voxels;
      occ.occlusionEstimationAll(occluded_voxels);
      // std::cout << "Proportion occluded: " << (float)occluded_voxels.size() / (float)(box(0) * box(1) * box(2)) << std::endl;
      PointCloud::Ptr cloud_occluded(new PointCloud);
      for (const auto &voxel : occluded_voxels)
      {
        Eigen::Vector4f coord = occ.getCentroidCoordinate(voxel);
        cloud_occluded->push_back(pcl::PointXYZ(coord(0), coord(1), coord(2)));
      }

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
      output->header.frame_id = "world";

      // Publish the data
      occ_pub.publish(*output);
    }
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
    std::cout << "Beginning to divide bounding box into voxels!" << std::endl;
    leaf_size_.resize(3);
    // row
    nr_ = floor((orig_bb_max_.x - orig_bb_min_.x) / LEAF_SIZE);
    leaf_size_[0] = (orig_bb_max_.x - orig_bb_min_.x) / nr_;
    // col
    nc_ = floor((orig_bb_max_.y - orig_bb_min_.y) / LEAF_SIZE);
    leaf_size_[1] = (orig_bb_max_.y - orig_bb_min_.y) / nc_;
    // level
    nl_ = floor((orig_bb_max_.z - orig_bb_min_.z) / LEAF_SIZE);
    leaf_size_[2] = (orig_bb_max_.z - orig_bb_min_.z) / nl_;
    std::cout << "Divided bounding box into voxels!" << std::endl;
    std::cout << "nr_: " << nr_ << " nc_: " << nc_ << " nl_: " << nl_ << std::endl;
  }

  void appendAndIncludePointCloudProb(const PointCloud &new_cloud, const float prob, IndexedPointsWithProb &ipp)
  {
    int index = ipp.size();
    for (auto it = new_cloud.begin(); it != new_cloud.end(); it++, index++)
    {
      ipp.insert(std::make_pair(index, std::make_pair(*it, prob))); // assumes uniform probability for a batch of points
    }
    std::cout << "Appended and included point cloud with probabilities!" << std::endl;
  }

  Eigen::Vector4f calculateNextBestView(const IndexedPointsWithProb &ipp)
  {
    std::cout << "Beginning calculation of next best view!" << std::endl;
    std::vector<float> view_entropy;
    std::vector<Eigen::Vector4f> views;
    std::vector<Eigen::Quaternionf> quats;
    Eigen::Vector4f best_view;
    best_view << 0.0f, 0.0f, 0.0f, 0.0f;
    float entropy = 0.0f;
    divideBoundingBoxIntoVoxels();
    generateViewCandidates(views, quats);
    for (const auto &v : views)
    {
      float e = calculateViewEntropy(v, ipp);
      view_entropy.push_back(e);
      std::cout << "Origin: " << v[0] << " " << v[1] << " " << v[2] << " Entropy: " << e << std::endl;
      if (e > entropy)
      {
        best_view = v;
        entropy = e;
      }
    }

    { // publish rviz arrows
      visualization_msgs::MarkerArray ma;
      for (int i = 0; i < views.size(); i++)
      {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time();
        marker.id = i;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = views.at(i)[0];
        marker.pose.position.y = views.at(i)[1];
        marker.pose.position.z = views.at(i)[2];
        // direction between view origin and object
        marker.pose.orientation.x = quats[i].x();
        marker.pose.orientation.y = quats[i].y();
        marker.pose.orientation.z = quats[i].z();
        marker.pose.orientation.w = quats[i].w();
        marker.scale.x = view_entropy[i]/entropy*0.1;
        marker.scale.y = 0.01;
        marker.scale.z = 0.01;
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        ma.markers.push_back(marker);
      }
      while (entropy_arrow_pub.getNumSubscribers()<1){
        std::cout<<"Waiting for arrow subscribers to connect..."<<std::endl;
        ros::Duration(1.0).sleep();
      }
      entropy_arrow_pub.publish(ma);
    }
    return best_view;
  }

  float calculateViewEntropy(const Eigen::Vector4f &origin, const IndexedPointsWithProb &ipp)
  {
    float entropy = 0.0f;
    std::unordered_map<int, float> cell_occupancy_prob;
    for (auto it = ipp.begin(); it != ipp.end(); it++)
    {
      // find out what grid coord it belongs in
      Eigen::Vector3i grid_coord = worldCoordToGridCoord(it->second.first.x, it->second.first.y, it->second.first.z);
      // convert this to an index
      int index = gridCoordToVoxelIndex(grid_coord);
      // std::cout << "Point #: " << it->first << " Grid Coord: " << grid_coord[0] << " " << grid_coord[1] << " " << grid_coord[2] << " Index: " << index << std::endl;
      // see whether it's in the map, add if it isn't
      auto it_prob = cell_occupancy_prob.find(index);
      float prob = it->second.second;
      if (it_prob == cell_occupancy_prob.end()) // couldn't find it
      {
        // std::cout << "Adding to cell_occupancy_prob: " << index << " " << prob << std::endl;
        cell_occupancy_prob.insert(std::make_pair(index, prob)); // TODO include initial probability
      }
      else // found it, update the probability
      {
        // take the average for now, TODO make running average later!
        it_prob->second = 0.5f * (it_prob->second + prob);
        // std::cout << "Updating cell_occupancy_prob: " << index << " " << it_prob->second << std::endl;
      }
    }
    // fill in the gaps!
    for (int i = 0; i < (nr_ + 1) * (nc_ + 1) * (nl_ + 1); i++)
    {
      auto it = cell_occupancy_prob.find(i);
      if (it == cell_occupancy_prob.end())
      {
        cell_occupancy_prob.insert(std::make_pair(i, P_FREE));
      }
    }
    std::set<int> cell_visited;
    for (auto it = ipp.begin(); it != ipp.end(); it++) // for each point in input cloud
    {
      std::vector<Eigen::Vector3i> out_ray;
      std::vector<Eigen::Vector3i> out_ray_unique;
      Eigen::Vector3i target_voxel = worldCoordToGridCoord(it->second.first.x, it->second.first.y, it->second.first.z);
      Eigen::Vector4f direction;
      direction << it->second.first.x - origin[0], it->second.first.y - origin[1], it->second.first.z - origin[2], 0.0f;
      direction.normalize();
      // std::cout << "Origin: " << origin[0] << " " << origin[1] << " " << origin[2] << " Direction: " << direction[0] << " " << direction[1] << " " << direction[2] << " Target Voxel: " << target_voxel[0] << " " << target_voxel[1] << " " << target_voxel[2] << std::endl;
      // std::cout << "Target: " << it->second.first.x << " " << it->second.first.y << " " << it->second.first.z << std::endl;

      rayTraversal(out_ray, target_voxel, origin, direction);
      for (size_t i = 0; i < out_ray.size(); i++) // for each voxel the ray passed through
      {
        int index = gridCoordToVoxelIndex(out_ray[i]);
        // std::cout << "Grid coord: " << out_ray[i][0] << " " << out_ray[i][1] << " " << out_ray[i][2] << " Voxel index: " << index << std::endl;
        auto it_cell = cell_visited.find(index);
        if (it_cell == cell_visited.end()) // if the voxel hasn't been included before
        {
          // std::cout << "Adding cell index to list: " << index << std::endl;
          cell_visited.insert(index);
          out_ray_unique.push_back(out_ray[i]);
        }
        else
        {
          // std::cout << "Not adding a repeat observation of voxel ID: " << index << std::endl;
        }
      }
      entropy += calculateEntropyAlongRay(out_ray_unique, cell_occupancy_prob);
    }
    return entropy;
  }

  float calculateEntropyAlongRay(const std::vector<Eigen::Vector3i> &ray, const std::unordered_map<int, float> &cell_occupancy_prob) // TODO distance weighted
  {
    float entropy = 0.0f;
    for (const auto &v : ray)
    {
      int index = gridCoordToVoxelIndex(v);
      // std::cout << ">> along ray... Grid coord: " << v[0] << " " << v[1] << " " << v[2] << " Voxel index: " << index << std::endl;
      auto it_prob = cell_occupancy_prob.find(gridCoordToVoxelIndex(v));
      ROS_ASSERT(it_prob != cell_occupancy_prob.end());
      float p = it_prob->second;
      entropy += -p * log(p) - (1.0f - p) * log(1.0f - p);
    }
    // std::cout << "Entropy from this ray cast: " << entropy << std::endl;
    return entropy;
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
    v[0] = (grid_coord(0) + 0.5f) * leaf_size_[0] + world_T_object_tf.getOrigin().getX();
    v[1] = (grid_coord(1) + 0.5f) * leaf_size_[1] + world_T_object_tf.getOrigin().getY();
    v[2] = (grid_coord(2) + 0.5f) * leaf_size_[2] + world_T_object_tf.getOrigin().getZ();
    v[3] = 0.0f;
    // std::cout << "grid coord: " << grid_coord.matrix() << " world coord:" << v.matrix() << std::endl;
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
                    const Eigen::Vector4f &direction)
  {
    // std::cout << "Beginning ray traversal!" << std::endl;
    float t_min = rayBoxIntersection(origin, direction);
    if (t_min < 0)
    {
      return;
    }
    // coordinate of the boundary of the voxel grid
    Eigen::Vector4f start = origin + t_min * direction;
    // std::cout << "Start world coord: " << start[0] << " " << start[1] << " " << start[2] << std::endl;

    // i,j,k coordinate of the voxel were the ray enters the voxel grid
    Eigen::Vector3i ijk = worldCoordToGridCoord(start[0], start[1], start[2]);
    // std::cout << "Entry voxel grid coord: " << ijk[0] << " " << ijk[1] << " " << ijk[2] << std::endl;

    // steps in which direction we have to travel in the voxel grid
    int step_x, step_y, step_z;

    // centroid coordinate of the entry voxel
    Eigen::Vector4f voxel_max = gridCoordToWorldCoord(ijk);
    // std::cout << "Entry voxel world coord: " << voxel_max[0] << " " << voxel_max[1] << " " << voxel_max[2] << std::endl;

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

    while ((ijk[0] < nr_ + 1) && (ijk[0] >= 0) &&
           (ijk[1] < nc_ + 1) && (ijk[1] >= 0) &&
           (ijk[2] < nl_ + 1) && (ijk[2] >= 0))
    {
      // add voxel to ray
      out_ray.push_back(ijk);
      Eigen::Vector4f wc = gridCoordToWorldCoord(ijk);
      // std::cout << "Saw voxel: " << ijk[0] << " " << ijk[1] << " " << ijk[2] << " at " << wc[0] << " " << wc[1] << " " << wc[2] << std::endl;
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

  float rayBoxIntersection(const Eigen::Vector4f &origin,
                           const Eigen::Vector4f &direction)
  {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    if (direction[0] >= 0)
    {
      tmin = (orig_bb_min_.x - origin[0]) / direction[0];
      tmax = (orig_bb_max_.x - origin[0]) / direction[0];
    }
    else
    {
      tmin = (orig_bb_max_.x - origin[0]) / direction[0];
      tmax = (orig_bb_min_.x - origin[0]) / direction[0];
    }

    if (direction[1] >= 0)
    {
      tymin = (orig_bb_min_.y - origin[1]) / direction[1];
      tymax = (orig_bb_max_.y - origin[1]) / direction[1];
    }
    else
    {
      tymin = (orig_bb_max_.y - origin[1]) / direction[1];
      tymax = (orig_bb_min_.y - origin[1]) / direction[1];
    }

    // std::cout << "tmin tmax tymin tymax: " << tmin << " " << tmax << " " << tymin << " " << tymax << std::endl;

    if ((tmin > tymax) || (tymin > tmax))
    {
      // PCL_ERROR("no intersection with the bounding box \n");
      tmin = -1.0f;
      return tmin;
    }

    if (tymin > tmin)
      tmin = tymin;
    if (tymax < tmax)
      tmax = tymax;

    if (direction[2] >= 0)
    {
      tzmin = (orig_bb_min_.z - origin[2]) / direction[2];
      tzmax = (orig_bb_max_.z - origin[2]) / direction[2];
    }
    else
    {
      tzmin = (orig_bb_max_.z - origin[2]) / direction[2];
      tzmax = (orig_bb_min_.z - origin[2]) / direction[2];
    }
    // std::cout << "tmin tmax tzmin tzmax: " << tmin << " " << tmax << " " << tzmin << " " << tzmax << std::endl;

    if ((tmin > tzmax) || (tzmin > tmax))
    {
      // PCL_ERROR("no intersection with the bounding box \n");
      tmin = -1.0f;
      return tmin;
    }

    if (tzmin > tmin)
      tmin = tzmin;
    if (tzmax < tmax)
      tmax = tzmax;
    // std::cout << "tmin: " << tmin << std::endl;
    return tmin;
  }

  void generateViewCandidates(std::vector<Eigen::Vector4f> &views, std::vector<Eigen::Quaternionf> &quats)
  {
    float az_min = 0.0f;
    float az_max = 2.0f * M_PI;
    float el_min = -M_PI / 2.0f;
    float el_max = M_PI / 2.0f;
    float az_incr = (az_max - az_min) / NUM_AZIMUTH_POINTS;
    float el_incr = (el_max - el_min) / NUM_ELEVATION_POINTS;
    // std::cout << "az_incr: " << az_incr << " el_incr: " << el_incr << std::endl;

    for (float az = az_min; az < az_max; az += az_incr)
    {
      for (float el = el_min; el < el_max; el += el_incr)
      {
        // std::cout << "az: " << az << " el: " << el << std::endl;
        Eigen::Vector4f v;
        v[0] = VIEW_RADIUS * cos(az) + world_T_object_tf.getOrigin().getX();
        v[1] = VIEW_RADIUS * sin(az) + world_T_object_tf.getOrigin().getY();
        v[2] = VIEW_RADIUS * sin(el) + world_T_object_tf.getOrigin().getZ();
        v[3] = 0.0f;
        views.push_back(v);
        // std::cout << "Candidate origin: " << v[0] << " " << v[1] << " " << v[2] << std::endl;
        // calculate quaternion from view to origin
        // https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
        Eigen::Quaternionf q;
        q = Eigen::AngleAxisf(el, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(az, Eigen::Vector3f::UnitZ());
        quats.push_back(q);
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