#include <grasped_reconstruction/grasped_reconstruction.h>

const float PI_F = 3.14159265358979f;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef std::unordered_map<int, std::pair<pcl::PointXYZ, float>> IndexedPointsWithProb;
class GraspedReconstruction
{
public:
  GraspedReconstruction(ros::NodeHandle &n) : nh_(n), anytime_pc_(new sensor_msgs::PointCloud2)
  {
    nh_ = n;
    calc_observed_points_sub = nh_.subscribe("/camera/depth/points", 1, &GraspedReconstruction::calculateObservedPointsClbk, this);
    calc_unobserved_points_sub = nh_.subscribe("/camera/depth/points", 1, &GraspedReconstruction::calculateUnobservedPointsClbk, this);
    pc_anytime_sub = nh_.subscribe("/camera/depth/points", 1, &GraspedReconstruction::pcAnytimeClbk, this);
    gm_sub = nh_.subscribe("/elevation_mapping/elevation_map", 1, &GraspedReconstruction::gmClbk, this);
    // color_sub = nh_.subscribe("/camera/depth/points", 1, &GraspedReconstruction::colorClbk, this);
    save_eef_pose_sub = nh_.subscribe("/save_current_eef_pose", 1, &GraspedReconstruction::saveCurrentEefPoseClbk, this);
    occ_pub = n.advertise<sensor_msgs::PointCloud2>("occluded_voxels", 1);
    ch_pub = n.advertise<sensor_msgs::PointCloud2>("ch_filtered", 1);
    combo_pub = n.advertise<sensor_msgs::PointCloud2>("part_occ", 1);
    object_pub = n.advertise<sensor_msgs::PointCloud2>("segmented_object", 1);
    tabletop_pub = n.advertise<sensor_msgs::PointCloud2>("tabletop", 1);
    entropy_arrow_pub = n.advertise<visualization_msgs::MarkerArray>("entropy_arrows", 1);
    bb_pub = n.advertise<visualization_msgs::Marker>("bbox", 1);
    anytime_pub = n.advertise<sensor_msgs::PointCloud2>("anytime", 1);
    cf_pub = n.advertise<sensor_msgs::PointCloud2>("color_filtered", 1);
    ch_points_pub = n.advertise<visualization_msgs::MarkerArray>("ch_points", 1);
    image_transport::ImageTransport it(n);
    hm_im_pub = it.advertise("height_map_image", 1);

    calculate_nbv_service_ = nh_.advertiseService("calculate_nbv", &GraspedReconstruction::calculateNbv, this);
    capture_and_process_observation_service_ = nh_.advertiseService("capture_and_process_observation", &GraspedReconstruction::captureAndProcessObservation, this);

    probability_by_state_.insert(std::make_pair(VoxelState::OCCUPIED, P_OCC));
    probability_by_state_.insert(std::make_pair(VoxelState::FREE, P_FREE));
    probability_by_state_.insert(std::make_pair(VoxelState::UNOBSERVED, P_UNOBS));
    // probability_by_state_.insert(std::make_pair(VoxelState::GRASP_OCCLUDED, P_OCC));
  }

  ros::NodeHandle nh_;
  ros::Subscriber calc_observed_points_sub, gm_sub, calc_unobserved_points_sub, save_eef_pose_sub, pc_anytime_sub, color_sub;
  ros::Publisher coeff_pub, object_pub, tabletop_pub, bb_pub, cf_pub, occ_pub, combo_pub, entropy_arrow_pub, nbv_pub, anytime_pub, ch_points_pub, ch_pub;
  ros::ServiceServer calculate_nbv_service_, capture_and_process_observation_service_;
  image_transport::Publisher hm_im_pub;
  tf::TransformListener listener;
  tf::TransformBroadcaster broadcaster;
  std::unordered_map<std::string, tf::StampedTransform> eef_pose_keyframes;
  tf::StampedTransform objorig_T_w_, origbb_T_w_;
  tf2_ros::StaticTransformBroadcaster static_broadcaster;
  int rMax, rMin, gMax, gMin, bMax, bMin;
  PointCloud combo_orig, orig_observed_, orig_unobserved_, combo_curr;
  PointCloud curr_pc;
  bool orig_observed_set_ = false;
  bool orig_unobserved_set_ = false;
  int NUM_AZIMUTH_POINTS = 8;
  int NUM_ELEVATION_POINTS = 8;
  float VIEW_RADIUS = 0.2f;
  float TABLETOP_HEIGHT = 0.735f;
  float P_OCC = 0.99f;
  float P_UNOBS = 0.5f;
  float P_FREE = 0.01f;
  std::string object_frame_id_;
  bool vg_initialized_ = false;
  IndexedPointsWithProb ipp_;
  std::unordered_map<int, float> cell_occupancy_prob_;
  PointCloud occluding_finger_points_;
  std::vector<Eigen::Vector4f> nbv_origins_;
  std::vector<Eigen::Quaternionf> nbv_orientations_;
  sensor_msgs::PointCloud2Ptr anytime_pc_; //(new sensor_msgs::PointCloud2);

  enum VoxelState
  {
    OCCUPIED,
    FREE,
    UNOBSERVED,
  };

  int num_observed_clbks = 0;
  int num_unobserved_clbks = 0;

  std::unordered_map<int, float> probability_by_state_;

  // canonical bounding box
  pcl::PointXYZ orig_bb_min_, orig_bb_max_;
  std::vector<float> leaf_size_;
  int nr_, nc_, nl_;
  float LEAF_SIZE = 0.01f;
  int num_voxels_;

  // gripper config
  tf::StampedTransform pi_T_fbl_, th_T_fbl_, in_T_fbl_;
  float FINGER_SCALE_FACTOR = 1.2f;

  bool captureAndProcessObservation(grasped_reconstruction::CaptureAndProcessObservation::Request &req, grasped_reconstruction::CaptureAndProcessObservation::Request &res)
  {
    sensor_msgs::PointCloud2Ptr msg_transformed(new sensor_msgs::PointCloud2());
    std::string target_frame("world");
    pcl_ros::transformPointCloud(target_frame, *anytime_pc_, *msg_transformed, listener);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*msg_transformed, *cloud);

    pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_cond(new pcl::ConditionAnd<pcl::PointXYZRGB>());
    if (nh_.hasParam("/rMax"))
    {
      color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZRGB>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZRGB>("h", pcl::ComparisonOps::LT, rMax)));
    }
    if (nh_.hasParam("/rMin"))
    {
      color_cond->addComparison(pcl::PackedHSIComparison<pcl::PointXYZRGB>::Ptr(new pcl::PackedHSIComparison<pcl::PointXYZRGB>("h", pcl::ComparisonOps::GT, rMin)));
    }
    // build the filter
    pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem;
    condrem.setCondition(color_cond);
    condrem.setInputCloud(cloud);
    condrem.setKeepOrganized(false);

    // apply filter
    condrem.filter(*cloud);

    if (eef_pose_keyframes.find("present") != eef_pose_keyframes.end() && eef_pose_keyframes.find("grasp") != eef_pose_keyframes.end())
    {
      {
        sensor_msgs::PointCloud2Ptr output(new sensor_msgs::PointCloud2());
        Eigen::Matrix4f p_T_w, g_T_w, w_T_p, w_T_g;
        pcl_ros::transformAsMatrix(eef_pose_keyframes.find("present")->second, p_T_w);
        pcl_ros::transformAsMatrix(eef_pose_keyframes.find("grasp")->second, g_T_w);
        std::cout << "\n p_T_w:\n"
                  << (p_T_w).matrix() << std::endl;
        std::cout << "\n g_T_w:\n"
                  << (g_T_w).matrix() << std::endl;
        w_T_p = Eigen::Matrix4f::Identity();
        w_T_p.block(0, 0, 3, 3) = p_T_w.block(0, 0, 3, 3).transpose();
        w_T_p.block(0, 3, 3, 1) = -p_T_w.block(0, 0, 3, 3).transpose() * p_T_w.block(0, 3, 3, 1);
        std::cout << "\n w_T_p:\n"
                  << (w_T_p).matrix() << std::endl;
        std::cout << "\n homog matrix:\n"
                  << (w_T_p * g_T_w).matrix() << std::endl;

        for (auto it = cloud->begin(); it != cloud->end(); ++it)
        {
          Eigen::Matrix4f pt, tx;
          pt = Eigen::Matrix4f::Identity();
          tx = Eigen::Matrix4f::Identity();
          pt(0, 3) = it->x;
          pt(1, 3) = it->y;
          pt(2, 3) = it->z;
          tx = (g_T_w * w_T_p) * pt;
          it->x = tx(0, 3);
          it->y = tx(1, 3);
          it->z = tx(2, 3);
        }
        sensor_msgs::PointCloud2Ptr cloud_in_smpc2(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(*cloud, *cloud_in_smpc2);
        sensor_msgs::PointCloud2 cloud_out_smpc2;
        pcl_ros::transformPointCloud(Eigen::Matrix4f::Identity(), *cloud_in_smpc2, *output);
        output->header.frame_id = "world";
        anytime_pub.publish(*output);
      }
    }
  }

  void getParams()
  {
    nh_.getParam("/bMax", bMax);
    nh_.getParam("/gMax", gMax);
    nh_.getParam("/rMax", rMax);
    nh_.getParam("/bMin", bMin);
    nh_.getParam("/gMin", gMin);
    nh_.getParam("/rMin", rMin);
    nh_.getParam("/object_frame_id", object_frame_id_);
  }

  void setVoxelProbabilities()
  {
    appendAndIncludePointCloudProb(orig_observed_, VoxelState::OCCUPIED);
    appendAndIncludePointCloudProb(orig_unobserved_, VoxelState::UNOBSERVED);

    for (auto it = ipp_.begin(); it != ipp_.end(); it++)
    {
      // find out what grid coord it belongs in
      Eigen::Vector3i grid_coord = worldCoordToGridCoord(it->second.first.x, it->second.first.y, it->second.first.z);
      // convert this to an index
      int index = gridCoordToVoxelIndex(grid_coord);
      // std::cout << "Point #: " << it->first << " Grid Coord: " << grid_coord[0] << " " << grid_coord[1] << " " << grid_coord[2] << " Index: " << index << std::endl;
      // see whether it's in the map, add if it isn't
      auto it_prob = cell_occupancy_prob_.find(index);
      float prob = it->second.second;
      if (it_prob == cell_occupancy_prob_.end()) // couldn't find it
      {
        // std::cout << "Adding to cell_occupancy_prob: " << index << " " << prob << std::endl;
        cell_occupancy_prob_.insert(std::make_pair(index, prob)); // TODO include initial probability
      }
      else // found it, update the probability
      {
        // take the average for now, TODO make running average later!
        it_prob->second = 0.5f * (it_prob->second + prob);
        // std::cout << "Updating cell_occupancy_prob: " << index << " " << it_prob->second << std::endl;
      }
    }
    setRemainderAsFree();
  }

  void setRemainderAsFree()
  {
    float prob = probability_by_state_.find(VoxelState::FREE)->second;
    for (int i = 0; i < num_voxels_; i++)
    {
      auto it = cell_occupancy_prob_.find(i);
      if (it == cell_occupancy_prob_.end())
      {
        cell_occupancy_prob_.insert(std::make_pair(i, prob));
      }
    }
  }

  void saveInitialObjectPose()
  {
    std::cout << "Saving initial object pose" << std::endl;
    try
    {
      ros::Time now = ros::Time::now();
      listener.waitForTransform("/world", object_frame_id_,
                                now, ros::Duration(3.0));
      listener.lookupTransform("/world", object_frame_id_,
                               now, objorig_T_w_);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }
  }

  void saveInitialBoundingBox()
  {
    std::cout << "Saving initial bounding box" << std::endl;
    try
    {
      ros::Time now = ros::Time::now();
      listener.waitForTransform("/world", "/orig_bb",
                                now, ros::Duration(3.0));
      listener.lookupTransform("/world", "/orig_bb",
                               now, origbb_T_w_);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }
  }

  void initializeVoxelGrid()
  {
    std::cout << "Initializing voxel grid" << std::endl;
    PointCloud::Ptr cl(new PointCloud);
    *cl = orig_observed_;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cl);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cl);
    pcl::PointXYZ orig_bb_obs_min, orig_bb_obs_max;
    pcl::getMinMax3D(*cl, orig_bb_obs_min, orig_bb_obs_max);
    std::cout << "obs min/max: " << orig_bb_obs_min << " " << orig_bb_obs_max << std::endl;
    *cl = orig_unobserved_;
    sor.filter(*cl);
    pcl::PointXYZ orig_bb_unobs_min, orig_bb_unobs_max;
    pcl::getMinMax3D(*cl, orig_bb_unobs_min, orig_bb_unobs_max);
    std::cout << "unobs min/max: " << orig_bb_unobs_min << " " << orig_bb_unobs_max << std::endl;

    orig_bb_min_.x = std::min(orig_bb_unobs_min.x, orig_bb_obs_min.x);
    orig_bb_min_.y = std::min(orig_bb_unobs_min.y, orig_bb_obs_min.y);
    orig_bb_min_.z = std::min(orig_bb_unobs_min.z, orig_bb_obs_min.z);

    orig_bb_max_.x = std::max(orig_bb_unobs_max.x, orig_bb_obs_max.x);
    orig_bb_max_.y = std::max(orig_bb_unobs_max.y, orig_bb_obs_max.y);
    orig_bb_max_.z = std::max(orig_bb_unobs_max.z, orig_bb_obs_max.z);

    std::cout << "Bounding box min: " << orig_bb_min_.x << " " << orig_bb_min_.y << " " << orig_bb_min_.z << std::endl;
    std::cout << "Bounding box max: " << orig_bb_max_.x << " " << orig_bb_max_.y << " " << orig_bb_max_.z << std::endl;

    publishBoundingBoxMarker();
    divideBoundingBoxIntoVoxels();
    setVoxelProbabilities();
    vg_initialized_ = true;
  }

  bool calculateNbv(grasped_reconstruction::CalculateNbv::Request &req, grasped_reconstruction::CalculateNbv::Response &res)
  {
    std::vector<float> view_entropies;
    std::vector<float> best_view_entropies;
    float highest_entropy = 0.0f;
    geometry_msgs::PoseStamped best_eef_pose;
    std::set<int> finger_occluded_voxels;
    Eigen::Quaternionf best_view;
    for (auto it = req.eef_poses.poses.begin(); it != req.eef_poses.poses.end(); it++) // go through every candidate pose
    {
      geometry_msgs::PoseStamped ps;
      ps.pose = *it;
      ps.header.frame_id = "/world";
      getVoxelIdsOccludedByFingers(ps, finger_occluded_voxels);
      Eigen::Quaternionf best_view_per_pose = calculateNextBestView(finger_occluded_voxels, view_entropies);
      float max_entropy = *std::max_element(view_entropies.begin(), view_entropies.end());
      if (it == req.eef_poses.poses.begin())
        ROS_ASSERT(max_entropy > 0.0f);
      if (max_entropy > highest_entropy)
      {
        best_eef_pose = ps;
        best_view = best_view_per_pose;
        best_view_entropies = view_entropies;
        highest_entropy = max_entropy;
      }
      finger_occluded_voxels.clear();
      view_entropies.clear();
    }

    tf::Quaternion q(best_view.x(), best_view.y(), best_view.z(), best_view.w());
    tf::Transform t;
    t.setRotation(q);
    tf::Transform n_T_w(t * objorig_T_w_);

    publishEntropyArrowSphere(best_view_entropies);

    geometry_msgs::Transform tf;
    tf.translation.x = n_T_w.getOrigin().x();
    tf.translation.y = n_T_w.getOrigin().y();
    tf.translation.z = n_T_w.getOrigin().z();
    tf.rotation.x = n_T_w.getRotation()[0];
    tf.rotation.y = n_T_w.getRotation()[1];
    tf.rotation.z = n_T_w.getRotation()[2];
    tf.rotation.w = n_T_w.getRotation()[3];
    res.nbv = tf;
    res.eef_pose = best_eef_pose;
  }

  void publishEntropyArrowSphere(std::vector<float> view_entropies)
  {
    visualization_msgs::MarkerArray ma;
    float entropy = *std::max_element(view_entropies.begin(), view_entropies.end());
    for (int i = 0; i < nbv_origins_.size(); i++)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = ros::Time();
      marker.id = i;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = nbv_origins_.at(i)[0];
      marker.pose.position.y = nbv_origins_.at(i)[1];
      marker.pose.position.z = nbv_origins_.at(i)[2];
      // direction between view origin and object
      marker.pose.orientation.x = nbv_orientations_[i].x();
      marker.pose.orientation.y = nbv_orientations_[i].y();
      marker.pose.orientation.z = nbv_orientations_[i].z();
      marker.pose.orientation.w = nbv_orientations_[i].w();
      marker.scale.x = view_entropies[i] / entropy * 0.1;
      marker.scale.y = 0.01;
      marker.scale.z = 0.01;
      marker.color.a = 1.0; // Don't forget to set the alpha!
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      ma.markers.push_back(marker);
    }
    entropy_arrow_pub.publish(ma);
  }

  void saveOpenGripperConfiguration()
  {
    std::cout << "Saving open gripper configuration" << std::endl;
    try
    {
      ros::Time now = ros::Time::now();
      listener.waitForTransform("/jaco_fingers_base_link", "/jaco_9_finger_thumb_tip",
                                now, ros::Duration(3.0));
      listener.lookupTransform("/jaco_fingers_base_link", "/jaco_9_finger_thumb_tip",
                               now, th_T_fbl_);
      listener.waitForTransform("/jaco_fingers_base_link", "/jaco_9_finger_index_tip",
                                now, ros::Duration(3.0));
      listener.lookupTransform("/jaco_fingers_base_link", "/jaco_9_finger_index_tip",
                               now, in_T_fbl_);
      listener.waitForTransform("/jaco_fingers_base_link", "/jaco_9_finger_pinkie_tip",
                                now, ros::Duration(3.0));
      listener.lookupTransform("/jaco_fingers_base_link", "/jaco_9_finger_pinkie_tip",
                               now, pi_T_fbl_);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }

    occluding_finger_points_.push_back(pcl::PointXYZ(0.0f, 0.0f, 0.0f));
    occluding_finger_points_.push_back(pcl::PointXYZ(th_T_fbl_.getOrigin().x() * FINGER_SCALE_FACTOR, th_T_fbl_.getOrigin().y() * FINGER_SCALE_FACTOR, th_T_fbl_.getOrigin().z() * FINGER_SCALE_FACTOR));
    occluding_finger_points_.push_back(pcl::PointXYZ(in_T_fbl_.getOrigin().x() * FINGER_SCALE_FACTOR, in_T_fbl_.getOrigin().y() * FINGER_SCALE_FACTOR, in_T_fbl_.getOrigin().z() * FINGER_SCALE_FACTOR));
    occluding_finger_points_.push_back(pcl::PointXYZ(pi_T_fbl_.getOrigin().x() * FINGER_SCALE_FACTOR, pi_T_fbl_.getOrigin().y() * FINGER_SCALE_FACTOR, pi_T_fbl_.getOrigin().z() * FINGER_SCALE_FACTOR));
    for (const auto &pt : occluding_finger_points_)
    {
      std::cout << pt << std::endl;
    }
  }

  void getVoxelIdsOccludedByFingers(const geometry_msgs::PoseStamped &ps, std::set<int> finger_occluded_voxels)
  {
    // std::cout<<"VG size: "<<voxel_grid.size()<<std::endl;
    // https://github.com/PointCloudLibrary/pcl/issues/1657
    pcl::CropHull<pcl::PointXYZ> cropHullFilter;
    boost::shared_ptr<PointCloud> hullCloud(new PointCloud());
    *hullCloud = occluding_finger_points_;
    tf::Transform tf;
    tf.setOrigin(tf::Vector3(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z));
    tf.setRotation(tf::Quaternion(ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w));
    // std::cout << tf.getOrigin().getX() << " " << tf.getOrigin().getY() << " " << tf.getOrigin().getZ() << " " << std::endl;
    // std::cout << tf.getRotation()[0] << " " << tf.getRotation()[1] << " " << tf.getRotation()[2] << " " << tf.getRotation()[3] << std::endl;

    pcl_ros::transformPointCloud(*hullCloud, *hullCloud, tf);
    for (const auto &pt : *hullCloud)
    {
      std::cout << pt << std::endl;
    }
    boost::shared_ptr<PointCloud> hullPoints(new PointCloud());

    std::vector<pcl::Vertices> hullPolygons;
    publishConvexHullMarker(*hullCloud);
    // setup hull filter
    pcl::ConvexHull<pcl::PointXYZ> cHull;
    cHull.setInputCloud(hullCloud);
    cHull.reconstruct(*hullPoints, hullPolygons);
    for (const auto &p : *hullPoints)
    {
      std::cout << "hp: " << p << std::endl;
    }
    std::cout << "Created convex hull!" << std::endl;

    cropHullFilter.setHullIndices(hullPolygons);
    cropHullFilter.setHullCloud(hullPoints);
    cropHullFilter.setDim(3); // if you uncomment this, it will work
    cropHullFilter.setCropOutside(false);

    // create point cloud
    boost::shared_ptr<PointCloud> pc(new PointCloud());


    // a point inside the hull
    for (size_t i = 0; i < num_voxels_; ++i)
    {
      Eigen::Vector4f w = voxelIndexToWorldCoord(i);
      pc->push_back(pcl::PointXYZ(w[0], w[1], w[2]));
      // std::cout << w.matrix() << std::endl;
    }
    pc->header.frame_id = "world";
    // for (size_t i = 0; i < pc->size(); ++i)
    // {
    // }

    //filter points
    cropHullFilter.setInputCloud(pc);
    boost::shared_ptr<PointCloud> filtered(new PointCloud());
    cropHullFilter.filter(*filtered);
    std::cout << "Proportion occluded by fingers: " << float(filtered->size()) / float(num_voxels_) << std::endl;
    pcl::PCLPointCloud2 cloud_filtered2;
    pcl::toPCLPointCloud2(*pc, cloud_filtered2);
    sensor_msgs::PointCloud2Ptr output(new sensor_msgs::PointCloud2());
    pcl_conversions::fromPCL(cloud_filtered2, *output);
    output->header.frame_id = "world";
    // Publish the data
    ch_pub.publish(*output);
  }

  void publishConvexHullMarker(const PointCloud &cloud)
  {
    visualization_msgs::MarkerArray ma;
    int count = 0;
    for (auto it = cloud.begin(); it != cloud.end(); it++)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = ros::Time();
      marker.id = count;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = it->x;
      marker.pose.position.y = it->y;
      marker.pose.position.z = it->z;
      std::cout << *it << std::endl;
      // direction between view origin and object
      marker.scale.x = 0.01;
      marker.scale.y = 0.01;
      marker.scale.z = 0.01;
      marker.color.a = 1.0; // Don't forget to set the alpha!
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;
      ma.markers.push_back(marker);
      count++;
    }
    ch_points_pub.publish(ma);
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
    // tf::TransformStamped
    tf::Transform t;
    t.setOrigin(tf::Vector3(orig_bb_min_.x, orig_bb_min_.y, orig_bb_min_.z));
    t.setRotation(tf::Quaternion(0, 0, 0, 1));
    geometry_msgs::TransformStamped static_transformStamped;

    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = "world";
    static_transformStamped.child_frame_id = "orig_bb";
    static_transformStamped.transform.translation.x = t.getOrigin().x();
    static_transformStamped.transform.translation.y = t.getOrigin().y();
    static_transformStamped.transform.translation.z = t.getOrigin().z();
    static_transformStamped.transform.rotation.x = t.getRotation()[0];
    static_transformStamped.transform.rotation.y = t.getRotation()[1];
    static_transformStamped.transform.rotation.z = t.getRotation()[2];
    static_transformStamped.transform.rotation.w = t.getRotation()[3];
    static_broadcaster.sendTransform(static_transformStamped);
  }

  void saveCurrentEefPoseClbk(const std_msgs::String &msg)
  {
    std::string phase = msg.data;
    tf::StampedTransform bb_bl;

    try
    {
      ros::Time now = ros::Time::now();
      if (phase == "grasp")
      {
        geometry_msgs::TransformStamped static_transformStamped;
        listener.waitForTransform("/world", "/jaco_fingers_base_link",
                                  now, ros::Duration(3.0));
        listener.lookupTransform("/world", "/jaco_fingers_base_link",
                                 now, bb_bl);
        static_transformStamped.header.stamp = ros::Time::now();
        static_transformStamped.header.frame_id = "world";
        static_transformStamped.child_frame_id = "jaco_fingers_base_link_at_grasp";
        static_transformStamped.transform.translation.x = bb_bl.getOrigin().x();
        static_transformStamped.transform.translation.y = bb_bl.getOrigin().y();
        static_transformStamped.transform.translation.z = bb_bl.getOrigin().z();
        static_transformStamped.transform.rotation.x = bb_bl.getRotation()[0];
        static_transformStamped.transform.rotation.y = bb_bl.getRotation()[1];
        static_transformStamped.transform.rotation.z = bb_bl.getRotation()[2];
        static_transformStamped.transform.rotation.w = bb_bl.getRotation()[3];
        static_broadcaster.sendTransform(static_transformStamped);
      }
      else if (phase == "present")
      {
        listener.waitForTransform("/world", "/jaco_fingers_base_link",
                                  now, ros::Duration(3.0));
        listener.lookupTransform("/world", "/jaco_fingers_base_link",
                                 now, bb_bl);
        geometry_msgs::TransformStamped static_transformStamped;
        static_transformStamped.header.stamp = ros::Time::now();
        static_transformStamped.header.frame_id = "world";
        static_transformStamped.child_frame_id = "jaco_fingers_base_link_at_present";
        static_transformStamped.transform.translation.x = bb_bl.getOrigin().x();
        static_transformStamped.transform.translation.y = bb_bl.getOrigin().y();
        static_transformStamped.transform.translation.z = bb_bl.getOrigin().z();
        static_transformStamped.transform.rotation.x = bb_bl.getRotation()[0];
        static_transformStamped.transform.rotation.y = bb_bl.getRotation()[1];
        static_transformStamped.transform.rotation.z = bb_bl.getRotation()[2];
        static_transformStamped.transform.rotation.w = bb_bl.getRotation()[3];
        static_broadcaster.sendTransform(static_transformStamped);
      }
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }
    auto it = eef_pose_keyframes.find(phase);
    if (it == eef_pose_keyframes.end())
    {
      eef_pose_keyframes.insert(std::make_pair(phase, bb_bl));
    }
    else
    {
      it->second = bb_bl;
    }
    for (const auto &e : eef_pose_keyframes)
    {
      std::cout << e.first << std::endl;
      std::cout << e.second.getOrigin().getX() << " " << e.second.getOrigin().getY() << " " << e.second.getOrigin().getZ() << " " << std::endl;
      std::cout << e.second.getRotation()[0] << " " << e.second.getRotation()[1] << " " << e.second.getRotation()[2] << " " << e.second.getRotation()[3] << std::endl;
      // std::cout<<e.second.getRotation()<<std::endl;
    }
  }

  void calculateObservedPointsClbk(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    if (!orig_observed_set_)
    {
      try
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
        pass.setFilterLimits(0, 0.4);
        pass.setFilterLimitsNegative(false); // allow to pass what is inside of this range
        pass.filter(*cloud);

        pass.setFilterFieldName("y");
        pass.setFilterLimits(-0.4, 0.2);
        pass.setFilterLimitsNegative(false); // allow to pass what is inside of this range
        pass.filter(*cloud);

        // Downsample this pc
        pcl::VoxelGrid<pcl::PointXYZ> downsample;
        downsample.setInputCloud(cloud);
        downsample.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
        // downsample.filter(*cloud);

        sensor_msgs::PointCloud2 cloud_cropped_with_partial_tabletop;
        pcl::toROSMsg(*cloud, cloud_cropped_with_partial_tabletop);
        tabletop_pub.publish(cloud_cropped_with_partial_tabletop); // publish the object and a bit of the tabletop to assist height map
        std::cout << "Published cloud cropped with partial tabletop" << std::endl;

        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.5, TABLETOP_HEIGHT);
        pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
        pass.filter(*cloud);

        orig_observed_ = *cloud;
        num_observed_clbks++;
        if (num_observed_clbks == 2)
        {
          orig_observed_set_ = true; // set to TRUE to stop it from going around again!
        }
        sensor_msgs::PointCloud2 cloud_cropped_with_no_tabletop;
        pcl::toROSMsg(*cloud, cloud_cropped_with_no_tabletop);
        object_pub.publish(cloud_cropped_with_no_tabletop);
        std::cout << "Published cloud cropped with no tabletop" << std::endl;
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("%s", ex.what());
      }
    }
  }

  void pcAnytimeClbk(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    *anytime_pc_ = *msg;
  }

  void calculateUnobservedPointsClbk(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    if (!orig_unobserved_set_)
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
      pass.setFilterLimits(-0.5, TABLETOP_HEIGHT);
      pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
      // pass.filter(*cloud);

      pass.setFilterFieldName("x");
      pass.setFilterLimits(-0.1, 0.3);
      pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
      // pass.filter(*cloud);

      pass.setFilterFieldName("y");
      pass.setFilterLimits(-0.2, 0.2);
      pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
      // pass.filter(*cloud);
      // std::cout << "Removed floor" << std::endl;

      // Downsample this pc
      pcl::VoxelGrid<pcl::PointXYZ> downsample;
      downsample.setInputCloud(cloud);
      downsample.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
      // downsample.filter(*cloud);

      pcl::PCLPointCloud2 cloud_filtered2;

      std::cout << "here1" << std::endl;
      cloud->header.frame_id = "/world";
      PointCloud::Ptr cloud2(new PointCloud());
      cloud2->header.frame_id = lens_frame;

      pcl_ros::transformPointCloud(lens_frame, *cloud, *cloud2, listener);
      std::cout << "here2" << std::endl;

      pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

      occ.setInputCloud(cloud2);
      occ.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
      occ.initializeVoxelGrid();

      Eigen::Vector3i box = occ.getMaxBoxCoordinates();
      std::cout << "box in unobs: \n"
                << box.matrix() << std::endl;
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
      PointCloud::Ptr cloud_occluded_world(new PointCloud);
      cloud_occluded_world->header.frame_id = "world";
      std::cout << "here3" << std::endl;
      pcl_ros::transformPointCloud(target_frame, *cloud_occluded, *cloud_occluded_world, listener);
      std::cout << "here4" << std::endl;

      pass.setInputCloud(cloud_occluded_world);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(-0.5, TABLETOP_HEIGHT);
      pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
      pass.filter(*cloud_occluded_world);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(0.0, 0.4);
      pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
      // pass.filter(*cloud_occluded_world);
      pass.setFilterFieldName("y");
      pass.setFilterLimits(-0.2, 0.2);
      pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
      pass.filter(*cloud_occluded_world);
      orig_unobserved_ = *cloud_occluded_world;
      num_unobserved_clbks++;
      if (num_unobserved_clbks == 2)
      {
        orig_unobserved_set_ = true; // set to TRUE to stop it from going around again!
      }
      pcl::toPCLPointCloud2(*cloud_occluded_world, cloud_filtered2);

      sensor_msgs::PointCloud2Ptr output(new sensor_msgs::PointCloud2());

      pcl_conversions::fromPCL(cloud_filtered2, *output);
      output->header.frame_id = "world";
      std::cout << "here5" << std::endl;

      // Publish the data
      occ_pub.publish(*output);
    }
    /*if (num_unobserved_clbks <3)
    {
      try
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
        pass.setFilterLimits(-0.5, TABLETOP_HEIGHT);
        pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
        pass.filter(*cloud);

        pass.setFilterFieldName("x");
        pass.setFilterLimits(0, 0.4);
        pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
        pass.filter(*cloud);

        pass.setFilterFieldName("y");
        pass.setFilterLimits(-0.2, 0.2);
        pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
        pass.filter(*cloud);
        // std::cout << "Removed floor" << std::endl;

        // Downsample this pc
        pcl::VoxelGrid<pcl::PointXYZ> downsample;
        downsample.setInputCloud(cloud);
        downsample.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
        // downsample.filter(*cloud);

        pcl::PCLPointCloud2 cloud_filtered2;

        std::cout << "here1" << std::endl;
        // ros::Duration(1.0).sleep();
        cloud->header.frame_id = lens_frame;
        pcl_ros::transformPointCloud(lens_frame, *cloud, *cloud, listener);
        std::cout << "here2" << std::endl;

        pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

        occ.setInputCloud(cloud);
        std::cout << "here2a" << std::endl;
        occ.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
        std::cout << "here2b" << std::endl;
        occ.initializeVoxelGrid();
        std::cout << "here2c" << std::endl;

        Eigen::Vector3i box = occ.getMaxBoxCoordinates();
        std::cout << "box in unobs: \n"
                  << box.matrix() << std::endl;
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
        PointCloud::Ptr cloud_occluded_world(new PointCloud);
        cloud_occluded_world->header.frame_id = "world";
        std::cout << "here3" << std::endl;
        pcl_ros::transformPointCloud(target_frame, *cloud_occluded, *cloud_occluded_world, listener);
        std::cout << "here4" << std::endl;

        pass.setInputCloud(cloud_occluded_world);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.5, TABLETOP_HEIGHT);
        pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
        // pass.filter(*cloud_occluded_world);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(0.0, 0.4);
        pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-0.2, 0.2);
        pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
        // pass.filter(*cloud_occluded_world);
        orig_unobserved_ = *cloud_occluded_world;
        num_unobserved_clbks++;

        pcl::toPCLPointCloud2(*cloud_occluded_world, cloud_filtered2);

        sensor_msgs::PointCloud2Ptr output(new sensor_msgs::PointCloud2());

        pcl_conversions::fromPCL(cloud_filtered2, *output);
        output->header.frame_id = "world";
        std::cout << "here5" << std::endl;

        // Publish the data
        occ_pub.publish(*output);
      }
      catch (tf::TransformException ex)
      {
        ROS_ERROR("%s", ex.what());
      }
    }*/
  }

  void gmClbk(const grid_map_msgs::GridMap::ConstPtr &msg)
  {
    cv_bridge::CvImagePtr cvImage(new cv_bridge::CvImage);
    // cv_bridge::CvImagePtr cvImageObservedMask(new cv_bridge::CvImage);
    grid_map::GridMap gridMap;
    // grid_map::GridMap gridMapMask;
    grid_map::GridMapRosConverter::fromMessage(*msg, gridMap);
    // grid_map::GridMapRosConverter::fromMessage(*msg, gridMapMask);

    // for (size_t i = 0; i < msg->data[0].data.size(); ++i)
    // {
    //   if (std::isnan(msg->data[0].data[i]))
    //   {
    //     std::cout << "nan!" << std::endl;
    //   }
    // }
    std::string layer = msg->layers[0];
    grid_map::GridMapRosConverter::toCvImage(gridMap, layer, "mono8", *cvImage);
    // grid_map::GridMapRosConverter::toCvImage(gridMapMask, layer, "mono8", *cvImageObservedMask);

    sensor_msgs::Image ros_image;
    // sensor_msgs::Image ros_image_observed_mask;
    cvImage->toImageMsg(ros_image);
    // cvImageObservedMask->toImageMsg(ros_image_observed_mask);
    ros_image.encoding = "mono8";
    // ros_image_observed_mask.encoding = "mono8";

    ros_image.header = std_msgs::Header();
    ros_image.header.stamp = ros::Time::now();
    // ros_image_observed_mask.header = std_msgs::Header();
    // ros_image_observed_mask.header.stamp = ros::Time::now();
    hm_im_pub.publish(ros_image);
    // hm_im_pub.publish(ros_image_observed_mask);
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
    std::cout << "ls0: " << leaf_size_[0] << " ls1: " << leaf_size_[1] << " ls2: " << leaf_size_[2] << std::endl;
    num_voxels_ = nr_ * nc_ * nl_; // (nr_ + 1) * (nc_ + 1) * (nl_ + 1);
  }

  void appendAndIncludePointCloudProb(const PointCloud &new_cloud, const int state)
  {
    float prob = probability_by_state_.find(state)->second;
    int index = ipp_.size();
    for (auto it = new_cloud.begin(); it != new_cloud.end(); it++, index++)
    {
      ipp_.insert(std::make_pair(index, std::make_pair(*it, prob))); // assumes uniform probability for a batch of points
    }
    std::cout << "Appended and included point cloud with probabilities!" << std::endl;
  }

  Eigen::Quaternionf calculateNextBestView(const std::set<int> &finger_occluded_voxels, std::vector<float> &view_entropies)
  {
    std::cout << "Beginning calculation of next best view!" << std::endl;
    Eigen::Vector4f best_view;
    best_view << 0.0f, 0.0f, 0.0f, 0.0f;
    float entropy = 0.0f;
    int best_view_id;
    int view_id = 0;
    for (const auto &v : nbv_origins_)
    {
      float e = calculateViewEntropy(v, finger_occluded_voxels);
      view_entropies.push_back(e);
      if (e > entropy)
      {
        best_view = v;
        entropy = e;
        best_view_id = view_id;
      }
      view_id++;
    }
    std::cout << "Origin: " << best_view[0] << " " << best_view[1] << " " << best_view[2] << " Entropy: " << entropy << std::endl;
    return nbv_orientations_[best_view_id];
  }

  float calculateViewEntropy(const Eigen::Vector4f &origin, const std::set<int> &finger_occluded_voxels)
  {
    float entropy = 0.0f;

    std::set<int> cell_visited;

    for (int i = 0; i < num_voxels_; i++) // for each point in voxel grid
    {
      std::vector<Eigen::Vector3i> out_ray;
      std::vector<Eigen::Vector3i> out_ray_unique;
      Eigen::Vector3i target_voxel = voxelIndexToGridCoord(i);
      Eigen::Vector4f target_voxel_w = voxelIndexToWorldCoord(i);
      Eigen::Vector4f direction;
      direction << target_voxel_w[0] - origin[0], target_voxel_w[1] - origin[1], target_voxel_w[2] - origin[2], 0.0f;
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
      entropy += calculateEntropyAlongRay(out_ray_unique, finger_occluded_voxels);
    }

    return entropy;
  }

  float calculateEntropyAlongRay(const std::vector<Eigen::Vector3i> &ray, const std::set<int> &finger_occluded_voxels) // TODO distance weighted
  {
    float entropy = 0.0f;
    for (const auto &v : ray)
    {
      int index = gridCoordToVoxelIndex(v);
      if (finger_occluded_voxels.find(index) != finger_occluded_voxels.end()) // if this voxel is hidden by the fingers
      {
        entropy += 0.0f; // don't learn anything from it
      }
      else
      {
        // std::cout << ">> along ray... Grid coord: " << v[0] << " " << v[1] << " " << v[2] << " Voxel index: " << index << std::endl;
        auto it_prob = cell_occupancy_prob_.find(gridCoordToVoxelIndex(v));
        ROS_ASSERT(it_prob != cell_occupancy_prob_.end());
        float p = it_prob->second;
        if (p > 0.6)
          break;
        entropy += -p * log(p) - (1.0f - p) * log(1.0f - p);
      }
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
    c = index % nc_;
    r = floor(temp / nc_);
    Eigen::Vector3i v;
    v << r, c, l;
    // std::cout<<"index: "<<index<<" r: "<<r<<" c: "<<c<<" l: "<<l<<std::endl;
    return v;
  }

  int gridCoordToVoxelIndex(const Eigen::Vector3i &coord)
  {
    return coord[2] * (nr_ * nc_) + coord[1] * nr_ + coord[0];
  }

  Eigen::Vector4f gridCoordToWorldCoord(const Eigen::Vector3i &grid_coord)
  {
    Eigen::Vector4f v;
    v[0] = (grid_coord(0) + 0.5f) * leaf_size_[0] + origbb_T_w_.getOrigin().getX();
    v[1] = (grid_coord(1) + 0.5f) * leaf_size_[1] + origbb_T_w_.getOrigin().getY();
    v[2] = (grid_coord(2) + 0.5f) * leaf_size_[2] + origbb_T_w_.getOrigin().getZ();
    v[3] = 0.0f;
    // std::cout << "grid coord: " << grid_coord(0) <<" "<<grid_coord(1)<<" "<<grid_coord(2)<< " world coord:" << v(0) <<" "<<v(1)<<" "<<v(2) << std::endl;
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

    // while ((ijk[0] < nr_ + 1) && (ijk[0] >= 0) && // ?????
    //        (ijk[1] < nc_ + 1) && (ijk[1] >= 0) &&
    //        (ijk[2] < nl_ + 1) && (ijk[2] >= 0))
    while ((ijk[0] < nr_ ) && (ijk[0] >= 0) && // ?????
           (ijk[1] < nc_ ) && (ijk[1] >= 0) &&
           (ijk[2] < nl_ ) && (ijk[2] >= 0))
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
    return tmin;
  }

  void generateViewCandidates()
  {
    std::cout << "Generating view candidates" << std::endl;
    float az_min = 0.0f + M_PI / 16;
    float az_max = 2.0f * M_PI + M_PI / 16;
    float el_min = -M_PI / 2.0f + M_PI / 16;
    float el_max = M_PI / 2.0f + M_PI / 16;
    float az_incr = (az_max - az_min) / NUM_AZIMUTH_POINTS;
    float el_incr = (el_max - el_min) / NUM_ELEVATION_POINTS;

    for (float az = az_min; az < az_max; az += az_incr)
    {
      for (float el = el_min; el < el_max; el += el_incr)
      {
        Eigen::Vector4f v;
        v[0] = VIEW_RADIUS * cos(az) * cos(el) + objorig_T_w_.getOrigin().getX();
        v[1] = VIEW_RADIUS * sin(az) * cos(el) + objorig_T_w_.getOrigin().getY();
        v[2] = VIEW_RADIUS * sin(el) + objorig_T_w_.getOrigin().getZ();
        v[3] = 0.0f;
        nbv_origins_.push_back(v);
        // calculate quaternion from view to origin
        // https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
        Eigen::Quaternionf q;
        Eigen::Vector3f axis;
        axis << -sin(az), cos(az), 0.0f;
        q = Eigen::AngleAxisf(-el, axis) * Eigen::AngleAxisf(az, Eigen::Vector3f::UnitZ());
        nbv_orientations_.push_back(q);
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
  gr.getParams();
  while (ros::ok())
  {
    if (!gr.vg_initialized_ && gr.orig_observed_set_ && gr.orig_unobserved_set_)
    {
      gr.saveInitialObjectPose();
      gr.saveOpenGripperConfiguration();
      gr.initializeVoxelGrid();
      gr.saveInitialBoundingBox();
      gr.generateViewCandidates();
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}