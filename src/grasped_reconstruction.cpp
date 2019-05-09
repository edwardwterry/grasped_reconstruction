#include <grasped_reconstruction/grasped_reconstruction.h>

const float PI_F = 3.14159265358979f;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef std::unordered_map<int, std::pair<pcl::PointXYZ, int>> IndexedPointsWithState;
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
    tabletop_sub = nh_.subscribe("/camera/depth/points", 1, &GraspedReconstruction::tabletopClbk, this);
    vol_gt_sub = nh_.subscribe("/gt_geom", 1, &GraspedReconstruction::setVolumetricGroundTruthClbk, this);
    // color_sub = nh_.subscribe("/camera/depth/points", 1,
    // &GraspedReconstruction::colorClbk, this);
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
    pc_by_category_pub = n.advertise<visualization_msgs::MarkerArray>("pc_by_category", 1);
    pc_occupied_pub = n.advertise<visualization_msgs::MarkerArray>("pc_occupied", 1);
    image_transport::ImageTransport it(n);
    hm_im_pub = it.advertise("height_map_image", 1);
    hm_vg_im_pub = it.advertise("height_map_vg_image", 1);
    hm_vg_mask_im_pub = it.advertise("height_map_vg_mask_image", 1);

    calculate_nbv_service_ = nh_.advertiseService("calculate_nbv", &GraspedReconstruction::calculateNbv, this);
    fingertips_in_collision_service_ = nh_.advertiseService("fingertips_in_collision", &GraspedReconstruction::fingertipsInCollisionWithObject, this);
    capture_and_process_observation_service_ = nh_.advertiseService("capture_and_process_observation", &GraspedReconstruction::captureAndProcessObservation, this);
    eval_gt_service_ = nh_.advertiseService("eval_gt", &GraspedReconstruction::evaluateOccupancyGridAgainstGroundTruth, this);

    probability_by_state_.insert(std::make_pair(Observation::OCCUPIED, P_OCC));
    probability_by_state_.insert(std::make_pair(Observation::FREE, P_FREE));
    probability_by_state_.insert(std::make_pair(Observation::UNOBSERVED, P_UNOBS));
    // probability_by_state_.insert(std::make_pair(Observation::GRASP_OCCLUDED,
    // P_OCC));
  }

  ros::NodeHandle nh_;
  ros::Subscriber calc_observed_points_sub, gm_sub, calc_unobserved_points_sub, save_eef_pose_sub, pc_anytime_sub, color_sub, tabletop_sub, vol_gt_sub;
  ros::Publisher coeff_pub, object_pub, tabletop_pub, bb_pub, cf_pub, occ_pub, combo_pub, entropy_arrow_pub, nbv_pub, anytime_pub, ch_points_pub, ch_pub, pc_by_category_pub, pc_occupied_pub;
  ros::ServiceServer calculate_nbv_service_, capture_and_process_observation_service_, eval_gt_service_, fingertips_in_collision_service_;
  image_transport::Publisher hm_im_pub, hm_vg_im_pub, hm_vg_mask_im_pub;
  tf::TransformListener listener;
  tf::TransformBroadcaster broadcaster;
  std::unordered_map<std::string, tf::StampedTransform> eef_pose_keyframes_;
  tf::StampedTransform origobj_T_w_, origbb_T_w_, l_T_w_;
  tf2_ros::StaticTransformBroadcaster static_broadcaster;
  int rMax, rMin, gMax, gMin, bMax, bMin;
  PointCloud combo_orig, orig_observed_, orig_unobserved_, combo_curr;
  PointCloud curr_pc;
  bool orig_observed_set_ = false;
  bool orig_unobserved_set_ = false;
  int NUM_AZIMUTH_POINTS = 8;
  int NUM_ELEVATION_POINTS = 6;
  float VIEW_RADIUS = 0.35f;
  float TABLETOP_HEIGHT = 0.735f;
  float P_OCC = 0.999f;
  float P_UNOBS = 0.5f;
  float P_FREE = 0.001f;
  std::string object_frame_id_;
  bool vg_initialized_ = false;
  IndexedPointsWithState ipp_;
  std::unordered_map<int, int> cell_occupancy_state_;
  std::unordered_map<int, int> cell_occupancy_state_gt_;
  PointCloud occluding_finger_points_;
  std::vector<Eigen::Vector4f> nbv_origins_;
  std::vector<Eigen::Quaternionf> nbv_orientations_;
  sensor_msgs::PointCloud2Ptr anytime_pc_; //(new sensor_msgs::PointCloud2);
  PointCloud bb_voxel_cloud_;
  float BB_SURFACE_MARGIN = 0.01; // [m]
  float hm_xmin_, hm_xmax_, hm_ymin_, hm_ymax_;
  int im_nr_, im_nc_;
  sensor_msgs::Image height_map_, height_map_mask_;
  std::vector<std::set<int>> finger_occluded_voxels_by_grasp_id_;
  std::vector<float> azimuths_, elevations_;

  enum Observation
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
  float LEAF_SIZE; // = 0.01f; // 0.01f
  int num_voxels_;

  // gripper config
  tf::StampedTransform pi_T_fbl_, th_T_fbl_, in_T_fbl_;
  float FINGER_SCALE_FACTOR = 1.3f;

  void publishHeightMapFromVoxelGrid()
  {
    float border = 0.2; // [m] in xy plane beyond bb extent
    get_hm_image_bounds(border);

    // https://stackoverflow.com/questions/20816955/how-to-set-all-pixels-of-an-opencv-mat-to-a-specific-value
    // height_map_ = cv::Mat(im_nr_, im_nc_, CV_32FC1, cv::Scalar(0.0f));
    // height_map_ = cv::Mat(im_nr_, im_nc_, CV_8UC1, cv::Scalar(0));
    // height_map_mask_ = cv::Mat(im_nr_, im_nc_, CV_8UC1, cv::Scalar(0));
    height_map_mask_.height = im_nr_;
    height_map_mask_.width = im_nc_;
    height_map_mask_.encoding = "mono8";
    height_map_mask_.step = im_nc_; // https://answers.ros.org/question/11312/what-is-image-step/

    height_map_.height = im_nr_;
    height_map_.width = im_nc_;
    height_map_.encoding = "mono8";
    height_map_.step = im_nc_; // https://answers.ros.org/question/11312/what-is-image-step/

    // std::cout << "hm size" << height_map_.rows << " " << height_map_.cols << " " << height_map_mask_.rows << " " << height_map_mask_.cols << std::endl;
    // std::cout<<height_map_mask<<std::endl;
    int dr, dc;
    std::vector<float> heights;
    hm_voxel_grid_rc_to_grid_offset(dr, dc);
    std::cout << "dr, dc: " << dr << " " << dc << std::endl;
    for (int r = 0; r < im_nr_; ++r)
    {
      for (int c = 0; c < im_nc_; ++c)
      {
        float x, y;
        hm_voxel_grid_rc_to_global(r, c, x, y);
        std::cout << "global pos at r c: " << r << " " << c << " " << x << " " << y << std::endl;
        // Eigen::Vector4f w;
        // w << x, y, 0.0f, 1.
        if (x < orig_bb_min_.x || x > orig_bb_max_.x || y < orig_bb_min_.y || y > orig_bb_max_.y)
        {
          height_map_mask_.data.push_back(0); // if outside bounds
          // height_map_.data.push_back(0.0f);   // if outside bounds
          heights.push_back(0.0f);
        }
        else
        {
          int l = nl_ - 1; // start at the top
          // grid coord to index
          int index = gridCoordToVoxelIndex(Eigen::Vector3i(r - dr, c - dc, l));
          std::cout << "index: " << index << std::endl;
          // get current occupancy state from it
          int state = cell_occupancy_state_[index];
          // std::cout << "here2b" << std::endl;
          int mask_val = 0;
          while (state == Observation::FREE && l > 0)
          {
            l--;
            index = gridCoordToVoxelIndex(Eigen::Vector3i(r - dr, c - dc, l));
            // std::cout << "here2c" << std::endl;
            state = cell_occupancy_state_[index];
            // std::cout << "here2d" << std::endl;
          }
          if (state == Observation::UNOBSERVED)
          {
            mask_val = 255;
            // std::cout << "here2e" << std::endl;
          }
          Eigen::Vector4f wc = gridCoordToWorldCoord(Eigen::Vector3i(r - dr, c - dc, l));
          std::cout << "r c l" << r - dr << " " << c - dc << " " << l << std::endl;
          // std::cout << "here2f" << wc.matrix() << std::endl;
          std::cout << "xyz: " << wc(0) << " " << wc(1) << " " << wc(2) << std::endl;
          // std::cout << "hm bounds: " << hm_xmin_ << " " << hm_xmax_ << " " << hm_ymin_ << " " << hm_ymax_ << " " << im_nr_ << " " << im_nc_ << std::endl;
          // hm_voxel_grid_global_to_rc(wc(0), wc(1), im_r, im_c);
          height_map_mask_.data.push_back(mask_val); // if outside bounds
          heights.push_back(wc(2));
          // static_cast<int>((wc(2)-TABLETOP_HEIGHT)*255);
        }
      }
    }

    float max_height = *std::max_element(heights.begin(), heights.end());
    float min_height = *std::min_element(heights.begin(), heights.end());
    for (const auto &h : heights)
    {
      height_map_.data.push_back(static_cast<int>((h - min_height - TABLETOP_HEIGHT)/(max_height - min_height - TABLETOP_HEIGHT)*255));
    }

    // for (int r = 0; r < nr_; ++r)
    // {
    //   for (int c = 0; c < nc_; ++c)
    //   {
    //     int l = nl_ - 1; // start at the top
    //     // grid coord to index
    //     int index = gridCoordToVoxelIndex(Eigen::Vector3i(r, c, l));
    //     // std::cout << "here2a" << std::endl;
    //     // get current occupancy state from it
    //     int state = cell_occupancy_state_[index];
    //     // std::cout << "here2b" << std::endl;
    //     int mask_val = 0;
    //     while (state == Observation::FREE && l > 0)
    //     {
    //       l--;
    //       index = gridCoordToVoxelIndex(Eigen::Vector3i(r, c, l));
    //       // std::cout << "here2c" << std::endl;
    //       state = cell_occupancy_state_[index];
    //       // std::cout << "here2d" << std::endl;
    //     }
    //     if (state == Observation::UNOBSERVED)
    //     {
    //       mask_val = 255;
    //       // std::cout << "here2e" << std::endl;
    //     }
    //     int im_r, im_c;
    //     Eigen::Vector4f wc = gridCoordToWorldCoord(Eigen::Vector3i(r, c, l));
    //     std::cout << "r c l" << r << " " << c << " " << l << std::endl;
    //     // std::cout << "here2f" << wc.matrix() << std::endl;
    //     std::cout << "xyz" << wc(0) << " " << wc(1) << " " << wc(2) << std::endl;
    //     // std::cout << "hm bounds: " << hm_xmin_ << " " << hm_xmax_ << " " << hm_ymin_ << " " << hm_ymax_ << " " << im_nr_ << " " << im_nc_ << std::endl;
    //     hm_voxel_grid_global_to_rc(wc(0), wc(1), im_r, im_c);
    //     // std::cout << wc(2) << std::endl;
    //     // std::cout<< height_map.at<float>(im_r, im_c)<<std::endl; // set it to the height of the highest free voxel
    //     // height_map_.at<int>(im_r, im_c) = static_cast<int>((wc(2)-TABLETOP_HEIGHT)*255); // set it to the height of the highest free voxel
    //     // std::cout << mask_val << std::endl;
    //     // height_map_mask_.at<int>(im_r, im_c) = mask_val; // if unobserved
    //     // std::cout << "here2i" << std::endl;
    //   }
    // }

    // for (int r = 0; r < im_nr_; ++r)
    // {
    //   for (int c = 0; c < im_nc_; ++c)
    //   {
    //     height_map_mask_.data.push_back(5 * r + c); // if unobserved
    //     // height_map_.at<int>(r, c) = r + c; // if unobserved
    //   }
    // } // std::cout << height_map_ << std::endl;
    hm_vg_im_pub.publish(height_map_);
    hm_vg_mask_im_pub.publish(height_map_mask_);
  }

  void get_hm_image_bounds(const float &border)
  {
    int num_border_padding_cells_x = border / leaf_size_[0];
    int num_border_padding_cells_y = border / leaf_size_[1];
    hm_xmin_ = orig_bb_min_.x - num_border_padding_cells_x * leaf_size_[0];
    hm_xmax_ = orig_bb_max_.x + num_border_padding_cells_x * leaf_size_[0];
    hm_ymin_ = orig_bb_min_.y - num_border_padding_cells_y * leaf_size_[1];
    hm_ymax_ = orig_bb_max_.y + num_border_padding_cells_y * leaf_size_[1];
    im_nr_ = (hm_xmax_ - hm_xmin_) / leaf_size_[0];
    im_nc_ = (hm_ymax_ - hm_ymin_) / leaf_size_[1];
    std::cout << "hm bounds: " << hm_xmin_ << " " << hm_xmax_ << " " << hm_ymin_ << " " << hm_ymax_ << " " << im_nr_ << " " << im_nc_ << std::endl;
  }

  void hm_voxel_grid_global_to_rc(const float &x, const float &y, int &im_r, int &im_c)
  {
    // std::cout << "hm bounds: " << x << " " << y << std::endl;
    // std::cout << "hm bounds: " << hm_xmin_ << " " << hm_xmax_ << " " << hm_ymin_ << " " << hm_ymax_ << std::endl;

    im_r = static_cast<int>((hm_xmax_ - x) / leaf_size_[0]);
    im_c = static_cast<int>((hm_ymax_ - y) / leaf_size_[1]);
    std::cout << "im_r, im_c " << im_r << " " << im_c << std::endl;
  }

  void hm_voxel_grid_rc_to_global(const int &r, const int &c, float &x, float &y)
  {
    x = hm_xmax_ - r * leaf_size_[0];
    y = hm_ymax_ - c * leaf_size_[1];
  }

  void hm_voxel_grid_rc_to_grid_offset(int &dr, int &dc)
  {
    dr = (hm_xmax_ - orig_bb_max_.x) / leaf_size_[0];
    dc = (hm_ymax_ - orig_bb_max_.y) / leaf_size_[1];
  }

  bool evaluateOccupancyGridAgainstGroundTruth(grasped_reconstruction::GTEval::Request &req, grasped_reconstruction::GTEval::Response &res)
  {
    std::cout << "Received service call!" << std::endl;
    Eigen::MatrixXi conf = Eigen::MatrixXi::Zero(3, 3);
    for (size_t i = 0; i < num_voxels_; ++i)
    {
      int gt_state = cell_occupancy_state_gt_.find(i)->second;
      int est_state = cell_occupancy_state_.find(i)->second;
      // std::cout<<"i ig est: "<<i<<" "<<gt_state<<" "<<est_state<<std::endl;
      conf(gt_state, est_state)++; // = conf(gt_state, est_state) + 1;
    }
    std::cout << "conf matrix: \n"
              << conf.matrix() << std::endl;
    res.GT_OCC_EST_OCC.data = conf(Observation::OCCUPIED, Observation::OCCUPIED);
    res.GT_OCC_EST_FREE.data = conf(Observation::OCCUPIED, Observation::FREE);
    res.GT_OCC_EST_UNOBS.data = conf(Observation::OCCUPIED, Observation::UNOBSERVED);
    res.GT_FREE_EST_OCC.data = conf(Observation::FREE, Observation::OCCUPIED);
    res.GT_FREE_EST_FREE.data = conf(Observation::FREE, Observation::FREE);
    res.GT_FREE_EST_UNOBS.data = conf(Observation::FREE, Observation::UNOBSERVED);
    res.GT_UNOBS_EST_OCC.data = conf(Observation::UNOBSERVED, Observation::OCCUPIED);
    res.GT_UNOBS_EST_FREE.data = conf(Observation::UNOBSERVED, Observation::FREE);
    res.GT_UNOBS_EST_UNOBS.data = conf(Observation::UNOBSERVED, Observation::UNOBSERVED);
    return true;
  }

  void setVolumetricGroundTruthClbk(const geometry_msgs::PoseArray &msg)
  {
    std::cout << "Received gt clbk message!" << std::endl;
    pcl::CropHull<pcl::PointXYZ> cropHullFilter;
    boost::shared_ptr<PointCloud> hullCloud(new PointCloud());
    for (const auto &p : msg.poses)
    {
      PointCloud pc_prim;
      for (float i = -0.5f; i < 1; i += 1.0f)
      {
        for (float j = -0.5f; j < 1; j += 1.0f)
        {
          for (float k = -0.5f; k < 1; k += 1.0f)
          {
            pc_prim.push_back(pcl::PointXYZ(origobj_T_w_.getOrigin().x() + p.position.x + i * p.orientation.x,
                                            origobj_T_w_.getOrigin().y() + p.position.y + j * p.orientation.y,
                                            origobj_T_w_.getOrigin().z() + p.position.z + k * p.orientation.z));
            std::cout << "new point: " << pc_prim.back() << std::endl;
          }
        }
      }
      *hullCloud = pc_prim;
      boost::shared_ptr<PointCloud> hullPoints(new PointCloud());
      std::vector<pcl::Vertices> hullPolygons;
      pcl::ConvexHull<pcl::PointXYZ> cHull;
      cHull.setInputCloud(hullCloud);
      cHull.reconstruct(*hullPoints, hullPolygons);
      cropHullFilter.setHullIndices(hullPolygons);
      cropHullFilter.setHullCloud(hullPoints);
      cropHullFilter.setDim(3);
      cropHullFilter.setCropOutside(true);
      boost::shared_ptr<PointCloud> pc(new PointCloud());

      // a point inside the hull
      for (size_t i = 0; i < num_voxels_; ++i)
      {
        Eigen::Vector4f w = voxelIndexToWorldCoord(i);
        pc->push_back(pcl::PointXYZ(w[0], w[1], w[2]));
      }
      //filter points
      cropHullFilter.setInputCloud(pc);
      boost::shared_ptr<PointCloud> filtered(new PointCloud());
      cropHullFilter.filter(*filtered);

      for (const auto &f : *filtered)
      {
        Eigen::Vector4f in_point;
        in_point << f.x, f.y, f.z, 1.0f;
        int index = worldCoordToVoxelIndex(in_point);
        cell_occupancy_state_gt_.insert(std::make_pair(index, Observation::OCCUPIED));
      }
    }
    for (size_t i = 0; i < num_voxels_; ++i)
    {
      if (cell_occupancy_state_gt_.find(i) == cell_occupancy_state_gt_.end())
      {
        cell_occupancy_state_gt_.insert(std::make_pair(i, Observation::FREE));
      }
    }
  }

  bool fingertipsInCollisionWithObject(grasped_reconstruction::FingertipsInCollision::Request &req, grasped_reconstruction::FingertipsInCollision::Response &res)
  {
    for (auto it = req.eef_poses.poses.begin(); it != req.eef_poses.poses.end(); it++) // go through every candidate pose
    {
      geometry_msgs::PoseStamped ps;
      ps.pose = *it;
      ps.header.frame_id = "/world";
      int in_collision = 0;
      tf::Transform tf;
      tf.setOrigin(tf::Vector3(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z));
      tf.setRotation(tf::Quaternion(ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w));
      boost::shared_ptr<PointCloud> hullCloud(new PointCloud());
      *hullCloud = occluding_finger_points_;
      pcl_ros::transformPointCloud(*hullCloud, *hullCloud, tf);
      for (const auto &p : *hullCloud)
      {
        Eigen::Vector4f w;
        w << p.x, p.y, p.z, 1.0f;
        // std::cout<<"w: "<<w[0]<<" "<<w[1]<<" "<<w[2]<<std::endl;
        int index = worldCoordToVoxelIndex(w);
        if (cell_occupancy_state_gt_.find(index) != cell_occupancy_state_gt_.end())
        {
          if (cell_occupancy_state_gt_.find(index)->second == Observation::OCCUPIED)
          {
            in_collision = 1;
          }
        }
      }
      res.in_collision.data.push_back(in_collision);
    }
    return true;
  }

  bool captureAndProcessObservation(grasped_reconstruction::CaptureAndProcessObservation::Request &req, grasped_reconstruction::CaptureAndProcessObservation::Response &res)
  {
    int grasp_id_selected = req.grasp_id.data;
    std::set<int> finger_occluded_voxels_at_selected_grasp_id = finger_occluded_voxels_by_grasp_id_[grasp_id_selected];
    sensor_msgs::PointCloud2Ptr msg_transformed(new sensor_msgs::PointCloud2());
    std::string target_frame("world");
    // std::cout << "here0" << std::endl;
    pcl_ros::transformPointCloud(target_frame, *anytime_pc_, *msg_transformed, listener);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*msg_transformed, *cloud);
    // std::cout << "here1" << std::endl;
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
    // std::cout << "here2" << std::endl;

    // apply filter
    condrem.filter(*cloud);

    if (eef_pose_keyframes_.find("present") != eef_pose_keyframes_.end() && eef_pose_keyframes_.find("grasp") != eef_pose_keyframes_.end())
    {
      // std::cout << "here3" << std::endl;

      {
        sensor_msgs::PointCloud2Ptr output(new sensor_msgs::PointCloud2());
        Eigen::Matrix4f p_T_w, g_T_w, w_T_p, w_T_g;
        pcl_ros::transformAsMatrix(eef_pose_keyframes_.find("present")->second, p_T_w);
        // std::cout << "here4" << std::endl;
        pcl_ros::transformAsMatrix(eef_pose_keyframes_.find("grasp")->second, g_T_w);
        // std::cout << "here5" << std::endl;
        // std::cout << "\n p_T_w:\n"
        //           << (p_T_w).matrix() << std::endl;
        // std::cout << "\n g_T_w:\n"
        //           << (g_T_w).matrix() << std::endl;
        w_T_g = Eigen::Matrix4f::Identity();
        w_T_g.block(0, 0, 3, 3) = g_T_w.block(0, 0, 3, 3).transpose();
        w_T_g.block(0, 3, 3, 1) = -g_T_w.block(0, 0, 3, 3).transpose() * g_T_w.block(0, 3, 3, 1);
        w_T_p = Eigen::Matrix4f::Identity();
        w_T_p.block(0, 0, 3, 3) = p_T_w.block(0, 0, 3, 3).transpose();
        w_T_p.block(0, 3, 3, 1) = -p_T_w.block(0, 0, 3, 3).transpose() * p_T_w.block(0, 3, 3, 1);
        // std::cout << "\n w_T_g:\n"
        //           << (w_T_g).matrix() << std::endl;
        // std::cout << "\n homog matrix:\n"
        //           << (w_T_g * p_T_w).matrix() << std::endl;
        // PointCloud bb_voxel_cloud_at_present = bb_voxel_cloud_;
        /*for (auto it = bb_voxel_cloud_at_present.begin(); it !=
          bb_voxel_cloud_at_present.end(); ++it) // build list of voxels to have
          rays cast into them
          {
          Eigen::Matrix4f pt, tx; pt = Eigen::Matrix4f::Identity(); tx =
          Eigen::Matrix4f::Identity(); pt(0, 3) = it->x; pt(1, 3) = it->y; pt(2,
          3) = it->z; tx = (p_T_w * w_T_g) * pt; it->x = tx(0, 3); it->y = tx(1,
          3); it->z = tx(2, 3);
        }*/
        // work out which boxes the observed points lie in
        std::set<int> voxel_ids_corresponding_to_observed_points;
        // std::set<int> voxel_ids_corresponding_to_unobserved_points;
        Eigen::Matrix4f T = g_T_w * w_T_p;
        // std::cout << "\n\n UPDATE OBSERVED POINTS \n\n"
        //           << std::endl;
        // first, take into account what you saw
        for (const auto &pt : *cloud)
        {
          getVoxelIdsOfPointsAtPresent(pt, T, voxel_ids_corresponding_to_observed_points);
        }
        for (const auto &v : voxel_ids_corresponding_to_observed_points)
        {
          updateVoxelProbability(v, Observation::OCCUPIED);
        }

        // move the ray shooting point to where the lens link would be wrt the orig_bb
        try
        {
          // std::cout << "here6" << std::endl;
          ros::Time now = ros::Time(0);
          listener.waitForTransform("/world", "/lens_link",
                                    ros::Time(0), ros::Duration(3.0));
          listener.lookupTransform("/world", "/lens_link",
                                   ros::Time(0), l_T_w_);
          // std::cout << "here7" << std::endl;
        }
        catch (tf::TransformException ex)
        {
          ROS_ERROR("%s", ex.what());
        }
        // camera wrt orig_bb
        Eigen::Vector4f lens_link_in_world;
        lens_link_in_world << l_T_w_.getOrigin().getX(), l_T_w_.getOrigin().getY(), l_T_w_.getOrigin().getZ(), 1.0f;
        // what is its orig box coordinate
        Eigen::Vector4f origin = T * lens_link_in_world;

        std::set<int> cell_visited;
        for (size_t i = 0; i < num_voxels_; i++) // for each point in voxel grid
        {
          // std::cout << "here8" << std::endl;
          std::vector<Eigen::Vector3i> out_ray;
          Eigen::Vector3i target_voxel = voxelIndexToGridCoord(i);
          Eigen::Vector4f target_voxel_w = voxelIndexToWorldCoord(i);
          Eigen::Vector4f direction;
          direction << target_voxel_w[0] - origin[0], target_voxel_w[1] - origin[1], target_voxel_w[2] - origin[2], 0.0f;
          direction.normalize();
          // std::cout << "Origin: " << origin[0] << " " << origin[1] << " " << origin[2] << " Direction: " << direction[0] << " " << direction[1] << " " << direction[2] << " Target Voxel : " << target_voxel[0] << " " << target_voxel[1] << " " << target_voxel[2] << std::endl;
          // std::cout << "Target: " << target_voxel_w[0] << " " << target_voxel_w[1] << " " << target_voxel_w[2] << std::endl;

          rayTraversal(out_ray, target_voxel, origin, direction);
          int default_state = Observation::FREE;
          bool occupied_voxel_has_been_passed = false;
          std::unordered_map<int, int> apriori_cell_occupancy_state = cell_occupancy_state_;
          for (size_t i = 0; i < out_ray.size(); i++) // for each voxel the ray passed through
          {
            int index = gridCoordToVoxelIndex(out_ray[i]);
            // std::cout << "Grid coord: " << out_ray[i][0] << " " << out_ray[i][1]
            //           << " " << out_ray[i][2] << " Voxel index: " << index << " Current state: " << apriori_cell_occupancy_state.find(index)->second << std::endl;
            auto it_cell = apriori_cell_occupancy_state.find(index);
            // auto it_visited = cell_visited.find(index);

            // if (it_visited == cell_visited.end()) // if the voxel hasn't been included before
            // {
            // std::cout << "Adding cell index to list: " << index << std::endl;
            // cell_visited.insert(index);
            if (occupied_voxel_has_been_passed)
            {
              if (it_cell->second != Observation::FREE && it_cell->second != Observation::OCCUPIED) // i.e, if it is UNOBSERVED
              {
                updateVoxelProbability(index, default_state);
              }
            }
            else
            {
              if (finger_occluded_voxels_at_selected_grasp_id.find(it_cell->first) == finger_occluded_voxels_at_selected_grasp_id.end())
              // if the voxel in question is NOT hidden by the fingers, an update is allowed to take place
              {
                if (it_cell->second == Observation::OCCUPIED)
                {
                  occupied_voxel_has_been_passed = true;
                  default_state = Observation::UNOBSERVED;
                  // std::cout << "Default state switched to UNOBS for voxel " << i << std::endl;
                }
                else
                {
                  updateVoxelProbability(index, default_state);
                }
              }
            }
          }
        }
      }
    }
    res.result.data = true;
    publishPointCloudByCategoryMarkerArray();
    publishPointCloudOccupiedMarkerArray();
    return true;
  }

  void publishPointCloudByCategoryMarkerArray()
  {
    visualization_msgs::MarkerArray ma;
    for (size_t i = 0; i < num_voxels_; ++i)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = ros::Time();
      marker.id = i;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      Eigen::Vector4f w = voxelIndexToWorldCoord(i);
      marker.pose.position.x = w[0];
      marker.pose.position.y = w[1];
      marker.pose.position.z = w[2];
      // direction between view origin and object
      marker.pose.orientation.x = 0.0f;
      marker.pose.orientation.y = 0.0f;
      marker.pose.orientation.z = 0.0f;
      marker.pose.orientation.w = 1.0f;
      marker.scale.x = LEAF_SIZE;
      marker.scale.y = LEAF_SIZE;
      marker.scale.z = LEAF_SIZE;
      int state = cell_occupancy_state_.find(i)->second;
      if (state == Observation::FREE)
      {
        marker.color.r = 1.0f;
        marker.color.g = 1.0f;
        marker.color.b = 1.0f;
      }
      else if (state == Observation::OCCUPIED)
      {
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
      }
      else
      {
        marker.color.r = 1.0f;
        marker.color.g = 0.0f;
        marker.color.b = 0.0f;
      }
      marker.color.a = 0.3; // Don't forget to set the alpha!
      // std::cout << "Index " << i << " with state " << state << " at: " << w[0] << " " << w[1] << " " << w[2] << std::endl;

      ma.markers.push_back(marker);
    }
    pc_by_category_pub.publish(ma);
  }

  void publishPointCloudOccupiedMarkerArray()
  {
    visualization_msgs::MarkerArray ma;
    for (size_t i = 0; i < num_voxels_; ++i)
    {
      int state = cell_occupancy_state_.find(i)->second;
      if (state == Observation::OCCUPIED)
      {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time();
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        Eigen::Vector4f w = voxelIndexToWorldCoord(i);
        marker.pose.position.x = w[0];
        marker.pose.position.y = w[1];
        marker.pose.position.z = w[2];
        // direction between view origin and object
        marker.pose.orientation.x = 0.0f;
        marker.pose.orientation.y = 0.0f;
        marker.pose.orientation.z = 0.0f;
        marker.pose.orientation.w = 1.0f;
        marker.scale.x = LEAF_SIZE;
        marker.scale.y = LEAF_SIZE;
        marker.scale.z = LEAF_SIZE;
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.5; // Don't forget to set the alpha!

        ma.markers.push_back(marker);
      }
    }
    pc_occupied_pub.publish(ma);
  }

  void getVoxelIdsOfPointsAtPresent(const pcl::PointXYZRGB &pt, const Eigen::Matrix4f &T, std::set<int> &map)
  {
    Eigen::Vector4f p_present;
    p_present << pt.x, pt.y, pt.z, 1.0f;
    // what is its orig box coordinate
    Eigen::Vector4f p_orig = T * p_present;
    // what is the voxel id corresponding to this point
    if (isPointInOrigBoundingBox(p_orig))
    {
      int index = worldCoordToVoxelIndex(p_orig);
      if (map.find(index) == map.end())
      {
        map.insert(index); // if not seen before
        // std::cout << "Add index: " << index << std::endl; // << " " << p_orig[1] << " " << p_orig[2] << " id: " << index << std::endl;
      }
      else
      {
        // std::cout << "Ignoring index: " << index << std::endl; // << " " << p_orig[1] << " " << p_orig[2] << " id: " << index << std::endl;
      }
    }
  }

  void getIdsOfOccludedVoxels(const pcl::PointCloud<pcl::PointXYZRGB> &c, const Eigen::Matrix4f &T, std::set<int> &map)
  {
    std::set<int> voxel_ids_corresponding_to_unobserved_points;
    pcl::VoxelGridOcclusionEstimation<pcl::PointXYZRGB> occ;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_occluded_world_pre(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_occluded_lens(new pcl::PointCloud<pcl::PointXYZRGB>);
    *cloud_occluded_world_pre = c;
    cloud_occluded_world_pre->header.frame_id = "world";
    cloud_occluded_lens->header.frame_id = "lens_link";
    pcl_ros::transformPointCloud("lens_link", *cloud_occluded_world_pre, *cloud_occluded_lens, listener);
    occ.setInputCloud(cloud_occluded_lens);
    occ.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    occ.initializeVoxelGrid();

    std::vector<Eigen::Vector3i> occluded_voxels;
    occ.occlusionEstimationAll(occluded_voxels);
    for (const auto &voxel : occluded_voxels)
    {
      Eigen::Vector4f coord = occ.getCentroidCoordinate(voxel); // in lens link
      cloud_occluded_lens->push_back(pcl::PointXYZRGB(coord(0), coord(1), coord(2)));
      std::cout << "Lens link point: " << coord(0) << " " << coord(1) << " " << coord(2) << std::endl;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_occluded_world(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl_ros::transformPointCloud("/world", *cloud_occluded_lens, *cloud_occluded_world, listener);
    for (const auto &pt : *cloud_occluded_world)
    {
      getVoxelIdsOfPointsAtPresent(pt, T, map);
    }
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud_occluded_lens, output);
    output.header.frame_id = "lens_link";
    anytime_pub.publish(output);
  }

  void updateVoxelProbability(const int voxel, const int observation)
  {
    auto it = cell_occupancy_state_.find(voxel);
    ROS_ASSERT(it != cell_occupancy_state_.end());
    // if (it->second != observation)
    // {
    //   std::cout << "Index " << voxel << " changed from " << it->second << " to " << observation << std::endl;
    // }
    // else
    // {
    //   std::cout << "Index " << voxel << " stayed at " << it->second << std::endl;
    // }
    it->second = observation;
  }

  bool isPointInOrigBoundingBox(const Eigen::Vector4f &p)
  {
    bool in_box = false;
    if (p(0) > orig_bb_min_.x && p(0) < orig_bb_max_.x && p(1) > orig_bb_min_.y && p(1) < orig_bb_max_.y && p(2) > orig_bb_min_.z && p(2) < orig_bb_max_.z)
    {
      in_box = true;
    }
    return in_box;
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
    nh_.getParam("/leaf_size", LEAF_SIZE);
  }

  void setInitialVoxelProbabilities()
  {
    for (const auto &pt : orig_observed_)
    {
      Eigen::Vector4f w;
      w << pt.x, pt.y, pt.z, 1.0f;
      int index = worldCoordToVoxelIndex(w);
      if (cell_occupancy_state_.find(index) == cell_occupancy_state_.end()) // not yet in set
      {
        cell_occupancy_state_.insert(std::make_pair(index, Observation::OCCUPIED));
        // std::cout << "Index " << index << " added as occupied at: " << w[0] << " " << w[1] << " " << w[2] << std::endl;
      }
      else
      {
        // std::cout << "Index " << index << " skipped as occupied at: " << w[0] << " " << w[1] << " " << w[2] << std::endl;
      } // if already in set, do nothing. ok because no other categories are in yet
    }

    for (const auto &pt : orig_unobserved_)
    {
      Eigen::Vector4f w;
      w << pt.x, pt.y, pt.z, 1.0f;
      int index = worldCoordToVoxelIndex(w);
      if (cell_occupancy_state_.find(index) == cell_occupancy_state_.end()) // not yet in set
      {
        cell_occupancy_state_.insert(std::make_pair(index, Observation::UNOBSERVED));
        // std::cout << "Index " << index << " added as unobserved at: " << w[0] << " " << w[1] << " " << w[2] << std::endl;
      }
      else // already in set
      {
        if (cell_occupancy_state_.find(index)->second == Observation::OCCUPIED)
        {
          // std::cout << "Index " << index << " changed from unobs to occ at: " << w[0] << " " << w[1] << " " << w[2] << std::endl;

          updateVoxelProbability(index, Observation::OCCUPIED); // if both occupied and unobserved in same voxel, prefer occupied
        }
        else
        {
          // std::cout << "Index " << index << " kept as unobs at: " << w[0] << " " << w[1] << " " << w[2] << std::endl;
          updateVoxelProbability(index, Observation::UNOBSERVED);
        }
      }
    }

    // fill in the blanks
    for (int i = 0; i < num_voxels_; i++)
    {
      Eigen::Vector4f w = voxelIndexToWorldCoord(i);
      if (cell_occupancy_state_.find(i) == cell_occupancy_state_.end()) // not yet in set
      {
        // std::cout << "Index " << i << " added as free at: " << w[0] << " " << w[1] << " " << w[2] << std::endl;
        cell_occupancy_state_.insert(std::make_pair(i, Observation::FREE));
      }
    }

    // std::cout << "cell occ state size: " << cell_occupancy_state_.size();
    // ROS_ASSERT(cell_occupancy_state_.size() == num_voxels_);
    // appendAndIncludePointCloudProb(orig_observed_, Observation::OCCUPIED);
    // appendAndIncludePointCloudProb(orig_unobserved_, Observation::UNOBSERVED);

    // for (auto it = ipp_.begin(); it != ipp_.end(); it++)
    // {
    //   // find out what grid coord it belongs in
    //   Eigen::Vector3i grid_coord = worldCoordToGridCoord(it->second.first.x, it->second.first.y, it->second.first.z);
    //   // convert this to an index
    //   int index = gridCoordToVoxelIndex(grid_coord);
    //   auto it_state = cell_occupancy_state_.find(index); // the record
    //   int state = it->second.second;                     // the current observation
    //   if (it_state == cell_occupancy_state_.end())       // couldn't find it
    //   {
    //     cell_occupancy_state_.insert(std::make_pair(index, state)); // TODO include initial probability
    //   }
    //   else // found it, update the state
    //   // if the prior observation was observed and current observation is unobserved, set it to the observed (occupied or free) one
    //   {
    //     /*if (it_state->second == Observation::FREE && state == Observation::UNOBSERVED)
    //     {
    //       it_state->second = Observation::FREE;
    //     }*/
    //     // No voxels are free yet, they will be below
    //     if (it_state->second == Observation::OCCUPIED && state == Observation::UNOBSERVED)
    //     {
    //       it_state->second = Observation::OCCUPIED;
    //     }
    //     // TODO include conflicting observations of free and occupied in the same box
    //     else
    //     {
    //       it_state->second = state;
    //     }
    //   }
    // }
  }

  void saveInitialObjectPose()
  {
    std::cout << "Saving initial object pose" << std::endl;
    try
    {
      ros::Time now = ros::Time(0);
      listener.waitForTransform("/world", object_frame_id_,
                                now, ros::Duration(3.0));
      listener.lookupTransform("/world", object_frame_id_,
                               now, origobj_T_w_);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }
    std::cout << "origobj xyz: " << origobj_T_w_.getOrigin().x() << " " << origobj_T_w_.getOrigin().y() << " " << origobj_T_w_.getOrigin().z() << std::endl;

    geometry_msgs::TransformStamped static_transformStamped;
    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = "world";
    static_transformStamped.child_frame_id = "orig_obj";
    static_transformStamped.transform.translation.x = origobj_T_w_.getOrigin().x();
    static_transformStamped.transform.translation.y = origobj_T_w_.getOrigin().y();
    static_transformStamped.transform.translation.z = origobj_T_w_.getOrigin().z();

    static_transformStamped.transform.rotation.x = origobj_T_w_.getRotation()[0];
    static_transformStamped.transform.rotation.y = origobj_T_w_.getRotation()[1];
    static_transformStamped.transform.rotation.z = origobj_T_w_.getRotation()[2];
    static_transformStamped.transform.rotation.w = origobj_T_w_.getRotation()[3];
    static_broadcaster.sendTransform(static_transformStamped);
  }

  void saveInitialBoundingBox()
  {
    std::cout << "Saving initial bounding box" << std::endl;
    try
    {
      ros::Time now = ros::Time(0);
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

    orig_bb_min_.x = std::min(orig_bb_unobs_min.x, orig_bb_obs_min.x) - BB_SURFACE_MARGIN;
    orig_bb_min_.y = std::min(orig_bb_unobs_min.y, orig_bb_obs_min.y) - BB_SURFACE_MARGIN;
    orig_bb_min_.z = std::min(orig_bb_unobs_min.z, orig_bb_obs_min.z) - BB_SURFACE_MARGIN;

    orig_bb_max_.x = std::max(orig_bb_unobs_max.x, orig_bb_obs_max.x) + BB_SURFACE_MARGIN;
    orig_bb_max_.y = std::max(orig_bb_unobs_max.y, orig_bb_obs_max.y) + BB_SURFACE_MARGIN;
    orig_bb_max_.z = std::max(orig_bb_unobs_max.z, orig_bb_obs_max.z) + BB_SURFACE_MARGIN;

    std::cout << "Bounding box min: " << orig_bb_min_.x << " " << orig_bb_min_.y << " " << orig_bb_min_.z << std::endl;
    std::cout << "Bounding box max: " << orig_bb_max_.x << " " << orig_bb_max_.y << " " << orig_bb_max_.z << std::endl;

    publishBoundingBoxMarker();
    saveInitialBoundingBox();
    ros::Duration(1.0).sleep();
    divideBoundingBoxIntoVoxels();
    setInitialVoxelProbabilities();
    publishHeightMapFromVoxelGrid();

    for (size_t i = 0; i < num_voxels_; ++i)
    {
      Eigen::Vector4f w = voxelIndexToWorldCoord(i);
      bb_voxel_cloud_.push_back(pcl::PointXYZ(w[0], w[1], w[2]));
    }

    vg_initialized_ = true;
  }

  bool calculateNbv(grasped_reconstruction::CalculateNbv::Request &req, grasped_reconstruction::CalculateNbv::Response &res)
  {
    finger_occluded_voxels_by_grasp_id_.clear();
    std::vector<float> view_entropies;
    std::vector<float> best_view_entropies;
    // std::vector<std::vector<float>> all_view_entropies;
    float highest_entropy = 0.0f;
    geometry_msgs::PoseStamped best_eef_pose;
    int best_view_id;
    std::set<int> finger_occluded_voxels;
    Eigen::Quaternionf best_view;
    int grasp_id_for_nbv = 0;
    int count = 0;
    for (auto it = req.eef_poses.poses.begin(); it != req.eef_poses.poses.end(); it++) // go through every candidate pose
    {
      geometry_msgs::PoseStamped ps;
      ps.pose = *it;
      ps.header.frame_id = "/world";
      getVoxelIdsOccludedByFingers(ps, finger_occluded_voxels);
      finger_occluded_voxels_by_grasp_id_.push_back(finger_occluded_voxels);
      int best_view_id_per_grasp_id;
      Eigen::Quaternionf best_view_per_grasp_id = calculateNextBestView(finger_occluded_voxels, view_entropies, best_view_id_per_grasp_id);
      // all_view_entropies.push_back(view_entropies);
      float max_entropy = *std::max_element(view_entropies.begin(), view_entropies.end());
      if (it == req.eef_poses.poses.begin())
        ROS_ASSERT(max_entropy > 0.0f);
      if (max_entropy > highest_entropy)
      {
        best_eef_pose = ps;
        best_view = best_view_per_grasp_id;
        best_view_id = best_view_id_per_grasp_id;
        grasp_id_for_nbv = count;
        std::cout << "best view id: " << best_view_id << " at grasp id: " << grasp_id_for_nbv << std::endl;
        best_view_entropies = view_entropies;
        highest_entropy = max_entropy;
      }
      finger_occluded_voxels.clear();
      view_entropies.clear();
      count++;
    }

    tf::Quaternion q(best_view.x(), best_view.y(), best_view.z(), best_view.w());
    tf::Transform t;
    t.setRotation(q);
    // tf::Transform n_T_w(t * origobj_T_w_);

    publishEntropyArrowSphere(best_view_entropies);

    geometry_msgs::Transform tf;
    tf.translation.x = nbv_origins_.at(best_view_id)[0];
    tf.translation.y = nbv_origins_.at(best_view_id)[1];
    tf.translation.z = nbv_origins_.at(best_view_id)[2];
    tf.rotation.x = t.getRotation()[0];
    tf.rotation.y = t.getRotation()[1];
    tf.rotation.z = t.getRotation()[2];
    tf.rotation.w = t.getRotation()[3];
    res.nbv = tf;
    // res.eef_pose = best_eef_pose.pose;
    res.selected_grasp_id.data = grasp_id_for_nbv;
    // this worked fine for a single NBV being returned

    geometry_msgs::TransformStamped static_transformStamped;

    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = "world";
    static_transformStamped.child_frame_id = "nbv";
    static_transformStamped.transform = tf;
    // orig
    // static_transformStamped.transform.translation.x = nbv_origins_.at(best_view_id)[0];
    // static_transformStamped.transform.translation.y = nbv_origins_.at(best_view_id)[1];
    // static_transformStamped.transform.translation.z = nbv_origins_.at(best_view_id)[2];
    // static_transformStamped.transform.rotation.x = best_view.x();
    // static_transformStamped.transform.rotation.y = best_view.y();
    // static_transformStamped.transform.rotation.z = best_view.z();
    // static_transformStamped.transform.rotation.w = best_view.w();
    static_broadcaster.sendTransform(static_transformStamped);

    // get the k best views across all grasps considered

    /*std::vector<float> all_view_entropies_row_vector;
    std::vector<std::pair<int, int>> top_ranked_nbv_ids_and_grasp_ids; // first is nbv id and second is grasp id
    for (size_t grasp_index = 0; grasp_index < all_view_entropies.size(); grasp_index++)
    {
      for (size_t view_index = 0; view_index < nbv_origins_.size(); view_index++)
      {
        all_view_entropies_row_vector.push_back(all_view_entropies[grasp_index][view_index]); // hope it's the right way around
        std::cout << "e at push: " << all_view_entropies[grasp_index][view_index] << std::endl;
      }
    }

    for (const auto &e : all_view_entropies_row_vector)
    {
      std::cout << "e: " << e << std::endl;
    }
    // https://stackoverflow.com/questions/14902876/indices-of-the-k-largest-elements-in-an-unsorted-length-n-array
    std::priority_queue<std::pair<float, std::pair<int, int>>> pq;
    for (int i = 0; i < all_view_entropies_row_vector.size(); ++i)
    {
      // std::cout << "i % nbv_orientations_.size(): " << i % nbv_orientations_.size() << std::endl;
      // std::cout << "i / nbv_orientations_.size(): " << i / nbv_orientations_.size() << std::endl;
      pq.push(std::make_pair(all_view_entropies_row_vector[i], std::make_pair(i / all_view_entropies.size(), i % all_view_entropies.size()))); // why backwards?!?!
    }
    int k = req.num_nbvs_to_request.data;
    for (int i = 0; i < k; ++i)
    {
      top_ranked_nbv_ids_and_grasp_ids.push_back(pq.top().second);
      std::cout << "Rank: " << i + 1 << " e: " << pq.top().first << " nbv index: " << pq.top().second.first << " grasp index: " << pq.top().second.second << std::endl;
      pq.pop();
    }

    geometry_msgs::PoseArray nbv_poses, eef_poses;
    std_msgs::Int32MultiArray selected_grasp_ids;
    nbv_poses.header.frame_id = "world";
    eef_poses.header.frame_id = "world";
    for (size_t i = 0; i < top_ranked_nbv_ids_and_grasp_ids.size(); ++i)
    {
      Eigen::Quaternionf nbv_orientation = nbv_orientations_[top_ranked_nbv_ids_and_grasp_ids[i].first];
      Eigen::Vector4f nbv_origin = nbv_origins_[top_ranked_nbv_ids_and_grasp_ids[i].first];
      int grasp_id = top_ranked_nbv_ids_and_grasp_ids[i].second;
      std::cout << "index: " << i << " grasp id: " << grasp_id << " nbv origin: \n"
                << nbv_origin.matrix() << std::endl;
      tf::Quaternion q2(nbv_orientation.x(), nbv_orientation.y(), nbv_orientation.z(), nbv_orientation.w());
      tf::Transform t2;
      t2.setRotation(q2);
      tf::Transform n_T_w2(t2 * origobj_T_w_);
      geometry_msgs::Pose nbv_pose;
      nbv_pose.position.x = n_T_w2.getOrigin().x();
      nbv_pose.position.y = n_T_w2.getOrigin().y();
      nbv_pose.position.z = n_T_w2.getOrigin().z();
      std::cout << "origobj_T_w_ xyz: " << origobj_T_w_.getOrigin().x() << " " << origobj_T_w_.getOrigin().y() << " " << origobj_T_w_.getOrigin().z() << std::endl;
      std::cout << "nbv xyz: " << nbv_pose.position.x << " " << nbv_pose.position.y << " " << nbv_pose.position.z << std::endl;
      nbv_pose.orientation.x = n_T_w2.getRotation()[0];
      nbv_pose.orientation.y = n_T_w2.getRotation()[1];
      nbv_pose.orientation.z = n_T_w2.getRotation()[2];
      nbv_pose.orientation.w = n_T_w2.getRotation()[3];
      nbv_poses.poses.push_back(nbv_pose);
      eef_poses.poses.push_back(req.eef_poses.poses[grasp_id]);
      selected_grasp_ids.data.push_back(grasp_id);
    }*/
    return true;
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
      ros::Time now = ros::Time(0);
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
    // for (const auto &pt : occluding_finger_points_)
    // {
    //   std::cout << pt << std::endl;
    // }
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
    // std::cout << tf.getOrigin().getX() << " " << tf.getOrigin().getY() << " "
    // << tf.getOrigin().getZ() << " " << std::endl; std::cout <<
    // tf.getRotation()[0] << " " << tf.getRotation()[1] << " " <<
    // tf.getRotation()[2] << " " << tf.getRotation()[3] << std::endl;

    pcl_ros::transformPointCloud(*hullCloud, *hullCloud, tf);
    // for (const auto &pt : *hullCloud)
    // {
    //   std::cout << pt << std::endl;
    // }
    boost::shared_ptr<PointCloud> hullPoints(new PointCloud());

    std::vector<pcl::Vertices> hullPolygons;
    publishConvexHullMarker(*hullCloud);
    // setup hull filter
    pcl::ConvexHull<pcl::PointXYZ> cHull;
    cHull.setInputCloud(hullCloud);
    cHull.reconstruct(*hullPoints, hullPolygons);
    // for (const auto &p : *hullPoints)
    // {
    //   std::cout << "hp: " << p << std::endl;
    // }
    std::cout << "Created convex hull!" << std::endl;

    cropHullFilter.setHullIndices(hullPolygons);
    cropHullFilter.setHullCloud(hullPoints);
    cropHullFilter.setDim(3); // if you uncomment this, it will work
    cropHullFilter.setCropOutside(true);

    // create point cloud
    boost::shared_ptr<PointCloud> pc(new PointCloud());

    // a point inside the hull
    for (size_t i = 0; i < num_voxels_; ++i)
    {
      Eigen::Vector4f w = voxelIndexToWorldCoord(i);
      pc->push_back(pcl::PointXYZ(w[0], w[1], w[2]));
      // std::cout << w.matrix() << std::endl;
    }
    // pc->header.frame_id = "world";

    //filter points
    cropHullFilter.setInputCloud(pc);
    boost::shared_ptr<PointCloud> filtered(new PointCloud());
    cropHullFilter.filter(*filtered);
    // get number of unobserved voxels
    int num_unobs_voxels = 0;
    for (const auto &v : cell_occupancy_state_)
    {
      if (v.second == Observation::UNOBSERVED)
      {
        num_unobs_voxels++;
      }
    }
    std::cout << "Proportion of unobserved occluded by fingers: " << float(filtered->size()) / float(num_unobs_voxels) << std::endl;
    pcl::PCLPointCloud2 cloud_filtered2;
    pcl::toPCLPointCloud2(*filtered, cloud_filtered2);
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

    tf::Transform t2;
    t2.setOrigin(tf::Vector3(orig_bb_max_.x, orig_bb_max_.y, orig_bb_max_.z));
    t2.setRotation(tf::Quaternion(0, 0, 0, 1));
    geometry_msgs::TransformStamped static_transformStamped2;

    static_transformStamped2.header.stamp = ros::Time::now();
    static_transformStamped2.header.frame_id = "world";
    static_transformStamped2.child_frame_id = "orig_bb_ctr";
    static_transformStamped2.transform.translation.x = 0.5 * (t.getOrigin().x() + t2.getOrigin().x());
    static_transformStamped2.transform.translation.y = 0.5 * (t.getOrigin().y() + t2.getOrigin().y());
    static_transformStamped2.transform.translation.z = 0.5 * (t.getOrigin().z() + t2.getOrigin().z());
    static_transformStamped2.transform.rotation.x = t.getRotation()[0];
    static_transformStamped2.transform.rotation.y = t.getRotation()[1];
    static_transformStamped2.transform.rotation.z = t.getRotation()[2];
    static_transformStamped2.transform.rotation.w = t.getRotation()[3];
    static_broadcaster.sendTransform(static_transformStamped2);
  }

  void saveCurrentEefPoseClbk(const std_msgs::String &msg)
  {
    std::string phase = msg.data;
    tf::StampedTransform bb_bl;
    std::cout << "here1" << std::endl;

    try
    {
      std::cout << "here2" << std::endl;
      ros::Time now = ros::Time(0);
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
        std::cout << "here3" << std::endl;
        listener.waitForTransform("/world", "/jaco_fingers_base_link",
                                  ros::Time(0), ros::Duration(3.0));
        listener.lookupTransform("/world", "/jaco_fingers_base_link",
                                 ros::Time(0), bb_bl);
        std::cout << "here4" << std::endl;
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
        std::cout << "made it here" << std::endl;
        static_broadcaster.sendTransform(static_transformStamped);
      }
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }
    auto it = eef_pose_keyframes_.find(phase);
    if (it == eef_pose_keyframes_.end())
    {
      eef_pose_keyframes_.insert(std::make_pair(phase, bb_bl));
      std::cout << "Saving eef pose clbk: " << phase << std::endl;
    }
    else
    {
      it->second = bb_bl;
    }
    std::cout << "done saving pose" << std::endl;
  }

  void tabletopClbk(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    // std::cout << "Preparing tabletop!" << std::endl;
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
    downsample.filter(*cloud);

    sensor_msgs::PointCloud2 cloud_cropped_with_partial_tabletop;
    pcl::toROSMsg(*cloud, cloud_cropped_with_partial_tabletop);
    tabletop_pub.publish(cloud_cropped_with_partial_tabletop); // publish the object and a bit of the tabletop to assist height map
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

        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.9, 10);
        pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
        // pass.filter(*cloud);

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
        // tabletop_pub.publish(cloud_cropped_with_partial_tabletop); // publish the object and a bit of the tabletop to assist height map
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

      pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(0.9, 10);
      pass.setFilterLimitsNegative(true); // allow to pass what is outside of this range
      pass.filter(*cloud);

      pass.setFilterFieldName("x");
      pass.setFilterLimits(-0.1, 0.3);
      pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
      // pass.filter(*cloud);

      pass.setFilterFieldName("y");
      pass.setFilterLimits(-0.2, 0.2);
      pass.setFilterLimitsNegative(false); // allow to pass what is outside of this range
      // pass.filter(*cloud); std::cout << "Removed floor" << std::endl;

      // Downsample this pc
      pcl::VoxelGrid<pcl::PointXYZ> downsample;
      downsample.setInputCloud(cloud);
      downsample.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
      // downsample.filter(*cloud);

      pcl::PCLPointCloud2 cloud_filtered2;

      // std::cout << "here1" << std::endl;
      cloud->header.frame_id = "/world";
      PointCloud::Ptr cloud2(new PointCloud());
      cloud2->header.frame_id = lens_frame;

      pcl_ros::transformPointCloud(lens_frame, *cloud, *cloud2, listener);
      // std::cout << "here2" << std::endl;

      pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

      occ.setInputCloud(cloud2);
      occ.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
      occ.initializeVoxelGrid();

      Eigen::Vector3i box = occ.getMaxBoxCoordinates();
      // std::cout << "box in unobs: \n"
      //           << box.matrix() << std::endl;
      PointCloud cloud_filtered = occ.getFilteredPointCloud();
      std::vector<Eigen::Vector3i> occluded_voxels;
      occ.occlusionEstimationAll(occluded_voxels);
      // std::cout << "Proportion occluded: " << (float)occluded_voxels.size() /
      // (float)(box(0) * box(1) * box(2)) << std::endl;
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
      // std::cout << "here3" << std::endl;
      pcl_ros::transformPointCloud(target_frame, *cloud_occluded, *cloud_occluded_world, listener);
      // std::cout << "here4" << std::endl;

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
      // std::cout << "here5" << std::endl;

      // Publish the data
      occ_pub.publish(*output);
    }
    /*if (num_unobserved_clbks <3)
      {
      try
      {
        std::cout << "Processing occlusion callback" << std::endl; //
        http://wiki.ros.org/pcl/Tutorials#pcl.2BAC8-Tutorials.2BAC8-hydro.sensor_msgs.2BAC8-PointCloud2
        // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
        sensor_msgs::PointCloud2Ptr msg_transformed(new
        sensor_msgs::PointCloud2()); std::string target_frame("world");
        std::string lens_frame("lens_link");
        pcl_ros::transformPointCloud(target_frame, *cloud_msg, *msg_transformed,
        listener); PointCloud::Ptr cloud(new PointCloud());
        pcl::fromROSMsg(*msg_transformed, *cloud);

        // remove the ground plane //
        http://pointclouds.org/documentation/tutorials/passthrough.php
        pcl::PassThrough<pcl::PointXYZ> pass; pass.setInputCloud(cloud);
        pass.setFilterFieldName("z"); pass.setFilterLimits(-0.5,
        TABLETOP_HEIGHT); pass.setFilterLimitsNegative(true); // allow to pass
        what is outside of this range pass.filter(*cloud);

        pass.setFilterFieldName("x"); pass.setFilterLimits(0, 0.4);
        pass.setFilterLimitsNegative(false); // allow to pass what is outside of
        this range pass.filter(*cloud);

        pass.setFilterFieldName("y"); pass.setFilterLimits(-0.2, 0.2);
        pass.setFilterLimitsNegative(false); // allow to pass what is outside of
        this range pass.filter(*cloud); // std::cout << "Removed floor" <<
        std::endl;

        // Downsample this pc pcl::VoxelGrid<pcl::PointXYZ> downsample;
        downsample.setInputCloud(cloud); downsample.setLeafSize(LEAF_SIZE,
        LEAF_SIZE, LEAF_SIZE); // downsample.filter(*cloud);

        pcl::PCLPointCloud2 cloud_filtered2;

        std::cout << "here1" << std::endl; // ros::Duration(1.0).sleep();
        cloud->header.frame_id = lens_frame;
        pcl_ros::transformPointCloud(lens_frame, *cloud, *cloud, listener);
        std::cout << "here2" << std::endl;

        pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

        occ.setInputCloud(cloud); std::cout << "here2a" << std::endl;
        occ.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE); std::cout << "here2b"
        << std::endl; occ.initializeVoxelGrid(); std::cout << "here2c" <<
        std::endl;

        Eigen::Vector3i box = occ.getMaxBoxCoordinates(); std::cout << "box in
        unobs: \n" << box.matrix() << std::endl; PointCloud cloud_filtered =
        occ.getFilteredPointCloud(); std::vector<Eigen::Vector3i>
        occluded_voxels; occ.occlusionEstimationAll(occluded_voxels); //
        std::cout << "Proportion occluded: " << (float)occluded_voxels.size() /
        (float)(box(0) * box(1) * box(2)) << std::endl; PointCloud::Ptr
        cloud_occluded(new PointCloud); for (const auto &voxel :
        occluded_voxels)
        {
          Eigen::Vector4f coord = occ.getCentroidCoordinate(voxel);
          cloud_occluded->push_back(pcl::PointXYZ(coord(0), coord(1), coord(2)));
        }

        //convert to world cloud_occluded->header.frame_id = "lens_link";
        PointCloud::Ptr cloud_occluded_world(new PointCloud);
        cloud_occluded_world->header.frame_id = "world"; std::cout << "here3" <<
        std::endl; pcl_ros::transformPointCloud(target_frame, *cloud_occluded,
        *cloud_occluded_world, listener); std::cout << "here4" << std::endl;

        pass.setInputCloud(cloud_occluded_world); pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.5, TABLETOP_HEIGHT);
        pass.setFilterLimitsNegative(true); // allow to pass what is outside of
        this range // pass.filter(*cloud_occluded_world);
        pass.setFilterFieldName("x"); pass.setFilterLimits(0.0, 0.4);
        pass.setFilterLimitsNegative(false); // allow to pass what is outside of
        this range pass.setFilterFieldName("y"); pass.setFilterLimits(-0.2,
        0.2); pass.setFilterLimitsNegative(false); // allow to pass what is
        outside of this range // pass.filter(*cloud_occluded_world);
        orig_unobserved_ = *cloud_occluded_world; num_unobserved_clbks++;

        pcl::toPCLPointCloud2(*cloud_occluded_world, cloud_filtered2);

        sensor_msgs::PointCloud2Ptr output(new sensor_msgs::PointCloud2());

        pcl_conversions::fromPCL(cloud_filtered2, *output);
        output->header.frame_id = "world"; std::cout << "here5" << std::endl;

        // Publish the data occ_pub.publish(*output);
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
    // grid_map::GridMapRosConverter::toCvImage(gridMapMask, layer, "mono8",
    // *cvImageObservedMask);

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
    // float prob = probability_by_state_.find(state)->second;
    int index = ipp_.size();
    for (auto it = new_cloud.begin(); it != new_cloud.end(); it++, index++)
    {
      ipp_.insert(std::make_pair(index, std::make_pair(*it, state))); // assumes uniform probability for a batch of points
    }
    std::cout << "Appended and included point cloud with probabilities!" << std::endl;
  }

  Eigen::Quaternionf calculateNextBestView(const std::set<int> &finger_occluded_voxels, std::vector<float> &view_entropies, int &best_view_id)
  {
    std::cout << "Beginning calculation of next best view!" << std::endl;
    Eigen::Vector4f best_view;
    best_view << 0.0f, 0.0f, 0.0f, 0.0f;
    float entropy = 0.0f;
    // int best_view_id;
    int view_id = 0;
    ros::Time start = ros::Time::now();
    for (const auto &v : nbv_origins_)
    {
      float e = calculateViewEntropy(v, finger_occluded_voxels);
      std::cout << "Az/El: " << azimuths_[view_id] << " " << elevations_[view_id] << " entropy: " << e << std::endl;
      view_entropies.push_back(e);
      if (e > entropy)
      {
        best_view = v;
        entropy = e;
        best_view_id = view_id;
      }
      view_id++;
    }
    std::cout << "NBV calc took: " << ros::Time::now() - start << std::endl;
    std::cout << "Origin: " << best_view[0] << " " << best_view[1] << " " << best_view[2] << " Entropy: " << entropy << "best id: " << best_view_id << std::endl;
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
      // std::cout << "Origin: " << origin[0] << " " << origin[1] << " " <<
      // origin[2] << " Direction: " << direction[0] << " " << direction[1] << "
      // " << direction[2] << " Target Voxel: " << target_voxel[0] << " " <<
      // target_voxel[1] << " " << target_voxel[2] << std::endl; std::cout <<
      // "Target: " << it->second.first.x << " " << it->second.first.y << " " <<
      // it->second.first.z << std::endl;

      rayTraversal(out_ray, target_voxel, origin, direction);
      for (size_t i = 0; i < out_ray.size(); i++) // for each voxel the ray passed through
      {
        int index = gridCoordToVoxelIndex(out_ray[i]);
        // std::cout << "Grid coord: " << out_ray[i][0] << " " << out_ray[i][1]
        // << " " << out_ray[i][2] << " Voxel index: " << index << std::endl;
        auto it_cell = cell_visited.find(index);
        if (it_cell == cell_visited.end()) // if the voxel hasn't been included before
        {
          // std::cout << "Adding cell index to list: " << index << std::endl;
          cell_visited.insert(index);
          out_ray_unique.push_back(out_ray[i]);
        }
        else
        {
          // std::cout << "Not adding a repeat observation of voxel ID: " <<
          // index << std::endl;
        }
      }
      entropy += calculateEntropyAlongRay(out_ray, finger_occluded_voxels); // was out ray unique
    }

    return entropy;
  }

  float calculateEntropyAlongRay(const std::vector<Eigen::Vector3i> &ray, const std::set<int> &finger_occluded_voxels) // TODO distance weighted
  {
    float entropy = 0.0f;
    bool continue_in_loop = true;
    for (const auto &v : ray)
    {
      if (continue_in_loop) // to make the former "break" behavior more explicit
      {
        int index = gridCoordToVoxelIndex(v);
        if (finger_occluded_voxels.find(index) != finger_occluded_voxels.end()) // if this voxel is hidden by the fingers
        {
          entropy += 0.0f; // don't learn anything
        }
        else
        {
          // std::cout << ">> along ray... Grid coord: " << v[0] << " " << v[1] <<
          // " " << v[2] << " Voxel index: " << index << std::endl;
          auto it_prob = cell_occupancy_state_.find(gridCoordToVoxelIndex(v));
          ROS_ASSERT(it_prob != cell_occupancy_state_.end());
          int state = it_prob->second;
          float p = probability_by_state_.find(state)->second;
          if (state == Observation::OCCUPIED)
            continue_in_loop = false;
          entropy += -p * log(p) - (1.0f - p) * log(1.0f - p);
        }
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
    return coord[2] * (nr_ * nc_) + coord[0] * nc_ + coord[1];
  }

  Eigen::Vector4f gridCoordToWorldCoord(const Eigen::Vector3i &grid_coord)
  {
    Eigen::Vector4f v;
    v[0] = (grid_coord(0) + 0.5f) * leaf_size_[0] + origbb_T_w_.getOrigin().getX();
    v[1] = (grid_coord(1) + 0.5f) * leaf_size_[1] + origbb_T_w_.getOrigin().getY();
    v[2] = (grid_coord(2) + 0.5f) * leaf_size_[2] + origbb_T_w_.getOrigin().getZ();
    v[3] = 0.0f;
    // std::cout << "grid coord: " << grid_coord(0) <<" "<<grid_coord(1)<<"
    // "<<grid_coord(2)<< " world coord:" << v(0) <<" "<<v(1)<<" "<<v(2) <<
    // std::endl;
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
    // std::cout << "Start world coord: " << start[0] << " " << start[1] << " "
    // << start[2] << std::endl;

    // i,j,k coordinate of the voxel were the ray enters the voxel grid
    Eigen::Vector3i ijk = worldCoordToGridCoord(start[0], start[1], start[2]);
    // std::cout << "Entry voxel grid coord: " << ijk[0] << " " << ijk[1] << " "
    // << ijk[2] << std::endl;

    // steps in which direction we have to travel in the voxel grid
    int step_x, step_y, step_z;

    // centroid coordinate of the entry voxel
    Eigen::Vector4f voxel_max = gridCoordToWorldCoord(ijk);
    // std::cout << "Entry voxel world coord: " << voxel_max[0] << " " <<
    // voxel_max[1] << " " << voxel_max[2] << std::endl;

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

    // while ((ijk[0] < nr_ + 1) && (ijk[0] >= 0) && // ????? (ijk[1] < nc_ + 1)
    //        && (ijk[1] >= 0) && (ijk[2] < nl_ + 1) && (ijk[2] >= 0))
    while ((ijk[0] < nr_) && (ijk[0] >= 0) && // ?????
           (ijk[1] < nc_) && (ijk[1] >= 0) &&
           (ijk[2] < nl_) && (ijk[2] >= 0))
    {
      // add voxel to ray
      out_ray.push_back(ijk);
      Eigen::Vector4f wc = gridCoordToWorldCoord(ijk);
      // std::cout << "Saw voxel: " << ijk[0] << " " << ijk[1] << " " << ijk[2]
      // << " at " << wc[0] << " " << wc[1] << " " << wc[2] << std::endl; check
      // if we reached target voxel
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
    float az_min = 0.0f;        // M_PI / 16;
    float az_max = 2.0f * M_PI; //+ M_PI / 16;
    float el_min = -M_PI / 2.0f + M_PI / 8;
    float el_max = M_PI / 2.0f - M_PI / 8;
    float az_incr = (az_max - az_min) / NUM_AZIMUTH_POINTS;
    float el_incr = (el_max - el_min) / NUM_ELEVATION_POINTS;

    // find center of orig bb
    float orig_bb_center_x = 0.5f * (orig_bb_min_.x + orig_bb_max_.x);
    float orig_bb_center_y = 0.5f * (orig_bb_min_.y + orig_bb_max_.y);
    float orig_bb_center_z = 0.5f * (orig_bb_min_.z + orig_bb_max_.z);

    for (float az = az_min; az < az_max; az += az_incr)
    {
      for (float el = el_min; el < el_max; el += el_incr)
      {
        Eigen::Vector4f v;
        v[0] = VIEW_RADIUS * cos(az) * cos(el) + orig_bb_center_x;
        v[1] = VIEW_RADIUS * sin(az) * cos(el) + orig_bb_center_y;
        v[2] = VIEW_RADIUS * sin(el) + orig_bb_center_z;
        v[3] = 0.0f;
        nbv_origins_.push_back(v);
        // calculate quaternion from view to origin
        // https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
        Eigen::Quaternionf q;
        Eigen::Vector3f axis;
        axis << -sin(az), cos(az), 0.0f;
        q = Eigen::AngleAxisf(-el, axis) * Eigen::AngleAxisf(az, Eigen::Vector3f::UnitZ());
        nbv_orientations_.push_back(q);
        azimuths_.push_back(az);
        elevations_.push_back(el);
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
      gr.publishPointCloudByCategoryMarkerArray();
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}