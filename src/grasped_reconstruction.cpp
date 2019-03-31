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
    // occ_pub = n.advertise<sensor_msgs::PointCloud2>("occluded_voxels", 1);
    // ch_pub = n.advertise<pcl_msgs::PolygonMesh>("convex_hull_mesh", 1);
    // hm_pub = n.advertise<sensor_msgs::PointCloud2>("object_without_table", 1);
    coeff_pub = n.advertise<pcl_msgs::ModelCoefficients>("output", 1);
    object_pub = n.advertise<sensor_msgs::PointCloud2>("segmented_object", 1);
    tabletop_pub = n.advertise<sensor_msgs::PointCloud2>("tabletop", 1);
    bb_pub = n.advertise<visualization_msgs::Marker>("bbox", 1);
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
  }
  ros::NodeHandle _n;
  ros::Subscriber pc_sub;
  ros::Publisher coeff_pub, object_pub, tabletop_pub, bb_pub;
  tf::TransformListener listener;
  tf::StampedTransform world_T_lens_link_tf;

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
    downsample.setLeafSize (0.01f, 0.01f, 0.01f);
    downsample.filter(*cloud);

    sensor_msgs::PointCloud2 tabletop_output;
    pcl::toROSMsg(*cloud, tabletop_output);
    tabletop_pub.publish(tabletop_output);

    pcl::ModelCoefficients coefficients;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    // remove the tabletop
    seg.setInputCloud(cloud);
    seg.segment(*inliers, coefficients);
    std::cout << "Removed tabletop" << std::endl;

    // Publish the model coefficients
    pcl_msgs::ModelCoefficients ros_coefficients;
    pcl_conversions::fromPCL(coefficients, ros_coefficients);
    coeff_pub.publish(ros_coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud);

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

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    object_pub.publish(output);
    std::cout << "Published cloud" << std::endl;

    pcl::PointXYZ min, max;
    pcl::getMinMax3D(*cloud, min, max);
    std::cout << "Got bounding box" << std::endl;

    if (coefficients.values.size() > 0)
    {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = ros::Time();
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x = 0.5f * (max.x + min.x);
      marker.pose.position.y = 0.5f * (max.y + min.y);
      marker.pose.position.z = 0.5f * (max.z - coefficients.values[3]);
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = max.x - min.x;
      marker.scale.y = max.y - min.y;
      marker.scale.z = max.z + coefficients.values[3];
      marker.color.a = 0.5; // Don't forget to set the alpha!
      marker.color.r = 1.0;
      marker.color.g = 0.5;
      marker.color.b = 0.0;
      bb_pub.publish(marker);
    }
  }

  void processPointCloud()
  {
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
    gr.processPointCloud();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}