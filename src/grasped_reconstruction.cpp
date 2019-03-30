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

    try
    {
      ros::Time now = ros::Time::now();
      listener.waitForTransform("/world", "/lens_link",
                                now, ros::Duration(3.0));
      listener.lookupTransform("/world", "/lens_link",
                               now, world_T_lens_link_tf);
    }
    catch(tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
    }
  }
  ros::NodeHandle _n;
  ros::Subscriber pc_sub;
  ros::Publisher coeff_pub;
  tf::TransformListener listener;
  tf::StampedTransform world_T_lens_link_tf;

  void pcClbk(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    // http://wiki.ros.org/pcl/Tutorials#pcl.2BAC8-Tutorials.2BAC8-hydro.sensor_msgs.2BAC8-PointCloud2
    // Convert the sensor_msgs/PointCloud2 data to pcl/PointCloud
    sensor_msgs::PointCloud2Ptr msg_transformed (new sensor_msgs::PointCloud2());
    std::string target_frame("world");
    pcl_ros::transformPointCloud(target_frame, *msg, *msg_transformed, listener);
    PointCloud cloud;
    pcl::fromROSMsg(*msg_transformed, cloud);

    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(cloud.makeShared());
    seg.segment(inliers, coefficients);

    // Publish the model coefficients
    pcl_msgs::ModelCoefficients ros_coefficients;
    pcl_conversions::fromPCL(coefficients, ros_coefficients);
    coeff_pub.publish(ros_coefficients);
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