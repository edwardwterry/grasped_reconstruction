#include "ros/ros.h"
#include <gazebo_msgs/SetModelState.h>
#include <gazebo/gazebo.hh>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <math.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <algorithm>
// #include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <Eigen/Dense>

// #include <sensor_msgs/PointCloud2.h>

const float PI_F = 3.14159265358979f;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
class GraspedReconstruction
{
public:
  GraspedReconstruction(ros::NodeHandle &n) : _n(n)
  {
    _n = n;
    pc_sub = _n.subscribe("/camera/depth/points", 1, &GraspedReconstruction::pcClbk, this);
    occ_pub = n.advertise<sensor_msgs::PointCloud2>("occluded_voxels", 1);
  }
  ros::NodeHandle _n;
  ros::Subscriber pc_sub;
  ros::Publisher occ_pub;
  void pcClbk(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    // Container for original & filtered data
    pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::PCLPointCloud2 cloud_filtered2;

    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud);
    // http://ros-developer.com/2017/08/03/converting-pclpclpointcloud2-to-pclpointcloud-and-reverse/
    // pcl::PointCloud<pcl::PointXYZ> *pc = new pcl::PointCloud<pcl::PointXYZ>;
    // pcl::PointCloudPtr pcPtr(pc);
    // pcl::fromPCLPointCloud2(*cloud, *pc);

    // Perform the actual filtering
    // pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    // sor.setInputCloud(cloudPtr);
    // sor.setLeafSize(0.1, 0.1, 0.1);
    // sor.filter(cloud_filtered);
    // pcl::PCLPointCloud2 point_cloud2;

    pcl::PointCloud<pcl::PointXYZ> *pc = new pcl::PointCloud<pcl::PointXYZ>;
    pcl::ConstPtr<pcl::PointCloud<pcl::PointXYZ>> pcPtr(pc);
    // pcl::PointCloud<pcl::PointXYZ> point_cloud;

    pcl::fromPCLPointCloud2(*cloud, *pc);

    // pcl::toPCLPointCloud2(point_cloud, point_cloud2);

    pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

    occ.initializeVoxelGrid();
    occ.setInputCloud(pcPtr);
    // occ.setLeafSize(0.1, 0.1, 0.1);
    // // occ.initializeVoxelGrid();
    // pcl::PointCloud<pcl::PointXYZ> cloud_filtered = occ.getFilteredPointCloud();
    // // std::vector<Eigen::Vector3i> occluded_voxels;
    // // occ.occlusionEstimationAll(occluded_voxels);
    // // for (const auto & voxel:occluded_voxels){
    // //   std::cout<<"voxel(0): "<<voxel(0)<<std::endl;
    // // }
    // pcl::toPCLPointCloud2(cloud_filtered, cloud_filtered2);
    // // Convert to ROS data type
    // sensor_msgs::PointCloud2 output;
    // pcl_conversions::fromPCL(cloud_filtered2, output);

    // // Publish the data
    // occ_pub.publish(output);
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
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}