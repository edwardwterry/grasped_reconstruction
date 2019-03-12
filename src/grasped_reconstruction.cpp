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
#include <pcl/surface/convex_hull.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl_msgs/PolygonMesh.h>
// #include <mesh_msgs/TriangleMesh.h>
#include <Eigen/Dense>
#include <pcl/people/person_cluster.h>
#include <pcl/people/height_map_2d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

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
    ch_sub = _n.subscribe("/extract_plane_indices/output", 1, &GraspedReconstruction::convexHullClbk, this);
    occ_pub = n.advertise<sensor_msgs::PointCloud2>("occluded_voxels", 1);
    ch_pub = n.advertise<pcl_msgs::PolygonMesh>("convex_hull_mesh", 1);
  }
  ros::NodeHandle _n;
  ros::Subscriber pc_sub, ch_sub;
  ros::Publisher occ_pub, ch_pub;

  void heightMapClbk(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    // Container for original & filtered data
    pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);

    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud);
    PointCloud *pc = new pcl::PointCloud<pcl::PointXYZ>;
    // https://stackoverflow.com/questions/10644429/create-a-pclpointcloudptr-from-a-pclpointcloud
    PointCloud::Ptr pcPtr(pc);

    // http://pointclouds.org/documentation/tutorials/planar_segmentation.php
    pcl::ModelCoefficients::Ptr ground_coeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(pcPtr);
    seg.segment(*inliers, *ground_coeffs);
    Eigen::Vector4f ground_coeffs_eigen = Eigen::Vector4f(ground_coeffs->values[0], ground_coeffs->values[1], ground_coeffs->values[2], ground_coeffs->values[3]);
    float sqrt_ground_coeffs = ground_coeffs_eigen.norm();
    // create person cluster
    bool head_centroid = false;
    bool vertical = false;
    // pcl::people::PersonCluster<pcl::PointXYZ> *person_cluster = new pcl::people::PersonCluster<pcl::PointXYZ>(pcPtr, *inliers, ground_coeffs_eigen, sqrt_ground_coeffs, head_centroid, vertical);
    pcl::people::PersonCluster<pcl::PointXYZ> person_cluster(pcPtr, *inliers, ground_coeffs_eigen, sqrt_ground_coeffs, head_centroid, vertical);

    // pcl::people::HeightMap2D<pcl::PointXYZ>::PointCloudPtr hm_ptr(new pcl::people::HeightMap2D<pcl::PointXYZ>);
    pcl::people::HeightMap2D<pcl::PointXYZ> hm;// (new pcl::people::HeightMap2D<pcl::PointXYZ>);
    hm.compute(person_cluster);
    // std::vector<int> height_map = hm_ptr->getHeightMap();
    std::vector<int> height_map = hm.getHeightMap();
    // hm_ptr->setInputCloud(pcPtr);
    // hm_ptr->setGround(ground_coeffs_eigen);
    // get bounding box
    // people::HeightMap2D to get highest point
    // set ground plane coefficients
    // compute
    // getHeightMap
    // create
  }

  void convexHullClbk(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    // Container for original & filtered data
    pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);

    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud);

    PointCloud *pc = new pcl::PointCloud<pcl::PointXYZ>;
    // https://stackoverflow.com/questions/10644429/create-a-pclpointcloudptr-from-a-pclpointcloud
    PointCloud::Ptr pcPtr(pc);

    pcl::fromPCLPointCloud2(*cloud, *pc);
    // https://answers.ros.org/question/59443/how-can-i-construct-convex-hull-from-point-cloud/
    pcl::ConvexHull<pcl::PointXYZ> cHull;
    pcl::PointCloud<pcl::PointXYZ> cHull_points;
    std::vector<pcl::Vertices> polygons;
    cHull.setInputCloud(pcPtr);
    cHull.reconstruct(cHull_points, polygons);

    // https://github.com/ethz-asl/infinitam/blob/897603eb95e859268d72d80cb870cf032976f40a/InfiniTAM/infinitam_ros_node.cpp#L412

    pcl_msgs::PolygonMesh pm;
    pm.header = std_msgs::Header();
    pm.header.frame_id = "lens_link";
    sensor_msgs::PointCloud2 pc2;
    pcl::toROSMsg(cHull_points, pc2);
    pm.cloud = pc2;
    std::vector<pcl_msgs::Vertices> polys;
    polys.reserve(polygons.size());
    int idx = 0;
    for (const auto &p : polygons)
    {
      pcl_msgs::Vertices vertices;
      for (const auto &v : p.vertices)
      {
        vertices.vertices.push_back(v);
      }
      polys.push_back(vertices);
    }
    pm.polygons = polys;
    ch_pub.publish(pm);
  }

  void pcClbk(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {
    // Container for original & filtered data
    pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
    pcl::PCLPointCloud2 cloud_filtered2;
    // PointCloud cloud_filtered;

    // Convert to PCL data type
    pcl_conversions::toPCL(*cloud_msg, *cloud);
    // std::cout<<"width: "<<cloudPtr->width<<std::endl;
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

    PointCloud *pc = new pcl::PointCloud<pcl::PointXYZ>;
    // https://stackoverflow.com/questions/10644429/create-a-pclpointcloudptr-from-a-pclpointcloud
    PointCloud::Ptr pcPtr(pc);

    pcl::fromPCLPointCloud2(*cloud, *pc);

    pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> occ;

    occ.setInputCloud(pcPtr);
    occ.setLeafSize(0.025, 0.025, 0.025);
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
    PointCloud cloud_occluded;
    for (const auto &voxel : occluded_voxels)
    {
      // std::cout<<"voxel(0): "<<voxel(0)<<std::endl;
      Eigen::Vector4f coord = occ.getCentroidCoordinate(voxel);
      cloud_occluded.push_back(pcl::PointXYZ(coord(0), coord(1), coord(2)));
    }
    // pcl::toPCLPointCloud2(cloud_filtered, cloud_filtered2);
    // // Convert to ROS data type
    pcl::toPCLPointCloud2(cloud_occluded, cloud_filtered2);

    sensor_msgs::PointCloud2 output;

    pcl_conversions::fromPCL(cloud_filtered2, output);
    output.header.frame_id = "lens_link";
    // output.height = 640;
    // output.width = 480;

    // Publish the data
    occ_pub.publish(output);
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