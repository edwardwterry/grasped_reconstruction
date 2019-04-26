// #define PCL_NO_PRECOMPILE
#include "ros/ros.h"
// #include <gazebo_msgs/SetModelState.h>
// #include <gazebo/gazebo.hh>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <math.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl_ros/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
// #include <pcl_ros/filters/voxel_grid_occlusion_estimation3.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
// #include <pcl_msgs/PolygonMesh.h>
// #include <mesh_msgs/TriangleMesh.h>
#include <Eigen/Dense>
// #include <pcl/people/person_cluster.h>
// #include <pcl/people/height_map_2d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
// #include <pcl/sample_consensus/method_types.h>
// #include <pcl/sample_consensus/model_types.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl_ros/transforms.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/search/kdtree.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <pcl/filters/crop_hull.h>
#include <grasped_reconstruction/CalculateNbv.h>