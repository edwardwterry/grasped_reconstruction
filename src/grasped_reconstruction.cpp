#include "ros/ros.h"
#include <gazebo_msgs/SetModelState.h>
#include <gazebo/gazebo.hh>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <math.h>
#include <octomap_msgs/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap/ColorOcTree.h>
#include <std_msgs/Float32.h>
#include <Eigen/Dense>

// #include <sensor_msgs/PointCloud2.h>

const float PI_F = 3.14159265358979f;

class CameraMotion
{
public:
  CameraMotion(const ros::NodeHandle &n) : n_(n)
  {
    transform_in.setOrigin(tf::Vector3(0.25f, 0.0f, 0.0f));
    transform_in.setRotation(tf::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
  };
  void rotateCameraAboutOrigin()
  {
    ros::Time now = ros::Time::now();
    ros::Duration diff = now - start;
    gazebo_msgs::ModelState ms;

    try
    {
      listener.waitForTransform("/world", "/camera_link", ros::Time(0), ros::Duration(3.0));
      listener.lookupTransform("/world", "/camera_link", ros::Time(0), transform_in);
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }

    float diff_secs = static_cast<float>(diff.toSec());
    float yaw = ROTATION_RATE * diff_secs;
    if (yaw > 2 * PI_F)
    {
      yaw -= 2 * PI_F;
    }
    q.setRPY(0.0f, 0.0f, -yaw + PI_F / 2); // point towards origin
    transform_out.setRotation(q);
    transform_out.setOrigin(tf::Vector3(CAMERA_OFFSET * sin(yaw), CAMERA_OFFSET * cos(yaw), 0.2f));
    br.sendTransform(tf::StampedTransform(transform_out, ros::Time::now(), "world", "base_link"));

    ms.model_name = (std::string) "kinect";
    ms.reference_frame = (std::string) "world";
    ms.pose.position.x = transform_in.getOrigin().x();
    ms.pose.position.y = transform_in.getOrigin().y();
    ms.pose.position.z = transform_in.getOrigin().z();
    ms.pose.orientation.x = transform_in.getRotation().x();
    ms.pose.orientation.y = transform_in.getRotation().y();
    ms.pose.orientation.z = transform_in.getRotation().z();
    ms.pose.orientation.w = transform_in.getRotation().w();
    camera_pub.publish(ms);
  }

private:
  ros::NodeHandle n_;
  ros::Publisher camera_pub = n_.advertise<gazebo_msgs::ModelState>("/gazebo/set_model_state", 1);
  ros::Publisher unknown_pub = n_.advertise<std_msgs::Float32>("/proportion_unknown", 1);
  ros::Publisher unknown_voxel_pub = n_.advertise<sensor_msgs::PointCloud2>("/unknown_voxels_pc2", 1);
  ros::Subscriber octree_sub = n_.subscribe("/octomap_binary", 1, &CameraMotion::octreeClbk, this);
  ros::Subscriber unknown_voxel_sub = n_.subscribe("/octomap_binary_unknown", 1, &CameraMotion::octreeUnknownClbk, this);
  ros::Time start = ros::Time::now();
  tf::TransformBroadcaster br;
  tf::TransformListener listener, listener2;
  tf::Transform transform_out;
  tf::StampedTransform transform_in, transform2_in;
  tf::Quaternion q;
  float ROTATION_RATE = 0.25f;
  float CAMERA_OFFSET = 1.0f; // [m]

  float HFOV = PI_F / 3.0f;
  int FRAME_WIDTH = 640;
  int FRAME_HEIGHT = 480;
  int cx = FRAME_WIDTH / 2;
  int cy = FRAME_HEIGHT / 2;
  float fx = (FRAME_WIDTH / 2.0f) / tan(HFOV / 2.0f);
  float fy = fx;
  float RESOLUTION = 0.05f;   // [m]
  float RAYCAST_RANGE = 3.0f; // [m]

  struct DirectedRay
  {
    octomap::point3d origin;
    octomap::point3d dir;
  };

  octomap::AbstractOcTree *tree;
  octomap::OcTree *octree; // can't forget nodes here. only adding!
  octomap::AbstractOcTree *tree_unobs;
  octomap::OcTree *octree_unobs;
  octomap::Octree octree_unobserved;
  octomap::Pointcloud octomap_unobs_pc;
  // octomap_msgs::Octomap octree_msg;

  // calculate ray casting directions

  std::vector<Eigen::Vector3d> calculateRayCastDirections()
  {
    Eigen::Vector3d dir;
    std::vector<Eigen::Vector3d> raycast_vectors;
    for (int x = 0; x < FRAME_WIDTH; x++)
    {
      for (int y = 0; y < FRAME_HEIGHT; y++)
      {
        dir(0) = (x - cx) / fx;
        dir(1) = (y - cy) / fy;
        dir(2) = 1.0f;
        dir.normalize();
        ray_directions.push_back(dir);
      }
    }
    return raycast_vectors;
  }

  // calculate total number of unknown voxels revealed from a given viewpoint
  // don't want to double count voxel (two rays may intersect the same voxel)
  // how can you keep track of something that's not there?

  tf::Quaternion getQuat(float yaw){
    return q.setRPY(0.0f, 0.0f, -yaw + PI_F / 2); // point towards origin
  }

  tf::Transform calculateNbv()
  {
    int num_views_to_consider = 6;
    std::vector<float> yaw_angles;
    std::vector<int> num_unknown_voxels;
    for (int i = 0; i < num_views_to_consider; i++)
    {
      yaw_angles.push_back((float)i / (2.0f * PI_F));
    }
    for (const auto &angle : yaw_angles)
    {
      num_unknown_voxels.push_back(calculateUnknownVoxels());
    }

  }

  void calculateUnknownVoxels(view)
  {
    octomap::OcTree octree_unobs_map(RESOLUTION);
    // https://answers.ros.org/question/252632/how-to-correctly-build-an-octomap-using-a-series-of-registered-point-clouds/
    octree_unobs_map.insertPointCloud(octomap_unobs_pc, octomap::point3d(0, 0, 0), 10.0f, false, false);
    int unknown = 0;
    octomap::KeyRay ray;

    // get tf of where the camera is pointing (transform_hyp)
    std::vector<Eigen::Vector3d> raycast_vectors = calculateRayCastDirections();
    // need to rotate raycast_vectors to align with transform_hyp
    octomap::point3d origin = octomap::point3d(transform_hyp.getOrigin().getX(), transform_hyp.getOrigin().getY(), transform_hyp.getOrigin().getZ());
    for (const auto &v : raycast_vectors)
    {
      octomap::point3d end = origin + v * RAYCAST_RANGE;
      octomap::computeRayKeys(origin, end, ray);
      for (octomap::KeyRay::iterator it = ray.begin(); it != ray.end(); it++)
      {
        if (octree_unobs->search(*it) && it.getLogOdds() > 0) // this is intended to avoid double counting of voxels which are found to be unobserved
        {
          it.getKey()->setLogOdds(octomap::logodds(0.0f));
          octree_unobs_map.updateInnerOccupancy();
          // change prob: https://answers.ros.org/question/53160/update-node-back-to-unknown-in-octomap/
          unknown++;
        }
      }
    }

    // if (octree != NULL)
    // {
    // https: //github.com/personalrobotics/or_octomap/issues/2
    // // https://groups.google.com/forum/#!topic/octomap/zNsuIyVtko8
  }

  // void octreeUnknownClbk(const octomap_msgs::Octomap &msg)
  // {
  //   tree_unobs = octomap_msgs::binaryMsgToMap(msg);
  //   octree_unobs = dynamic_cast<octomap::OcTree *>(tree_unobs);
  // }
  void octreeClbk(const octomap_msgs::Octomap &msg)
  {
    tree = octomap_msgs::binaryMsgToMap(msg);
    octree = dynamic_cast<octomap::OcTree *>(tree);

    int visited = 0;
    int unknown = 0;
    pcl::PointXYZ p;
    pcl::PointCloud<pcl::PointXYZ> temp_data_cloud;

    for (double ix = -0.2; ix < 0.2; ix += RESOLUTION)
      for (double iy = -0.2; iy < 0.2; iy += RESOLUTION)
        for (double iz = 0.0; iz < 0.3; iz += RESOLUTION)
        {
          visited++;
          if (!octree->search(ix, iy, iz))
          {
            unknown++;
            p.x = ix;
            p.y = iy;
            p.z = iz;
            temp_data_cloud.push_back(p);
          }
        }
    // from http://ros-developer.com/2017/02/23/converting-pcl-point-cloud-to-ros-pcl-cloud-message-and-the-reverse/
    // make a new point cloud composed of only unobserved voxels
    sensor_msgs::PointCloud2 unknown_voxel_msg;
    pcl::toROSMsg(temp_data_cloud, unknown_voxel_msg);
    // http://docs.ros.org/jade/api/octomap_ros/html/conversions_8h.html
    octomap::pointCloud2ToOctomap(unknown_voxel_msg, octomap_unobs_pc);

    // unknown_voxel_pub.publish(unknown_voxel_msg);
    std::cout << "Proportion unknown: " << float(unknown) / float(visited) << std::endl;
    std_msgs::Float32 unknown_msg;
    unknown_msg.data = float(unknown) / float(visited);
    unknown_pub.publish(unknown_msg);
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "grasped_reconstruction");

  ros::NodeHandle n;
  CameraMotion cm(n);
  ros::Rate loop_rate(10);

  while (ros::ok())
  {
    cm.rotateCameraAboutOrigin();
    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}