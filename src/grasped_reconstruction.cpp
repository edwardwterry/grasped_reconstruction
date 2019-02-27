#include "ros/ros.h"
#include <gazebo_msgs/SetModelState.h>
#include <gazebo/gazebo.hh>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <math.h>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap/ColorOcTree.h>
#include <std_msgs/Float32.h>
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
  ros::Subscriber octree_sub = n_.subscribe("/octomap_binary", 1, &CameraMotion::octreeClbk, this);
  ros::Time start = ros::Time::now();
  tf::TransformBroadcaster br;
  tf::TransformListener listener, listener2;
  tf::Transform transform_out;
  tf::StampedTransform transform_in, transform2_in;
  tf::Quaternion q;
  float ROTATION_RATE = 0.25f;
  float CAMERA_OFFSET = 1.0f; // [m]
  // octomap_msgs::Octomap octree_msg;

  void calculateUnknownVoxels()
  {
    // if (octree != NULL)
    // {
    // https: //github.com/personalrobotics/or_octomap/issues/2
    // // https://groups.google.com/forum/#!topic/octomap/zNsuIyVtko8
  }
  void octreeClbk(const octomap_msgs::Octomap &msg)
  {
    octomap::AbstractOcTree *tree = octomap_msgs::binaryMsgToMap(msg);
    octomap::OcTree *octree = dynamic_cast<octomap::OcTree *>(tree);
    int visited = 0;
    int unknown = 0;
    double resolution = 0.05;p
    for (double ix = -0.2; ix < 0.2; ix += resolution)
      for (double iy = -0.2; iy < 0.2; iy += resolution)
        for (double iz = 0.0; iz < 0.3; iz += resolution)
        {
          visited++;
          if (!octree->search(ix, iy, iz))
          {
            unknown++;
          }
        }
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