#include "ros/ros.h"
#include <gazebo_msgs/SetModelState.h>
#include <gazebo/gazebo.hh>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <math.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/io.h>
#include "pcl_ros/transforms.h"
#include "pcl_ros/impl/transforms.hpp"

const float PI_F = 3.14159265358979f;

class CameraMotion
{
public:
  CameraMotion(const ros::NodeHandle &n) : n_(n){};
  void rotateCameraAboutOrigin()
  {
    ros::Time now = ros::Time::now();
    ros::Duration diff = now - start;
    float diff_secs = static_cast<float>(diff.toSec());
    // int sign = (diff_secs % 2 == 1) ? 1 : -1;
    float yaw = ROTATION_RATE * diff_secs;
    // float x = sign * 1.0f;
    if (yaw > 2 * PI_F)
    {
      yaw -= 2 * PI_F;
    }
    q.setRPY(0.0f, 0.0f, yaw);
    transform_out.setRotation(q);
    transform_out.setOrigin(tf::Vector3(0.0f, 0.0f, 0.0f));
    br.sendTransform(tf::StampedTransform(transform_out, ros::Time::now(), "map", "center_link"));
    gazebo_msgs::ModelState ms;

    try
    {
      listener.lookupTransform("/map", "/camera_link", ros::Time(0), transform_in);
      // transform_in_tf.setOrigin(getOrigin(transform_in));
      // transform_in_tf.setRotation(getRotation(transform_in));
      std::string target_frame("/camera_link");
      pcl_ros::transformPointCloud(target_frame, cloud_in, cloud_out, listener);
      cloud_pub.publish(cloud_out);
    }
    catch (tf::TransformException ex)
    {
      ms.pose.position.x = 1.0f;
      ms.pose.position.y = 0.0f;
      ms.pose.position.z = 0.25f;
      ROS_ERROR("%s", ex.what());

      ros::Duration(1.0).sleep();
    }

    ms.model_name = (std::string) "kinect";
    ms.reference_frame = (std::string) "map";
    ms.pose.position.x = transform_in.getOrigin().x();
    ms.pose.position.y = transform_in.getOrigin().y();
    ms.pose.position.z = transform_in.getOrigin().z();
    ms.pose.orientation.x = transform_in.getRotation().x();
    ms.pose.orientation.y = transform_in.getRotation().y();
    ms.pose.orientation.z = transform_in.getRotation().z();
    ms.pose.orientation.w = transform_in.getRotation().w();
    camera_pub.publish(ms);
  }

  void pcClbk(const sensor_msgs::PointCloud2 &msg)
  {
    cloud_in = msg;
  }

private:
  ros::NodeHandle n_;
  ros::Publisher camera_pub = n_.advertise<gazebo_msgs::ModelState>("/gazebo/set_model_state", 1);
  ros::Publisher cloud_pub = n_.advertise<sensor_msgs::PointCloud2>("/point_cloud_transformed", 1);
  ros::Subscriber cloud_sub = n_.subscribe("/camera/depth/points", 1, &CameraMotion::pcClbk, this);
  ros::Time start = ros::Time::now();
  tf::TransformBroadcaster br;
  tf::TransformListener listener;
  tf::Transform transform_out;
  tf::StampedTransform transform_in;
  tf::Quaternion q;
  int increment = 0;
  float ROTATION_RATE = 0 * 0.25f;
  float CAMERA_OFFSET = 2.0f; // [m]

  sensor_msgs::PointCloud2 cloud_in, cloud_out;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "grasped_reconstruction");

  ros::NodeHandle n;

  // ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);

  CameraMotion cm(n);
  ros::Rate loop_rate(10);

  while (ros::ok())
  {

    // chatter_pub.publish(msg);
    cm.rotateCameraAboutOrigin();
    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}
