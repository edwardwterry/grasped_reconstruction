#include "ros/ros.h"
#include <gazebo_msgs/SetModelState.h>
#include <gazebo/gazebo.hh>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <math.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap/CountingOcTree.h>
#include <octomap/OcTree.h>
#include <octomap/OcTreeBaseImpl.h>
#include <std_msgs/Float32.h>
#include <algorithm>

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
    // std::cout << "Publishing transform at: [xyz] [xyzw] " << ms.pose.position.x << " " << ms.pose.position.y << " " << ms.pose.position.z << " " << ms.pose.orientation.w << " " << ms.pose.orientation.y << " " << ms.pose.orientation.z << " " << ms.pose.orientation.w << std::endl;
  }

  bool nbv_calculated = false;
  bool received_octomap_msg = false;

  tf::Transform calculateNbv()
  {
    std::cout << "Calculating NBV..." << std::endl;
    tf::Transform nbv;
    int num_views_to_consider = 6;
    std::vector<float> yaw_angles;
    std::vector<int> num_unobs_voxels;
    for (int i = 0; i < num_views_to_consider; i++)
    {
      yaw_angles.push_back((float)i * (2.0f * PI_F) / num_views_to_consider);
    }
    if (octree->getRoot() != NULL)
    {
      std::cout << "Tree is not empty!" << std::endl;
      for (const auto &yaw : yaw_angles)
      {
        int num_unobs = calculateUnobsVoxels(getViewTransform(yaw));
        num_unobs_voxels.push_back(num_unobs);
      }
      nbv = getViewTransform(*std::max_element(num_unobs_voxels.begin(), num_unobs_voxels.end()));
    }
    nbv_calculated = true;

    return nbv;
  }
  octomap::AbstractOcTree *tree;
  octomap::OcTree *octree;
  // octomap::Pointcloud octomap_unobs_pc;

private:
  ros::NodeHandle n_;
  ros::Publisher camera_pub = n_.advertise<gazebo_msgs::ModelState>("/gazebo/set_model_state", 1);
  ros::Publisher unobs_pub = n_.advertise<std_msgs::Float32>("/proportion_unobs", 1);
  // ros::Publisher unknown_voxel_pub = n_.advertise<sensor_msgs::PointCloud2>("/unknown_voxels_pc2", 1);
  ros::Subscriber octree_sub = n_.subscribe("/octomap_binary", 1, &CameraMotion::octreeClbk, this);
  // ros::Subscriber unknown_voxel_sub = n_.subscribe("/octomap_binary_unknown", 1, &CameraMotion::octreeUnknownClbk, this);
  ros::Time start = ros::Time::now();
  tf::TransformBroadcaster br;
  tf::TransformListener listener, listener2;
  tf::Transform transform_out;
  tf::StampedTransform transform_in, transform2_in;
  tf::Quaternion q;
  float ROTATION_RATE = 0.0f;
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

  // calculate ray casting directions
  std::vector<octomath::Vector3> calculateRayCastDirections()
  {
    std::vector<octomath::Vector3> raycast_vectors;
    int STRIDE = 50;
    for (int x = 0; x < FRAME_WIDTH; x += STRIDE)
    {
      for (int y = 0; y < FRAME_HEIGHT; y += STRIDE)
      {
        octomath::Vector3 dir((x - cx) / fx, (y - cy) / fy, 1.0f);
        dir.normalize();
        raycast_vectors.push_back(dir);
      }
    }
    return raycast_vectors;
  }

  // calculate total number of unknown voxels revealed from a given viewpoint
  // don't want to double count voxel (two rays may intersect the same voxel)
  // how can you keep track of something that's not there?

  // tf::Quaternion getQuat(float yaw)
  // {
  //   return q.setRPY(0.0f, 0.0f, -yaw + PI_F / 2); // point towards origin
  // }

  int calculateUnobsVoxels(const tf::Transform &view)
  {
    std::cout << "Calculating unobserved voxels..." << std::endl;
    octomap::point3d_list unobs_centers;
    octomap::point3d pmin = octomap::point3d(-0.2f, -0.2f, 0.0f);
    octomap::point3d pmax = octomap::point3d(0.2f, 0.2f, 0.4f);
    octree->getUnknownLeafCenters(unobs_centers, pmin, pmax);
    for (const auto &centers : unobs_centers)
    {
      std::cout << "Unknown coords: " << centers.x() << " " << centers.y() << " " << centers.z() << std::endl;
    }
    // std::cout<<"Made it!"<<std::endl;
    // octomap::OcTree octree_unobs(RESOLUTION);
    // https://answers.ros.org/question/252632/how-to-correctly-build-an-octomap-using-a-series-of-registered-point-clouds/
    // octree_unobs.insertPointCloud(octomap_unobs_pc, octomap::point3d(0, 0, 0), 10.0f, false, false);
    int unobs = 0;

    // get tf of where the camera is pointing (view)

    std::vector<octomath::Vector3> raycast_vectors = calculateRayCastDirections();
    std::cout << "num of rays to be cast: " << raycast_vectors.size() << std::endl;

    // need to rotate raycast_vectors to align with view
    octomap::point3d origin = octomap::point3d(view.getOrigin().getX(), view.getOrigin().getY(), view.getOrigin().getZ());
    octomath::Quaternion rotation = octomath::Quaternion(view.getRotation().w(), view.getRotation().x(), view.getRotation().y(), view.getRotation().z());
    int count = 0;
#pragma omp parallel for
    for (const auto &v : raycast_vectors)
    {
      // if (count % 10000 == 0)
      // {
      //   std::cout << "Count: " << count << std::endl;
      // }
      std::cout << "Ray: " << count << std::endl;
      count++;
      std::vector<octomap::point3d> ray;
      octomap::point3d end = origin + v * RAYCAST_RANGE;
      std::cout << "Origin: " << origin.x() << " " << origin.y() << " " << origin.z() << std::endl;
      std::cout << "End: " << end.x() << " " << end.y() << " " << end.z() << std::endl;
      octree->computeRay(origin, end, ray);
      for (const auto &coord : ray)
      {
        std::cout << "Searching at this coordinate: " << coord.x() << " " << coord.y() << " " << coord.z() << std::endl;
        octomap::OcTreeNode *n = octree->search(coord);
        if (n != NULL)
        {
          // std::cout << "Node exists!" << std::endl;
          if (octree->isNodeOccupied(n))
          {
            // std::cout << "Voxel is occupied!" << std::endl;
            break; // you've hit an occupied cell, which occludes everything behind it
          }
          else
          {
            // std::cout << "Free voxel!" << std::endl;
          }
        }

        else // should be unobserved by this stage
        {
          std::list<octomap::point3d>::iterator it;
          it = std::find(unobs_centers.begin(), unobs_centers.end(), coord);
          if (it != unobs_centers.end())
          {
            std::cout << "This coord is unobserved: " << it->x() << " " << it->y() << " " << it->z() << std::endl;
            unobs++;
            unobs_centers.erase(it); // remove to avoid double counting for a potential future ray
          }
        }

        // Note: beware of conflating num voxels and their volume: https://answers.ros.org/question/149126/octomap-volume-estimation/

        // // stop if you hit an occupied cell (so you don't end up counting unobserved voxels behind occupied ones)
        // if (octree->search(r) && octree->isNodeOccupied(*it))
        // {
        //   break;
        // }
        // // if you hit an unobserved voxel
        // if (octree_unobs.search(*it) && octree_unobs.isOccupied(*it))
        // {
        //   // this is intended to avoid double counting of voxels which are found to be unobserved
        //   it->setLogOdds(octomap::logodds(0.0f));
        //   octree_unobs.updateInnerOccupancy();
        //   // change prob: https://answers.ros.org/question/53160/update-node-back-to-unknown-in-octomap/
        //   unobs++;
        // }
      }
    }
    std::cout << "Finished doing calcs!" << std::endl;

    return unobs;
    // if (octree != NULL)
    // {
    // https: //github.com/personalrobotics/or_octomap/issues/2
    // // https://groups.google.com/forum/#!topic/octomap/zNsuIyVtko8
  }

  tf::Transform getViewTransform(const float yaw)
  {
    std::cout << "Getting view transform for angle: " << yaw << std::endl;
    tf::Transform view;
    tf::Quaternion q;
    q.setRPY(0.0f, 0.0f, -yaw + PI_F / 2); // point towards origin
    view.setRotation(q);
    view.setOrigin(tf::Vector3(CAMERA_OFFSET * sin(yaw), CAMERA_OFFSET * cos(yaw), 0.2f));
    return view;
  }

  void octreeClbk(const octomap_msgs::Octomap &msg)
  {
    received_octomap_msg = true;
    // std::cout << "Receiving octomap message... " << std::endl;
    tree = octomap_msgs::binaryMsgToMap(msg);
    octree = dynamic_cast<octomap::OcTree *>(tree);

    int visited = 0;
    int unobs = 0;
    pcl::PointXYZ p;
    pcl::PointCloud<pcl::PointXYZ> temp_data_cloud;

    for (double ix = -0.2; ix < 0.2; ix += RESOLUTION)
      for (double iy = -0.2; iy < 0.2; iy += RESOLUTION)
        for (double iz = 0.0; iz < 0.3; iz += RESOLUTION)
        {
          visited++;
          if (!octree->search(ix, iy, iz))
          {
            unobs++;
            p.x = ix;
            p.y = iy;
            p.z = iz;
            temp_data_cloud.push_back(p);
          }
        }
    // from http://ros-developer.com/2017/02/23/converting-pcl-point-cloud-to-ros-pcl-cloud-message-and-the-reverse/
    // make a new point cloud composed of only unobserved voxels
    // sensor_msgs::PointCloud2 unobs_voxel_msg;
    // pcl::toROSMsg(temp_data_cloud, unobs_voxel_msg);
    // http://docs.ros.org/jade/api/octomap_ros/html/conversions_8h.html
    // octomap::pointCloud2ToOctomap(unobs_voxel_msg, octomap_unobs_pc);
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

    // std::cout<<"Address of root node: "<<(cm.octree->getRoot())<<std::endl;
    if (!cm.nbv_calculated && cm.received_octomap_msg)
    {
      tf::Transform nbv = cm.calculateNbv();
    }
    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}