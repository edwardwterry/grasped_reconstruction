#!usr/bin/env python

import time
import sys
from sensor_msgs.msg import Image
import rospy
import roslib
from scipy.ndimage import filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from matplotlib import pyplot as plt
from geometry_msgs.msg import Pose, Point
from grid_map_msgs.msg import GridMap
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelProperties
from gazebo_msgs.msg import JointProperties
from moveit_msgs.msg import GetPositionIK
from moveit_msgs.msg import GetPositionFK
import math
import pickle
import os
import tf
import yaml


class GraspDataCollection:
    def __init__(self):
        self.total_num_trials = 100
        self.current_trial_no = 0
        self.object_name = 'cube1'
        self.pre_grasp_height = 0.85
        self.reference_frame = 'robot_base'
        self.model_name = 'jaco_on_table'
        self.palm_link = 'jaco_fingers_base_link'
        self.joint_names = self.load_joint_properties()
        self.joint_states = self.get_joint_states()
        self.object_height = 0.0

    def increment_current_trial_no(self):
        self.current_trial_no += 1

    def get_object_height(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            req = rospy.ServiceProxy('/gazebo/get_model_state', ModelState)
            res = req(self.object_name)
            height = res.pose.position.z
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def position_object(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            req = rospy.ServiceProxy('/gazebo/set_model_state', ModelState)
            ms = ModelState()
            ms.model_state.model_name = self.object_name
            ms.model_state.pose.position.x = 0.2
            ms.model_state.pose.position.x = 0.0
            ms.model_state.pose.position.x = 0.76
            ms.model_state.reference_frame = self.reference_frame
            res = req(ms)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e        

    def get_ik(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            req = rospy.ServiceProxy('/compute_ik', GetPositionIK)
            ik = GetPositionIK()
            ik.group_name = 'Arm'
            ik.robot_state.frame_id = self.reference_frame
            names = []
            vals = []
            for name, val in zip(self.joint_names, self.joint_states):
                names.append(name)
                vals.append(val)
            ik.robot_state.name = names
            ik.robot_state.position = vals
            ik.ik_link_name = self.palm_link
            
            res = req(ik)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def get_robot_state(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            req = rospy.ServiceProxy('/compute_ik', GetPositionIK)
            ik = GetPositionIK()
            ik.group_name = 'Arm'
            ik.robot_state.frame_id = self.reference_frame
            res = req(ik)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e     

    def load_joint_properties(self):
        rospy.wait_for_service('/gazebo/get_model_properties')
        try:
            req = rospy.ServiceProxy('/gazebo/get_model_properties', ModelProperties)
            res = req(self.model_name)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e   
        return res.joint_names

    def get_joint_states(self):
        rospy.wait_for_service('/gazebo/get_joint_properties')
        try:
            req = rospy.ServiceProxy('/gazebo/get_joint_properties', JointProperties)
            for joint in self.joint_names:
                self.joint_states[joint] = req(joint)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e   
    
def main(args):
    rospy.init_node('grasp_training')
    gdc = GraspDataCollection()
    while gdc.current_trial_no < gdc.total_num_trials:
        # position the object
        gdc.position_object()
        # save the height map
        # calculate the x, y, th for a grasp
        # get ik for pre-grasp position

        # move to pre-grasp position
        # get ik for grasp position
        # save ik for grasp position
        # move to grasp position
        # close fingers
        # get ik for raised position
        # move to raised position
        # wait for 3 seconds
        # get the object's height
        # save the object's height
        # open the fingers
        # export the saved variables
        # increment the counter
        gdc.increment_current_trial_no()
        rospy.spin()
        if rospy.is_shutdown():
            break

if __name__ == '__main__':
    main(sys.argv)
