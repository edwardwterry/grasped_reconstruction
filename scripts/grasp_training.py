#!usr/bin/env python

import time
import sys
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from geometry_msgs.msg import Pose, Point, PoseStamped
from grid_map_msgs.msg import GridMap
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetModelProperties
from gazebo_msgs.srv import GetJointProperties
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
# from sensor_msgs.msg import JointState
import math
import pickle
import os
import tf
import yaml
from tempfile import TemporaryFile


class GraspDataCollection:
    def __init__(self):
        self.total_num_trials = 100
        self.current_trial_no = 0
        self.object_name = 'cube1'
        self.planning_group_name = 'Arm'
        self.pre_grasp_height = 0.85
        self.post_grasp_height = 0.77
        self.lift_height = 1.0
        self.joint_angle_tolerance = 0.0001
        self.reference_frame = 'robot_base'
        self.model_name = 'jaco_on_table'
        self.palm_link = 'jaco_fingers_base_link'
        self.joint_to_exclude = 'base_to_jaco_on_table'
        # reads in the joint names as a list
        self.joint_names = self.load_joint_properties()
        # dictionary {joint_name: value}
        self.joint_states = self.get_joint_states()
        self.object_height = 0.0
        self.object_position = [0.2, 0.0, 0.76]
        self.phase = 'pre'
        self.finger_joint_angles_grasp = 0.53
        self.finger_joint_angles_ungrasp = 0.05
        self.finger_joint_names = ['jaco_finger_joint_0',
                                   'jaco_finger_joint_2', 'jaco_finger_joint_4']
        self.arm_home_state = {}
        self.hm_sub = rospy.Subscriber("/height_map_image",
                                       Image, self.hm_clbk,  queue_size=1)
        self.mpr_pub = rospy.Publisher("/move_group/motion_plan_request", MotionPlanRequest, queue_size=1)
        self.height_map = None

    def hm_clbk(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError as e:
            print(e)
        # TODO extract max and min of cv_image
        self.height_map = cv2.normalize(
            cv_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def save_arm_home_state(self):
        self.arm_home_state = self.get_joint_states()

    def return_arm_home(self):
        self.move_to_state(self.arm_home_state)

    def increment_current_trial_no(self):
        self.current_trial_no += 1

    def get_object_height(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            req = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            res = req(self.object_name)[0]
            height = res.pose.position.z
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def position_object(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            req = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            ms = ModelState()
            ms.model_name = self.object_name
            ms.pose.position.x = self.object_position[0]
            ms.pose.position.y = self.object_position[1]
            ms.pose.position.z = self.object_position[2]
            ms.reference_frame = self.reference_frame
            res = req(ms)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def get_ik(self, phase):
        self.phase = phase
        rospy.wait_for_service('/compute_ik')
        try:
            req = rospy.ServiceProxy('/compute_ik', GetPositionIK)
            ik = PositionIKRequest()
            rs = RobotState()
            ik.group_name = self.planning_group_name
            rs.joint_state.header.frame_id = self.reference_frame
            names = []
            vals = []
            for name, val in self.joint_states.items():
                if name != self.joint_to_exclude:
                    names.append(name)
                    vals.append(val[0])
            rs.joint_state.name = names
            rs.joint_state.position = vals
            ik.ik_link_name = self.palm_link
            ik.robot_state = rs
            ik.pose_stamped = self.generate_grasp_pose()
            res = req(ik)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def load_joint_properties(self):
        rospy.wait_for_service('/gazebo/get_model_properties')
        try:
            req = rospy.ServiceProxy(
                '/gazebo/get_model_properties', GetModelProperties)
            res = req(self.model_name)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        return res.joint_names

    def get_single_joint_state(self, joint_name):
        rospy.wait_for_service('/gazebo/get_joint_properties')
        try:
            req = rospy.ServiceProxy(
                '/gazebo/get_joint_properties', GetJointProperties)
            state = req(joint_name).position
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        return state

    def get_joint_states(self):
        joint_states = {}
        for j in self.joint_names:
            joint_states[j] = self.get_single_joint_state(j)
        return joint_states

    def generate_grasp_pose(self):
        sig_pos = 0.2  # [m] std dev for position
        x = np.random.normal(self.object_position[0], sig_pos)
        y = np.random.normal(self.object_position[1], sig_pos)
        th = np.random.uniform(0.0, math.pi * 2.0)
        # https://www.programcreek.com/python/example/70252/geometry_msgs.msg.PoseStamped
        ps = PoseStamped()
        ps.header.frame_id = self.reference_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        if self.phase == 'pre':
            ps.pose.position.z = self.pre_grasp_height
        elif self.phase == 'grasp':
            ps.pose.position.z = self.post_grasp_height
        else:
            ps.pose.position.z = self.lift_height
        q = tf.transformations.quaternion_from_euler(0, 0, th)
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        return ps

    def move_to_state(self, joint_states):
        mpr = MotionPlanRequest()
        con = Constraints()
        for name, val in self.joint_states.items():
            if name != self.joint_to_exclude:
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = val
                jc.tolerance_above = self.joint_angle_tolerance
                jc.tolerance_below = self.joint_angle_tolerance
                con.joint_constraints.append(jc)
        mpr.goal_constraints = con
        mpr.group_name = self.planning_group_name
        mpr.allowed_planning_time = 3.0 # [s]
        self.mpr_pub.publish(mpr)

    def execute_grasp_action(self, action):
        if action == 'close':
            angle = self.finger_joint_angles_grasp
        elif action == 'open':
            angle = self.finger_joint_angles_grasp
        joint_states = self.get_joint_states()
        for finger_joint_name in self.finger_joint_names:
            joint_states[finger_joint_name] = angle
        self.move_to_state

    def save(self, height_map, ik_pre, height):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
        outfile = TemporaryFile()
        np.savez(outfile, height_map=height_map, height=height)


def main(args):
    rospy.init_node('grasp_data_collection')
    gdc = GraspDataCollection()
    gdc.save_arm_home_state()
    while gdc.current_trial_no < gdc.total_num_trials:
        gdc.position_object()
        height_map = gdc.height_map
        ik_pre = gdc.get_ik('pre')
        gdc.move_to_state(ik_pre)
        gdc.move_to_state(gdc.get_ik('grasp'))
        gdc.execute_grasp_action('close')
        gdc.move_to_state(gdc.get_ik('lift'))
        # wait til it gets there, TODO put elsewhere as appropriate
        rospy.sleep(3)
        height = gdc.get_object_height()
        gdc.execute_grasp_action('open')
        gdc.save(height_map, ik_pre, height)
        gdc.increment_current_trial_no()
        gdc.return_arm_home()

        rospy.spin()
        if rospy.is_shutdown():
            break


if __name__ == '__main__':
    main(sys.argv)
