#!usr/bin/env python

import time
import sys
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from geometry_msgs.msg import Pose, Point, PoseStamped, Vector3, Vector3Stamped
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
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Grasp
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, RobotTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from grasp_execution_msgs.msg import GraspControlAction, GraspControlGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import GripperTranslation
from moveit_msgs.msg import PickupAction, PickupActionGoal, PickupGoal
from actionlib_msgs.msg import GoalID
import math
import pickle
import os
import tf
import yaml
from tempfile import TemporaryFile
import actionlib
import copy


class GraspDataCollection:
    def __init__(self, verbosity):
        self.verbose = verbosity
        self.bridge = CvBridge()
        self.total_num_trials = 1
        self.current_trial_no = 0
        self.object_name = 'cube1'
        self.planning_group_name = 'Arm'
        self.pre_grasp_height = 1.3
        self.post_grasp_height = 1.3
        self.lift_height = 1.3
        self.joint_angle_tolerance = 0.01
        self.reference_frame = 'world'
        self.model_name = 'jaco_on_table'
        self.palm_link = 'jaco_fingers_base_link'
        self.palm_link_eef = 'jaco_6_hand_limb'
        self.eef_link = 'Wrist' #'jaco_6_hand_limb'
        self.joint_to_exclude = 'base_to_jaco_on_table'
        self.joint_traj_action_topic = '/jaco/joint_trajectory_action'
        self.grasp_action_topic_jen = '/jaco/grasp_action'
        self.grasp_action_topic = '/pickup' # TODO goal or no goal?
        # reads in the joint names as a list
        self.joint_names = self.load_joint_properties()
        # dictionary {joint_name: value}
        self.joint_states = self.get_joint_states()
        self.object_height = 0.0 # [m]
        self.object_position = [0.4, 0.0, 0.76]
        self.phase = 'pre'
        self.finger_joint_angles_grasp = 0.53
        self.finger_joint_angles_ungrasp = 0.05
        self.finger_joint_names = ['jaco_finger_joint_0',
                                   'jaco_finger_joint_2', 'jaco_finger_joint_4']
        self.arm_home_state = {}
        self.hm_sub = rospy.Subscriber("/height_map_image",
                                       Image, self.hm_clbk,  queue_size=1)
        self.height_map = None  # TODO check for is None
        self.joint_states_ik_seed = self.generate_joint_states_ik_seed()

        print 'Initialization complete'

    def generate_joint_states_ik_seed(self):
        vals = {'jaco_arm_4_joint': 1.8836414733128066,
                'jaco_arm_0_joint': -4.365035046220363,
                'jaco_arm_5_joint': 4.332552336172351,
                'jaco_finger_joint_0': -4.664683020028093e-05,
                'jaco_arm_3_joint': 0.5245155869919846,
                'jaco_finger_joint_2': 0.0036287100146745743,
                'base_to_jaco_on_table': 4.460805946848723e-09,
                'jaco_finger_joint_4': -5.652861570215606e-05,
                'jaco_arm_1_joint': -2.298114280915076,
                'jaco_arm_2_joint': -2.8895313400862497}
        return vals

    def hm_clbk(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError as e:
            print(e)
        # TODO extract max and min of cv_image
        self.height_map = cv2.normalize(
            cv_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def save_arm_home_state(self):
        if self.verbose:
            print 'Saving arm home state'
        self.arm_home_state = self.joint_states

    def return_arm_home(self):
        if self.verbose:
            print 'Returning arm to home position'
        self.move_to_state(self.arm_home_state)

    def increment_current_trial_no(self):
        if self.verbose:
            print 'Moving to the next trial'
        self.current_trial_no += 1

    def get_eef_pose(self):
        # js = self.get_joint_states()
        js = self.get_ik('grasp')
        rospy.wait_for_service('/compute_fk')
        try:
            req = rospy.ServiceProxy('/compute_fk', GetPositionFK)
            header = Header()
            header.frame_id = "/" + self.reference_frame
            fk_link_names = [self.palm_link_eef]
            rs = RobotState()
            rs.joint_state.header.frame_id = "/" + self.reference_frame
            names = []
            vals = []
            for name, val in js.items():
                if name != self.joint_to_exclude:
                    names.append(name)
                    vals.append(val)
            rs.joint_state.name = names
            rs.joint_state.position = vals
            res = req(header, fk_link_names, rs)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e        
        return res.pose_stamped[0]

    def get_object_height(self):
        if self.verbose:
            print 'Getting object height'
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            req = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            res = req(self.object_name, '')
            print res.success
            height = res.pose.position.z
            if self.verbose:
                print 'Height: ', height
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        return height

    def position_object(self):
        if self.verbose:
            print 'Positioning object'
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
        if self.verbose:
            print 'Computing IK for', phase
        rospy.wait_for_service('/compute_ik')
        try:
            req = rospy.ServiceProxy('/compute_ik', GetPositionIK)
            ik = PositionIKRequest()
            rs = RobotState()
            ik.group_name = self.planning_group_name
            rs.joint_state.header.frame_id = "/" + self.reference_frame
            names = []
            vals = []
            for name, val in self.joint_states_ik_seed.items():
                if name != self.joint_to_exclude:
                    names.append(name)
                    vals.append(val)
                    # vals.append(val[0])
            rs.joint_state.name = names
            rs.joint_state.position = vals
            ik.ik_link_name = self.palm_link
            ik.robot_state = rs
            ik.pose_stamped = self.generate_grasp_pose()
            ik.timeout.secs = 3.0  # [s]
            ik.attempts = 100
            # print '\nIK message:', ik
            res = req(ik)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        js = {}
        for name, val in zip(res.solution.joint_state.name, res.solution.joint_state.position):
            js[name] = val
        return js

    def load_joint_properties(self):
        if self.verbose:
            print 'Loading joint properties'
        rospy.wait_for_service('/gazebo/get_model_properties')
        try:
            req = rospy.ServiceProxy(
                '/gazebo/get_model_properties', GetModelProperties)
            res = req(self.model_name)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        return res.joint_names

    def get_joint_states(self):
        rospy.wait_for_service('/gazebo/get_joint_properties')
        joint_states = {}
        try:
            req = rospy.ServiceProxy(
                '/gazebo/get_joint_properties', GetJointProperties)
            for j in self.joint_names:
                state = req(j).position[0]
                joint_states[j] = state
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        print 'Finished retrieving joint states'
        return joint_states

    def generate_grasp_pose(self):
        if self.verbose:
            print 'Generating grasp pose'
        sig_pos = 0.0001  # [m] std dev for position
        x = np.random.normal(self.object_position[0], sig_pos)
        y = np.random.normal(self.object_position[1], sig_pos)
        th = 0  # np.random.uniform(0.0, math.pi * 2.0)
        # https://www.programcreek.com/python/example/70252/geometry_msgs.msg.PoseStamped
        ps = PoseStamped()
        ps.header.frame_id = "/" + self.reference_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = self.get_object_height() + 0.25
        # if self.phase == 'pre':
        #     ps.pose.position.z = self.pre_grasp_height
        # elif self.phase == 'grasp':
        #     ps.pose.position.z = self.post_grasp_height
        # else:
        #     ps.pose.position.z = self.lift_height
        q = tf.transformations.quaternion_from_euler(
            math.pi, 0, th, axes='sxyz')
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        return ps

    def move_to_state(self, joint_states):
        if self.verbose:
            print 'Moving to next state'
        mpr = MotionPlanRequest()
        con = Constraints()
        for name, val in joint_states.items():
            if name != self.joint_to_exclude:
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = val
                jc.tolerance_above = self.joint_angle_tolerance
                jc.tolerance_below = self.joint_angle_tolerance
                jc.weight = 1.0
                con.joint_constraints.append(jc)
        mpr.goal_constraints = [con]
        mpr.group_name = self.planning_group_name
        mpr.allowed_planning_time = 3.0  # [s]
        try:
            req = rospy.ServiceProxy(
                '/plan_kinematic_path', GetMotionPlan)
            res = req(mpr)
            traj = res.motion_plan_response.trajectory
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        client = actionlib.SimpleActionClient(
            self.joint_traj_action_topic, FollowJointTrajectoryAction)
        client.wait_for_server()

        # http://docs.ros.org/diamondback/api/control_msgs/html/index-msg.html
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj.joint_trajectory
        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(15.0))
        # http://docs.ros.org/api/actionlib/html/classactionlib_1_1simple__action__client_1_1SimpleActionClient.html
        # print client.get_state()

    def execute_grasp_action(self, action):
        js_closed = {}
        js_open = {}

        if self.verbose:
            print 'Executing grasp action:', action
        for finger_joint_name in self.finger_joint_names:
            angle_closed = self.finger_joint_angles_grasp
            angle_open = self.finger_joint_angles_ungrasp
            js_closed[finger_joint_name] = angle_closed
            js_open[finger_joint_name] = angle_open
        
        pgpost = JointTrajectory()
        gpost = JointTrajectory()

        jt_grasp = JointTrajectory()
        for (name_cl, val_cl), (name_op, val_op) in zip(js_closed.items(), js_open.items()):
            pgpost.joint_names.append(name_cl)
            gpost.joint_names.append(name_cl)
            jtp_c = JointTrajectoryPoint()
            jtp_c.positions.append(val_cl)
            jtp_c.time_from_start.secs = 2 # [s]
            jtp_o = JointTrajectoryPoint()
            jtp_o.positions.append(val_op)
            jtp_o.time_from_start.secs = 2 # [s]
            pgpost.points.append(jtp_c)
            gpost.points.append(jtp_o)

        if action == 'close': # reverse the prior order
            pgpost, gpost = gpost, pgpost # TODO verify that these have been properly switched!

        grasp = Grasp()

        # gp = PoseStamped()
        gp = self.get_eef_pose()
        print "eef pose: ", gp
        # print self.generate_grasp_pose()

        # rospy.sleep(20)

        pregrapp = GripperTranslation() # check wrt which axis
        postgrretr = GripperTranslation()
        postplretr = GripperTranslation()

        v3s = Vector3Stamped()
        v3s.header.frame_id = "/" + self.reference_frame
        v3s.vector.z = -1.0
        pregrapp.direction = v3s
        pregrapp.desired_distance = 0.2 # [m]
        pregrapp.min_distance = 0.02 # [m]

        v3s = Vector3Stamped()
        v3s.header.frame_id = "/" + self.reference_frame
        v3s.vector.z = 1.0
        postgrretr.direction = v3s
        postgrretr.desired_distance = 0.2 # [m]
        postgrretr.min_distance = 0.02 # [m]        

        v3s = Vector3Stamped()
        v3s.header.frame_id = "/" + self.palm_link
        v3s.vector.z = 1.0
        postplretr.direction = v3s
        postplretr.desired_distance = 0.3 # [m]
        postplretr.min_distance = 0.02 # [m]    

        grasp.pre_grasp_posture = pgpost
        grasp.grasp_posture = gpost
        grasp.grasp_pose = gp
        grasp.grasp_quality = 0.5
        grasp.pre_grasp_approach = pregrapp
        grasp.post_grasp_retreat = postgrretr
        # grasp.post_place_retreat = postplretr
        grasp.max_contact_force = -1
        grasp.allowed_touch_objects = [self.object_name]        

        goal = PickupGoal()
        # goal.target_name = self.object_name
        goal.group_name = self.planning_group_name
        goal.end_effector = self.eef_link
        goal.possible_grasps = [grasp]
        goal.allowed_planning_time = 3.0 # [s]

        print goal

        client = actionlib.SimpleActionClient(
            self.grasp_action_topic, PickupAction)
        client.wait_for_server()
        if self.verbose:
            print 'Executing grasp action:', action        
        # print goal
        client.send_goal(goal)

    def save(self, height_map, ik_pre, height):
        if self.verbose:
            print 'Saving data'
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
        # outfile = TemporaryFile()
        np.savez('output', height_map=height_map, height=height)


def main(args):
    rospy.init_node('grasp_data_collection')
    gdc = GraspDataCollection(args[1])
    gdc.save_arm_home_state()
    while gdc.current_trial_no < gdc.total_num_trials:
        gdc.position_object()
        height_map = gdc.height_map # TODO make not None
        # ik_pre = gdc.get_ik('pre')
        # gdc.move_to_state(ik_pre)
        # gdc.move_to_state(gdc.get_ik('grasp'))
        # raw_input()
        gdc.execute_grasp_action('close')
        # gdc.move_to_state(gdc.get_ik('lift'))
        # height = gdc.get_object_height()
        # gdc.execute_grasp_action('open')
        # gdc.save(height_map, ik_pre, height)
        # gdc.increment_current_trial_no()
        # gdc.return_arm_home()

        rospy.spin()
        if rospy.is_shutdown():
            break


if __name__ == '__main__':
    main(sys.argv)
