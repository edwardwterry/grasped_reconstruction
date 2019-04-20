#!usr/bin/env python

import time
import sys
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from geometry_msgs.msg import Pose, Point, PoseStamped, Vector3, Vector3Stamped, TransformStamped
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
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Grasp
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, RobotTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from grasp_execution_msgs.msg import GraspControlAction, GraspControlGoal, GraspAction, GraspGoal, GraspData
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
from tf import TransformListener, TransformBroadcaster

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
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
        self.orig_bb = 'orig_bb'
        self.eef_link = 'Wrist'  # 'jaco_6_hand_limb'
        self.lens_link = 'lens_link'
        self.nbv = 'nbv'
        self.joints_to_exclude = ['base_to_jaco_on_table', 'jaco_finger_joint_0',
                                   'jaco_finger_joint_2', 'jaco_finger_joint_4']
        self.joint_traj_action_topic = '/jaco/joint_trajectory_action'
        self.grasp_action_topic_jen = '/jaco/grasp_execution/grasp'
        self.grasp_action_topic = '/pickup'  # TODO goal or no goal?
        # reads in the joint names as a list
        self.joint_names = self.load_joint_properties()
        # dictionary {joint_name: value}
        self.joint_states = self.get_joint_states()
        self.object_height = 0.0  # [m]
        self.object_position = [0.2, 0.0, 0.76]
        self.offset_by_phase = {'pre': 0.1, 'grasp': 0.0} #pre was 0.12
        self.phase = 'pre'
        self.finger_joint_angles_grasp = 0.5
        self.finger_joint_angles_ungrasp = 0.1
        self.finger_joint_names = ['jaco_finger_joint_0',
                                   'jaco_finger_joint_2', 'jaco_finger_joint_4']
        self.arm_home_state = {}
        self.hm_sub = rospy.Subscriber("/height_map_image",
                                       Image, self.hm_clbk,  queue_size=1)
        self.height_map = None  # TODO check for is None
        self.joint_states_ik_seed = self.generate_joint_states_ik_seed()
        self.joint_states_presentation_pose = self.generate_joint_states_presentation_pose()
        self.finger_pub = rospy.Publisher('/jaco/joint_control', JointState, queue_size=1)
        self.eef_pose_at_grasp = Pose()
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

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

    def generate_joint_states_presentation_pose(self):
        vals = {'jaco_arm_4_joint': 1.5841377943674804,
                'jaco_arm_0_joint': -1.39408873308499,
                'jaco_arm_5_joint': -1.4472548035118011,
                # 'jaco_finger_joint_0': 0.53,
                'jaco_arm_3_joint': -2.9214876031823875,
                # 'jaco_finger_joint_2': 0.53,
                'base_to_jaco_on_table': 4.460805946848723e-09,
                # 'jaco_finger_joint_4': 0.53,
                'jaco_arm_1_joint': -0.5458189934369335,
                'jaco_arm_2_joint': -0.40208614052374525}
        return vals

    def generate_nbv_pose(self):
        # https://answers.ros.org/question/133331/multiply-two-tf-transforms-converted-to-4x4-matrices-in-python/

        # E_T_C
        self.tf_listener.waitForTransform("/" + self.object_name, "/" + self.palm_link, rospy.Time(0), rospy.Duration(3.0))
        E_T_C = self.tf_listener.lookupTransform("/" + self.object_name, "/" + self.palm_link, rospy.Time(0))
        E_T_C_t = tf.transformations.translation_matrix(E_T_C[0])
        E_T_C_r = tf.transformations.quaternion_matrix(E_T_C[1])
        E_T_C_m = np.dot(E_T_C_t, E_T_C_r)
        print 'E_T_C_m\n', E_T_C_m

        # C_T_L
        C_T_L = TransformStamped()
        C_T_L.header.frame_id = "/" + self.lens_link
        C_T_L.child_frame_id = "/" + self.object_name

        # # N_T_O
        self.tf_listener.waitForTransform("/" + self.orig_bb, "/" + self.nbv, rospy.Time(0), rospy.Duration(3.0))
        N_T_O = self.tf_listener.lookupTransform("/" + self.orig_bb, "/" + self.nbv, rospy.Time(0))
        N_T_O_t = tf.transformations.translation_matrix(N_T_O[0])
        N_T_O_r = tf.transformations.quaternion_matrix(N_T_O[1])
        # N_T_O_r = tf.transformations.quaternion_matrix([0, 0, 0, 1])
        N_T_O_m = np.dot(N_T_O_t, N_T_O_r)
        print 'N_T_O_m\n', N_T_O_m
         
        # # C_T_L = C_T_P * P_T_L
        # prime
        # C_T_P_t = tf.transformations.translation_matrix([0, 0, 0, 1])
        # C_T_P_t = tf.transformations.quaternion_matrix([-0.5, -0.5, 0.5, 0.5])
        # C_T_P_m = np.dot(C_T_P_t, C_T_P_t)

        # P_T_L_t = tf.transformations.translation_matrix([0.0, 0, 0.3, 1.0])
        # P_T_L_r = tf.transformations.quaternion_matrix(N_T_O[1])
        # # P_T_L_r = tf.transformations.quaternion_matrix([0, 0, 0, 1])
        # P_T_L_m = np.dot(P_T_L_t, P_T_L_r)
        # C_T_L_m = np.dot(P_T_L_m, C_T_P_m)
        # print 'C_T_L_m\n', C_T_L_m
        # end prime

        # alternative:
        C_T_P_t = tf.transformations.translation_matrix([0, 0, 0, 1])
        C_T_P_r = tf.transformations.quaternion_matrix(N_T_O[1])
        C_T_P_m = np.dot(C_T_P_t, C_T_P_r)
        print 'C_T_P_m\n', C_T_P_m

        P_T_L_t = tf.transformations.translation_matrix([0.0, 0, 0.3, 1.0])
        P_T_L_r = tf.transformations.quaternion_matrix([-0.5, -0.5, 0.5, -0.5]) # point nbv x axis towards camera
        # P_T_L_r = tf.transformations.quaternion_matrix([0, 0, 0, 1])
        P_T_L_m = np.dot(P_T_L_t, P_T_L_r)
        print 'P_T_L_m\n', P_T_L_m
        C_T_L_m = np.dot(P_T_L_m, C_T_P_m)
        print 'C_T_L_m\n', C_T_L_m
        # end alternative

        # L_T_O 
        self.tf_listener.waitForTransform("/" + self.orig_bb, "/" + self.lens_link, rospy.Time(0), rospy.Duration(3.0))
        L_T_O = self.tf_listener.lookupTransform("/" + self.orig_bb, "/" + self.lens_link, rospy.Time(0))
        L_T_O_t = tf.transformations.translation_matrix(L_T_O[0])
        L_T_O_r = tf.transformations.quaternion_matrix(L_T_O[1])
        L_T_O_m = np.dot(L_T_O_t, L_T_O_r)
        print 'L_T_O_m\n', L_T_O_m

        # O_T_W 
        self.tf_listener.waitForTransform("/" + self.reference_frame, "/" + self.orig_bb, rospy.Time(0), rospy.Duration(3.0))
        O_T_W = self.tf_listener.lookupTransform("/" + self.reference_frame, "/" + self.orig_bb, rospy.Time(0))
        O_T_W_t = tf.transformations.translation_matrix(O_T_W[0])
        O_T_W_r = tf.transformations.quaternion_matrix(O_T_W[1])
        O_T_W_m = np.dot(O_T_W_t, O_T_W_r)
        print 'O_T_W_m\n', O_T_W_m

        # E_T_W
        L_T_W_m = np.dot(O_T_W_m, L_T_O_m)
        print 'L_T_W_m\n', L_T_W_m
        C_T_W_m = np.dot(L_T_W_m, C_T_L_m)
        print 'C_T_W_m\n', C_T_W_m
        E_T_W_m = np.dot(C_T_W_m, E_T_C_m)
        E_T_W_t = tf.transformations.translation_from_matrix(E_T_W_m)
        # E_T_W_r = tf.transformations.quaternion_from_matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        E_T_W_r = tf.transformations.quaternion_from_matrix(E_T_W_m)

        print 'E_T_W_m\n', E_T_W_m

        ps = PoseStamped()
        ps.header.frame_id = "/" + self.reference_frame
        ps.pose.position.x = E_T_W_t[0]#.32455 # E_T_W_t[0]
        ps.pose.position.y = E_T_W_t[1]#0.10394 # E_T_W_t[1]
        ps.pose.position.z = E_T_W_t[2]#1.0798 # E_T_W_t[2]
        ps.pose.orientation.x = E_T_W_r[0]#-0.541971 #E_T_W_r[0]
        ps.pose.orientation.y = E_T_W_r[1]#0.32541 #E_T_W_r[1]
        ps.pose.orientation.z = E_T_W_r[2]#0.60627 #E_T_W_r[2]
        ps.pose.orientation.w = E_T_W_r[3]#0.48251 #E_T_W_r[3]
        return self.get_ik('present', ps)

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
                if name not in self.joints_to_exclude:
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
            # if self.verbose:
                # print 'Height: ', height
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

    def get_ik(self, phase, eef_pose=None):
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
            if phase == 'present':
                badly_named_var = self.joint_states_presentation_pose
            else:
                badly_named_var = self.joint_states_ik_seed
            for name, val in badly_named_var.items():
                if name not in self.joints_to_exclude:
                    names.append(name)
                    vals.append(val)
                    # vals.append(val[0])
            rs.joint_state.name = names
            rs.joint_state.position = vals
            ik.ik_link_name = self.palm_link
            ik.robot_state = rs
            if eef_pose is None: # just for the grasp
                ik.pose_stamped = self.generate_grasp_pose(
                    self.offset_by_phase[phase])
            else: # for the presentation pose
                ik.pose_stamped = eef_pose
            ik.timeout.secs = 3.0  # [s]
            ik.attempts = 10
            print '\nIK message:', ik
            res = req(ik)
            print res
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        js = {}
        for name, val in zip(res.solution.joint_state.name, res.solution.joint_state.position):
            js[name] = val
        print js
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

    def generate_grasp_pose(self, offset):
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
        ps.pose.position.z = self.get_object_height() + 0.18 + offset  # 0.24 0.18good 0.4
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
            if name not in self.joints_to_exclude:
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
        # print mpr
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

    def move_from_pregrasp_to_grasp(self, second):
        if self.verbose:
            print 'Moving from pregrasp to grasp'
        # Cartesian path
        header = Header()
        header.frame_id = "/" + self.reference_frame
        start_state = RobotState()
        start_state.joint_state.header.frame_id = "/" + self.reference_frame
        names = []
        vals = []
        js = self.get_joint_states()
        for name, val in js.items():
            if name not in self.joints_to_exclude:
                names.append(name)
                vals.append(val)
        start_state.joint_state.name = names
        start_state.joint_state.position = vals
        group_name = self.planning_group_name
        link_name = self.palm_link_eef
        wp_start = self.get_eef_pose()
        wp_end = self.get_eef_pose()
        if second:
            wp_end.pose.position.z = 0.9
        waypoints = [wp_start.pose, wp_end.pose]
        # print waypoints
        jump_threshold = 10
        max_step = 0.02
        avoid_collisions = True
        path_constraints = Constraints()
        try:
            req = rospy.ServiceProxy(
                '/compute_cartesian_path', GetCartesianPath)
            res = req(header, start_state, group_name, link_name, waypoints,
                      max_step, jump_threshold, avoid_collisions, path_constraints)
            traj = res.solution
            # print res
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        client = actionlib.SimpleActionClient(
            self.joint_traj_action_topic, FollowJointTrajectoryAction)
        client.wait_for_server()

        # http://docs.ros.org/diamondback/api/control_msgs/html/index-msg.html
        print "Moving to cartesian goal"
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj.joint_trajectory
        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(15.0))
        # http://docs.ros.org/api/actionlib/html/classactionlib_1_1simple__action__client_1_1SimpleActionClient.html
        # print client.get_state()

    def move_from_grasp_to_raised(self):
        if self.verbose:
            print 'Moving from grasp to raised'
        # Cartesian path
        header = Header()
        header.frame_id = "/" + self.reference_frame
        start_state = RobotState()
        start_state.joint_state.header.frame_id = "/" + self.reference_frame
        names = []
        vals = []
        js = self.get_joint_states()
        for name, val in js.items():
            if name not in self.joints_to_exclude:
                names.append(name)
                vals.append(val)
        start_state.joint_state.name = names
        start_state.joint_state.position = vals
        group_name = self.planning_group_name
        link_name = self.palm_link_eef
        wp_start = self.get_eef_pose()
        wp_end = self.get_eef_pose()
        wp_end.pose.position.z = 1.2
        waypoints = [wp_start.pose, wp_end.pose]
        # print waypoints
        jump_threshold = 10
        max_step = 0.02
        avoid_collisions = False
        path_constraints = Constraints()
        try:
            req = rospy.ServiceProxy(
                '/compute_cartesian_path', GetCartesianPath)
            res = req(header, start_state, group_name, link_name, waypoints,
                      max_step, jump_threshold, avoid_collisions, path_constraints)
            traj = res.solution
            # print res
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        client = actionlib.SimpleActionClient(
            self.joint_traj_action_topic, FollowJointTrajectoryAction)
        client.wait_for_server()

        # http://docs.ros.org/diamondback/api/control_msgs/html/index-msg.html
        print "Moving to raised position"
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj.joint_trajectory
        client.send_goal(goal)
        client.wait_for_result(rospy.Duration.from_sec(15.0))
        # http://docs.ros.org/api/actionlib/html/classactionlib_1_1simple__action__client_1_1SimpleActionClient.html
        # print client.get_state()        

    def actuate_fingers(self, action):
        js = JointState()
        js.name = self.finger_joint_names
        if action == 'close':
            js.position = [self.finger_joint_angles_grasp] * 3
        elif action == 'open':
            js.position = [self.finger_joint_angles_ungrasp] * 3
        self.finger_pub.publish(js)
        rospy.sleep(3)
    
    def save_eef_pose_at_grasp(self):
        self.tf_listener.waitForTransform("/" + self.reference_frame, "/" + self.palm_link_eef, rospy.Time(0), rospy.Duration(4.0))
        tf = self.tf_listener.lookupTransform("/" + self.reference_frame, "/" + self.palm_link_eef, rospy.Time(0))

        # print trans, rot

        self.eef_pose_at_grasp.position.x = trans[0]
        self.eef_pose_at_grasp.position.y = trans[1]
        self.eef_pose_at_grasp.position.z = trans[2]
        self.eef_pose_at_grasp.orientation.x = rot[0]
        self.eef_pose_at_grasp.orientation.y = rot[1]
        self.eef_pose_at_grasp.orientation.z = rot[2]
        self.eef_pose_at_grasp.orientation.w = rot[3]

        # self.tf_broadcaster.sendTransform(tf)
        # print "Sent transform!"
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
        # gdc.save_eef_pose_at_grasp()
        gdc.position_object()
        # height_map = gdc.height_map  # TODO make not None
        # gdc.pickup()
        # gdc.execute_grasp_action('open')
        gdc.move_to_state(gdc.get_ik('pre'))
        gdc.move_from_pregrasp_to_grasp(False)
        # gdc.save_eef_pose_at_grasp()
        gdc.actuate_fingers('close')
        gdc.move_from_grasp_to_raised()
        # gdc.move_to_state(gdc.generate_joint_states_presentation_pose())
        gdc.move_to_state(gdc.generate_nbv_pose())
        # gdc.move_to_state(gdc.get_ik('pre'))
        # gdc.move_from_pregrasp_to_grasp(True)
        # gdc.actuate_fingers('open')
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
