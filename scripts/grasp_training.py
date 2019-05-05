#!usr/bin/env python

import time
import sys
from sensor_msgs.msg import Image
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from geometry_msgs.msg import Pose, Point, PoseStamped, Vector3, Vector3Stamped, TransformStamped, PoseArray, Transform
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
from moveit_msgs.msg import Grasp, PlanningScene, AllowedCollisionMatrix, AttachedCollisionObject, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, RobotTrajectory
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String, Bool, Int32
from grid_map_msgs.msg import GridMap
from grasped_reconstruction.srv import *
from grasp_execution_msgs.msg import GraspControlAction, GraspControlGoal, GraspAction, GraspGoal, GraspData
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import GripperTranslation
from moveit_msgs.msg import PickupAction, PickupActionGoal, PickupGoal
from actionlib_msgs.msg import GoalID
from std_srvs.srv import Empty
import math
import pickle
import os
import tf
import yaml
from tempfile import TemporaryFile
import actionlib
import copy
from tf import TransformListener, TransformBroadcaster
import tf2_ros
import imutils

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class GraspDataCollection:
    def __init__(self, verbosity):
        self.verbose = verbosity
        self.bridge = CvBridge()
        self.total_num_trials = 2
        self.current_trial_no = 0
        self.object_name = 'obj'  # change if  a new obj type
        # self.object_name = 'cube1'  # change if  a new obj type
        self.planning_group_name = 'Arm'
        self.pre_grasp_height = 1.3
        self.post_grasp_height = 1.3
        self.lift_height = 1.3
        self.joint_angle_tolerance = 0.01 # was 0.005
        self.hand_height_offset = 0.26  # 0.18 for cube, 0.20? for objects
        self.reference_frame = 'world'
        self.model_name = 'jaco_on_table'
        self.palm_link = 'jaco_fingers_base_link'
        self.palm_link_eef = 'jaco_6_hand_limb'
        self.orig_bb = 'orig_bb'
        self.orig_obj = 'orig_obj'
        self.eef_link = 'Wrist'  # 'jaco_6_hand_limb'
        self.lens_link = 'lens_link'
        self.nbv = 'nbv'
        self.nbv_tf = Transform()
        self.grasp_eef_pose_to_achieve_nbv = PoseStamped()
        # self.joints_to_exclude = ['base_to_jaco_on_table', 'jaco_finger_joint_0', # was working for cube
        #                           'jaco_finger_joint_2', 'jaco_finger_joint_4']
        self.joints_to_exclude = ['base_to_jaco_on_table']
        self.joint_traj_action_topic = '/jaco/joint_trajectory_action'
        self.grasp_action_topic_jen = '/jaco/grasp_execution/grasp'
        self.grasp_action_topic = '/pickup'  # TODO goal or no goal?
        self.finger_links_allowed_to_touch = ['jaco_8_finger_index', 'jaco_8_finger_pinkie',
                                              'jaco_8_finger_thumb', 'jaco_9_finger_index_tip', 'jaco_9_finger_pinkie_tip', 'jaco_9_finger_thumb_tip']
        # reads in the joint names as a list
        self.joint_names = self.load_joint_properties()
        self.finger_joint_states_at_grasp = {}
        # dictionary {joint_name: value}
        self.joint_states = self.get_joint_states()
        self.object_height = 0.0  # [m]
        self.object_position = [0.2, 0.0, 0.76]  # change if new object!!!
        self.offset_by_phase = {'pre': 0.1,
                                'grasp': 0.0, 'nbv_eval': 0.0}  # pre was 0.12
        self.phase = 'pre'
        self.finger_joint_angles_grasp = 0.9  # was 0.5
        self.finger_joint_angles_ungrasp = 0.1
        self.finger_joint_names = ['jaco_finger_joint_0',
                                   'jaco_finger_joint_2', 'jaco_finger_joint_4']
        self.arm_home_state = {}
        self.hm_sub = rospy.Subscriber("/height_map_image",
                                       Image, self.hm_clbk,  queue_size=1)
        self.hm_image = None  # TODO check for is None
        self.joint_states_ik_seed = self.generate_joint_states_ik_seed()
        self.joint_states_presentation_pose = self.generate_joint_states_presentation_pose()
        self.finger_pub = rospy.Publisher(
            '/jaco/joint_control', JointState, queue_size=1)
        self.gm_sub = rospy.Subscriber(
            "/elevation_mapping/elevation_map", GridMap, self.gm_clbk, queue_size=1)
        self.joint_states_at_pre = JointState()
        self.joint_states_at_raised = JointState()
        self.eef_height_at_grasp = 0.0
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()
        self.save_current_eef_pose_pub = rospy.Publisher(
            '/save_current_eef_pose', String, queue_size=1)
        # self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # self.tf_save_request_server = rospy.Service('tf_save_request', )
        self.grasp_theta = 0.0
        self.num_roll_angle_options = 8
        self.present_roll_angle_options = [
            x * 2.0 * math.pi / self.num_roll_angle_options for x in range(self.num_roll_angle_options)]

        self.num_nbvs_to_request = 5
        self.top_ranked_nbvs = []

        # gm params
        self.gm_res = 0.01  # [m]
        self.gm_ctr_x = self.object_position[0]  # [m]
        self.gm_ctr_y = self.object_position[1]  # [m]
        self.hm_scale_factor = 1.0
        self.sample_point_spacing = 0.04  # [m]
        self.gm_received = False
        self.hm_received = False
        self.num_grasp_candidates_to_generate = 2
        self.num_grasp_angles_to_try = 4
        self.hm_image = None
        self.nr = 0
        self.nc = 0
        self.basic_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                              (0, 1), (1, -1), (1, 0), (1, 1)]
        self.scaled_offsets = self.basic_offsets
        self.stride = 0

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

    def get_nbvs_and_grasp_poses(self, candidate_grasps):
        rospy.wait_for_service('calculate_nbv')
        try:
            calculate_nbv = rospy.ServiceProxy('calculate_nbv', CalculateNbv)
            eef_poses = PoseArray()
            # p = self.generate_grasp_pose('nbv_eval')
            for g in candidate_grasps:
                eef_poses.poses.append(g.pose)
            num_to_req = Int32()
            num_to_req.data = self.num_nbvs_to_request
            res = calculate_nbv(eef_poses, num_to_req)
            print res
            for p in res.nbv_poses.poses:
                self.top_ranked_nbvs.append(p)
            # self.nbv_tf = res.nbv
            self.grasp_eef_pose_to_achieve_nbv.header.frame_id = "world"
            # just take the best one for simplicity
            self.grasp_eef_pose_to_achieve_nbv.pose = res.eef_poses.poses[0]
            # print self.nbv_tf, self.grasp_eef_pose_to_achieve_nbv
            # grasp_pose = res.eef_pose.pose
            # dummy = Transform()
            # dummy.rotation.x = -0.5
            # dummy.rotation.y = 0.5
            # dummy.rotation.z = 0.5
            # dummy.rotation.w = -0.5
            # self.nbv_tf = dummy

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def convertTransformMsgToTf(self, transform):
        tf_t = [transform.translation.x,
                transform.translation.y, transform.translation.z]
        tf_r = [transform.rotation.x, transform.rotation.y,
                transform.rotation.z, transform.rotation.w]
        return (tf_t, tf_r)

    def convertPoseMsgToTf(self, pose_msg):
        tf_t = [pose_msg.position.x,
                pose_msg.position.y, pose_msg.position.z]
        tf_r = [pose_msg.orientation.x, pose_msg.orientation.y,
                pose_msg.orientation.z, pose_msg.orientation.w]
        return (tf_t, tf_r)

    def generate_nbv_pose(self, angle, rank):
        # https://answers.ros.org/question/133331/multiply-two-tf-transforms-converted-to-4x4-matrices-in-python/

        # E_T_C
        self.tf_listener.waitForTransform(
            "/" + self.object_name, "/" + self.palm_link, rospy.Time(0), rospy.Duration(3.0))
        E_T_C = self.tf_listener.lookupTransform(
            "/" + self.object_name, "/" + self.palm_link, rospy.Time(0))
        # print E_T_C
        E_T_C_t = tf.transformations.translation_matrix(E_T_C[0])
        E_T_C_r = tf.transformations.quaternion_matrix(E_T_C[1])
        E_T_C_m = np.dot(E_T_C_t, E_T_C_r)
        # print 'E_T_C_m\n', E_T_C_m

        # C_T_L
        C_T_L = TransformStamped()
        C_T_L.header.frame_id = "/" + self.lens_link
        C_T_L.child_frame_id = "/" + self.object_name

        # # N_T_O
        # self.tf_listener.waitForTransform(
        #     "/" + self.orig_obj, "/" + self.nbv, rospy.Time(0), rospy.Duration(3.0))
        # N_T_O = self.tf_listener.lookupTransform(
        #     "/" + self.orig_obj, "/" + self.nbv, rospy.Time(0))
        N_T_O = self.convertPoseMsgToTf(self.top_ranked_nbvs[rank])
        N_T_O_t = tf.transformations.translation_matrix(N_T_O[0])
        N_T_O_r = tf.transformations.quaternion_matrix(N_T_O[1])
        # N_T_O_r = tf.transformations.quaternion_matrix([0, 0, 0, 1])
        N_T_O_m = np.dot(N_T_O_t, N_T_O_r)
        # print 'N_T_O_m\n', N_T_O_m

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
        # print 'C_T_P_m\n', C_T_P_m

        P_T_L_t = tf.transformations.translation_matrix([0.0, 0, 0.3, 1.0])
        P_T_L_r = [[0, math.cos(angle), math.sin(angle), 0], [0, math.sin(
            angle), -math.cos(angle), 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
        # P_T_L_r = tf.transformations.quaternion_matrix(
        #     [-0.5, -0.5, 0.5, -0.5])  # point nbv x axis towards camera
        P_T_L_m = np.dot(P_T_L_t, P_T_L_r)
        # print 'P_T_L_m\n', P_T_L_m
        C_T_L_m = np.dot(P_T_L_m, C_T_P_m)
        # print 'C_T_L_m\n', C_T_L_m
        # end alternative

        # L_T_O
        self.tf_listener.waitForTransform(
            "/" + self.orig_obj, "/" + self.lens_link, rospy.Time(0), rospy.Duration(3.0))  # or original cube1??
        L_T_O = self.tf_listener.lookupTransform(
            "/" + self.orig_obj, "/" + self.lens_link, rospy.Time(0))
        L_T_O_t = tf.transformations.translation_matrix(L_T_O[0])
        L_T_O_r = tf.transformations.quaternion_matrix(L_T_O[1])
        L_T_O_m = np.dot(L_T_O_t, L_T_O_r)
        # print 'L_T_O_m\n', L_T_O_m

        # O_T_W
        self.tf_listener.waitForTransform(
            "/" + self.reference_frame, "/" + self.orig_obj, rospy.Time(0), rospy.Duration(3.0))
        O_T_W = self.tf_listener.lookupTransform(
            "/" + self.reference_frame, "/" + self.orig_obj, rospy.Time(0))
        # print 'O_T_W\n', O_T_W
        O_T_W_t = tf.transformations.translation_matrix(O_T_W[0])
        O_T_W_r = tf.transformations.quaternion_matrix(O_T_W[1])
        O_T_W_m = np.dot(O_T_W_t, O_T_W_r)
        O_T_W_t2 = tf.transformations.translation_from_matrix(O_T_W_m)
        O_T_W_r2 = tf.transformations.quaternion_from_matrix(O_T_W_m)
        # print 'O_T_W_m\n', O_T_W_m

        # E_T_W
        L_T_W_m = np.dot(O_T_W_m, L_T_O_m)
        # print 'L_T_W_m\n', L_T_W_m
        C_T_W_m = np.dot(L_T_W_m, C_T_L_m)
        C_T_W_t = tf.transformations.translation_from_matrix(C_T_W_m)
        C_T_W_r = tf.transformations.quaternion_from_matrix(C_T_W_m)
        print 'C_T_W_m\n', C_T_W_m
        E_T_W_m = np.dot(C_T_W_m, E_T_C_m)
        E_T_W_t = tf.transformations.translation_from_matrix(E_T_W_m)
        # E_T_W_r = tf.transformations.quaternion_from_matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        E_T_W_r = tf.transformations.quaternion_from_matrix(E_T_W_m)
        # print E_T_W_t, E_T_W_r

        self.tf_broadcaster.sendTransform(
            E_T_W_t, E_T_W_r, rospy.Time.now(), 'pres_eef', 'world')

        # print 'E_T_W_m\n', E_T_W_m

        ps = PoseStamped()
        ps.header.frame_id = "/" + self.reference_frame
        ps.pose.position.x = E_T_W_t[0]  # .32455 # E_T_W_t[0]
        ps.pose.position.y = E_T_W_t[1]  # 0.10394 # E_T_W_t[1]
        ps.pose.position.z = E_T_W_t[2]  # 1.0798 # E_T_W_t[2]
        ps.pose.orientation.x = E_T_W_r[0]  # -0.541971 #E_T_W_r[0]
        ps.pose.orientation.y = E_T_W_r[1]  # 0.32541 #E_T_W_r[1]
        ps.pose.orientation.z = E_T_W_r[2]  # 0.60627 #E_T_W_r[2]
        ps.pose.orientation.w = E_T_W_r[3]  # 0.48251 #E_T_W_r[3]
        js, outcome = self.get_ik('present', ps)

        obj_present_tf = PoseStamped()
        obj_present_tf.header.frame_id = "/" + self.reference_frame
        obj_present_tf.pose.position.x = C_T_W_t[0]  # .32455 # E_T_W_t[0]
        obj_present_tf.pose.position.y = C_T_W_t[1]  # 0.10394 # E_T_W_t[1]
        obj_present_tf.pose.position.z = C_T_W_t[2]  # 1.0798 # E_T_W_t[2]
        obj_present_tf.pose.orientation.x = C_T_W_r[0]  # -0.541971 #E_T_W_r[0]
        obj_present_tf.pose.orientation.y = C_T_W_r[1]  # 0.32541 #E_T_W_r[1]
        obj_present_tf.pose.orientation.z = C_T_W_r[2]  # 0.60627 #E_T_W_r[2]
        obj_present_tf.pose.orientation.w = C_T_W_r[3]  # 0.48251 #E_T_W_r[3]

        obj_orig_tf = PoseStamped()
        obj_orig_tf.header.frame_id = "/" + self.reference_frame
        obj_orig_tf.pose.position.x = O_T_W_t2[0]  # .32455 # E_T_W_t[0]
        obj_orig_tf.pose.position.y = O_T_W_t2[1]  # 0.10394 # E_T_W_t[1]
        obj_orig_tf.pose.position.z = O_T_W_t2[2]  # 1.0798 # E_T_W_t[2]
        obj_orig_tf.pose.orientation.x = O_T_W_r2[0]  # -0.541971 #E_T_W_r[0]
        obj_orig_tf.pose.orientation.y = O_T_W_r2[1]  # 0.32541 #E_T_W_r[1]
        obj_orig_tf.pose.orientation.z = O_T_W_r2[2]  # 0.60627 #E_T_W_r[2]
        obj_orig_tf.pose.orientation.w = O_T_W_r2[3]  # 0.48251 #E_T_W_r[3]     
        print 'orig obj tf\n', obj_orig_tf   
        return js, outcome, obj_present_tf, obj_orig_tf

    def hm_clbk(self, msg):
        try:
            self.hm_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError as e:
            print(e)
        # TODO extract max and min of cv_image
        # self.hm_image = cv2.normalize(
        #     self.hm_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.hm_received = True
        # print self.hm_image

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
        js, _ = self.get_ik('grasp')
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

    def move_object_to_pose_magically(self, pose, offset=False):
        print "moving object to: \n", pose
        if offset:
            pose.pose.position.z = pose.pose.position.z - 0.02
        
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            req = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            ms = ModelState()
            ms.model_name = self.object_name
            ms.pose = pose.pose
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
            if eef_pose is None:  # just for the grasp
                ik.pose_stamped = self.grasp_eef_pose_to_achieve_nbv
            else:  # for the presentation pose
                ik.pose_stamped = eef_pose
            if not phase == 'present':
                ik.pose_stamped.pose.position.z = self.get_object_height() + self.hand_height_offset + \
                    self.offset_by_phase[phase]  # 0.24 self.hand_height_offsetgood 0.4
            print 'Aiming for eef height: ', ik.pose_stamped.pose.position.z
            ik.timeout.secs = 3.0  # [s]
            ik.attempts = 1
            # print '\nIK message:', ik
            res = req(ik)
            outcome = res.error_code.val
            print 'ik result, to help to debug finger angle', res
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        js = {}
        for name, val in zip(res.solution.joint_state.name, res.solution.joint_state.position):
            js[name] = val
        print 'joint states requested by ik:\n', js
        return js, outcome

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

    def generate_grasp_pose(self, phase):
        offset = self.offset_by_phase[phase]
        if self.verbose:
            print 'Generating grasp pose'
        sig_pos = 0.0001  # [m] std dev for position
        x = np.random.normal(self.object_position[0], sig_pos)
        y = np.random.normal(self.object_position[1], sig_pos)
        if phase == 'pre':
            th = 0.0  # np.random.uniform(0.0, math.pi * 2.0)
            self.grasp_theta = th
        elif phase == 'nbv_eval':
            th = 0.0  # np.random.uniform(0.0, math.pi * 2.0)
            print 'theta of grasp candidate: ', th
        # https://www.programcreek.com/python/example/70252/geometry_msgs.msg.PoseStamped
        ps = PoseStamped()
        ps.header.frame_id = "/" + self.reference_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = self.get_object_height() + self.hand_height_offset + \
            offset  # 0.24 0.18good 0.4
        q = tf.transformations.quaternion_from_euler(
            math.pi, 0, self.grasp_theta, axes='sxyz')
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
        aco = AttachedCollisionObject()
        aco.link_name = self.palm_link
        aco.object.id = self.object_name
        for finger in self.finger_links_allowed_to_touch:
            aco.touch_links.append(finger)
        # print 'attached collision object\n', aco
        # supplement
        # print 'joint states inside move_to_state, just before motion plan req\n', joint_states
        for name, val in joint_states.items():
            if name not in self.joints_to_exclude:
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = val
                jc.tolerance_above = self.joint_angle_tolerance
                jc.tolerance_below = self.joint_angle_tolerance
                jc.weight = 1.0
                con.joint_constraints.append(jc)
        mpr.start_state.attached_collision_objects.append(aco)
        mpr.goal_constraints = [con]
        mpr.group_name = self.planning_group_name
        mpr.num_planning_attempts = 10
        mpr.allowed_planning_time = 3.0  # [s]
        # print 'mpr in move_to_state\n', mpr
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
            wp_end.pose.position.z = self.get_object_height() + self.hand_height_offset  # 0.9
        print 'height wp end: ', wp_end.pose.position.z
        waypoints = [wp_start.pose, wp_end.pose]
        # print waypoints
        jump_threshold = 10
        max_step = 0.02
        avoid_collisions = False  # was true
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
        print "js in move from grasp to raised", js
        for name, val in js.items():
            if name not in self.joints_to_exclude:
                names.append(name)
                vals.append(val)
        start_state.joint_state.name = names
        start_state.joint_state.position = vals
        print 'start state in move from grasp to raised\n', start_state
        group_name = self.planning_group_name
        link_name = self.palm_link_eef
        wp_start = self.get_eef_pose()
        print 'in move from grasp to raised, wp_start, eef_pose:\n', wp_start
        wp_end = self.get_eef_pose()
        print 'in move from grasp to raised, wp_end, eef_pose:\n', wp_end
        wp_end.pose.position.z = 1.2
        print 'height wp end: ', wp_end.pose.position.z
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
            print 'move from grasp to raised result of compute cart path:\n', res
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

    def move_from_preplace_to_place(self):
        if self.verbose:
            print 'Moving from preplace to place'
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
        wp_end.pose.position.z = self.eef_height_at_grasp
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
        print "Moving to place position"
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
            self.eef_height_at_grasp = self.get_eef_pose().pose.position.z
        elif action == 'open':
            js.position = [self.finger_joint_angles_ungrasp] * 3
        self.finger_pub.publish(js)
        rospy.sleep(3)

    def save_joint_states_at_pre(self):
        self.joint_states_at_pre = self.get_joint_states()
        print 'joint states at pre\n', self.joint_states_at_pre

        # self.tf_listener.waitForTransform("/" + self.reference_frame, "/" + self.palm_link_eef, rospy.Time(0), rospy.Duration(4.0))
        # tf = self.tf_listener.lookupTransform("/" + self.reference_frame, "/" + self.palm_link_eef, rospy.Time(0))

        # # print trans, rot

        # self.eef_pose_at_grasp.position.x = trans[0]
        # self.eef_pose_at_grasp.position.y = trans[1]
        # self.eef_pose_at_grasp.position.z = trans[2]
        # self.eef_pose_at_grasp.orientation.x = rot[0]
        # self.eef_pose_at_grasp.orientation.y = rot[1]
        # self.eef_pose_at_grasp.orientation.z = rot[2]
        # self.eef_pose_at_grasp.orientation.w = rot[3]

        # self.tf_broadcaster.sendTransform(tf)
        # print "Sent transform!"

    # def save_joint_states_at_grasp(self):
    #     self.joint_states_at_grasp = self.get_joint_states()
    #     print 'joint states at grasp\n', self.joint_states_at_pre

    def save_joint_states_at_raised(self):
        self.joint_states_at_raised = self.get_joint_states()
        print 'joint states at raised\n', self.joint_states_at_raised

    def save_current_eef_pose(self, phase):
        while self.save_current_eef_pose_pub.get_num_connections() < 1:
            print 'waiting for eef save connections'
            pass
        msg = String()
        msg.data = phase
        self.save_current_eef_pose_pub.publish(msg)
        print 'Saved eef pose at', phase

    def save(self, height_map, ik_pre, height):
        if self.verbose:
            print 'Saving data'
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
        # outfile = TemporaryFile()
        np.savez('output', height_map=height_map, height=height)

    def capture_and_process_observation(self):
        print "Capture and process obs"
        try:
            req = rospy.ServiceProxy(
                '/capture_and_process_observation', CaptureAndProcessObservation)
            msg = Bool()
            msg.data = True
            res = req(msg)
            print 'capture and process response:', res
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # Hash functions
    def index_to_pixel_coord(self, index):
        c = i % self.nc
        r = i / self.nc
        return r, c

    def pixel_coord_to_index(self, r, c):
        return r * self.nc + c

    def pixel_coord_to_local_coord(self, r, c):
        xl = self.gm_res * (self.nc / 2 - (c + 0.5))
        yl = self.gm_res * (self.nr / 2 - (r + 0.5))
        return xl, yl

    def local_coord_to_pixel_coord(self, x, y):
        c = self.nc / 2 - x / self.gm_res - 0.5
        r = self.nr / 2 - y / self.gm_res - 0.5
        return r, c

    def index_to_local_coord(self, index):
        return self.pixel_coord_to_local_coord(self.index_to_pixel_coord(index))

    def local_coord_to_index(self, x, y):
        return self.pixel_coord_to_index(self.local_coord_to_pixel_coord(x, y))

    def local_coord_to_world_coord(self, x, y):
        return self.gm_ctr_x + x, self.gm_ctr_y + y

    def world_coord_to_local_coord(self, x, y):
        return x - self.gm_ctr_x, y - self.gm_ctr_y

    def pixel_coord_to_image_coord(self, r, c):
        return c, r

    def image_coord_to_pixel_coord(self, x, y):
        return y, x

    def pixel_coord_to_world_coord(self, r, c):
        xl, yl = self.pixel_coord_to_local_coord(r, c)
        return self.local_coord_to_world_coord(xl, yl)

    def world_coord_to_image_coord(self, x, y):
        xl, yl = self.world_coord_to_local_coord(x, y)
        r, c = self.local_coord_to_pixel_coord(xl, yl)
        return self.pixel_coord_to_image_coord(r, c)

    def image_coord_to_world_coord(self, x, y):
        r, c = self.image_coord_to_pixel_coord(x, y)
        xl, yl = self.pixel_coord_to_local_coord(r, c)
        return self.local_coord_to_world_coord(xl, yl)

    def gm_clbk(self, msg):
        self.gm_res = msg.info.resolution
        self.gm_ctr_x = msg.info.pose.position.x
        self.gm_ctr_y = msg.info.pose.position.y
        self.nr = int(msg.info.length_x / self.gm_res)
        # print self.nr
        self.nc = int(msg.info.length_y / self.gm_res)
        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        # except CvBridgeError as e:
        #     print(e)

        # self.hm_image = cv2.normalize(
        #     cv_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.stride = int(self.sample_point_spacing / self.gm_res)
        self.scaled_offsets = [
            (x[0] * (self.stride - 1), x[1] * (self.stride - 1)) for x in self.basic_offsets]
        self.gm_received = True
        print "gm received!"

    def gm_mask_clbk(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError as e:
            print(e)
        self.hm_mask_image = cv2.normalize(
            cv_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        nz = cv.findNonZero(self.hm_mask_image)

    def generateSampleMidpoints(self):
        midpoints = []
        for r in range(self.stride - 1, self.nr - self.stride + 1):
            for c in range(self.stride - 1, self.nc - self.stride + 1):
                midpoints.append((r, c))
        # print 'midpoints', midpoints
        return midpoints  # self.generateGridsAroundMidpoints(midpoints)

    # def generateGridsAroundMidpoints(self, midpoints):
    #     # row-major
    #     grids = []
    #     # offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
    #             #    (0, 1), (1, -1), (1, 0), (1, 1)]
    #     for pt in midpoints:
    #         one_square = []
    #         for offset in self.scaled_offsets:
    #             one_square.append([(pt[0] + offset[0]),
    #                                (pt[1] + offset[1])])
    #         grids.append(one_square)
    #     return grids

    def midpoint_and_angle_to_eef_pose(self, midpoint, angle):
        xw, yw = self.pixel_coord_to_world_coord(midpoint[0], midpoint[1])
        ps = PoseStamped()
        quat = tf.transformations.quaternion_from_euler(
            -math.pi, 0, angle + math.pi/2)  # TBA +/-!!! WAS -angle, should be opposite of what's in transform_hm 90deg HACK
        ps.header.frame_id = self.reference_frame
        ps.pose.position.x = xw
        ps.pose.position.y = yw
        ps.pose.position.z = 0.93  # TBA
        ps.pose.orientation.x = quat[0]
        ps.pose.orientation.y = quat[1]
        ps.pose.orientation.z = quat[2]
        ps.pose.orientation.w = quat[3]
        return ps

    def stand_in_coord_and_angle_to_eef_pose(self, coord, angle):
        xw, yw = coord
        ps = PoseStamped()
        quat = tf.transformations.quaternion_from_euler(
            -math.pi, 0, angle + math.pi/2)  # TBA +/-!!! WAS -angle, should be opposite of what's in transform_hm 90deg HACK
        ps.header.frame_id = self.reference_frame
        ps.pose.position.x = xw
        ps.pose.position.y = yw
        ps.pose.position.z = self.get_object_height() + self.hand_height_offset + \
            self.offset_by_phase['grasp']  # 0.93 # TBA
        ps.pose.orientation.x = quat[0]
        ps.pose.orientation.y = quat[1]
        ps.pose.orientation.z = quat[2]
        ps.pose.orientation.w = quat[3]
        return ps

    # def get_grid_midpoint(self, grid):
    #     return grid[len(grid)/2]

    def predict_grasp_success_probability(self, midpoint, angle):
        # get the actual points to be sampled
        hm_vals_at_midpoint = self.get_height_map_values_at_midpoint_and_angle(
            midpoint, angle)
        # put these into the model
        sum = self.convolveTemplateWithImage(
            self.getExampleWeights(), hm_vals_at_midpoint)
        # np.sum(hm_vals_at_midpoint) # reshape(grid, grid.size()) # np.random.uniform()
        prob = sum
        return prob  # TODO update

    def getExampleWeights(self):
        return [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    def convolveTemplateWithImage(self, template, pixels):
        sum = 0
        pixels = np.divide(pixels, 255.0)
        template = np.reshape(np.asarray(template), [3, 3])
        # print pixels, template
        # for p, t in zip(pixels, template):
        #     sum += [p[0], p[1]] *
        return np.sum(np.multiply(pixels, template)) / 9.0

    def transform_hm(self, x, y, angle):
        # shift to a new center, which will become the center of rotation
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        im = self.hm_image
        Mt = np.asarray([[1.0, 0.0, float(self.nr/2 - y)],
                         [0.0, 1.0, float(self.nc/2 - x)]])
        image_tr = cv2.warpAffine(
            im, Mt, im.shape[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        image_rot = imutils.rotate(image_tr, -angle*180/math.pi)
        return image_rot

    def get_height_map_values_at_midpoint_and_angle(self, midpoint, angle):
        num_elements_in_grid = len(self.scaled_offsets)
        side_length = math.sqrt(num_elements_in_grid)  # HACK!!!
        vals = []
        hm_transformed_to_desired_eef_pose = self.transform_hm(
            midpoint[0], midpoint[1], angle)
        # print midpoint, angle
        # print hm_transformed_to_desired_eef_pose
        for offset in self.scaled_offsets:
            # TODO r/c?
            vals.append(
                hm_transformed_to_desired_eef_pose[offset[0] + self.nr/2, offset[1] + self.nc/2])
        vals = np.reshape(vals, [side_length, side_length])
        # print vals
        return vals

    def generate_grasp_pose_candidates(self, midpoints, angles, count=0):
        probs = []
        # print 'orig image', self.hm_image
        # print 'scaled offets:', self.scaled_offsets
        # UNCOMMENT THIS FOR FULL HEIGHT MAP READING
        # for midpoint in midpoints:
        #     for angle in angles:
        #         # print "midpoint: ", midpoint, " angle: ", angle
        #         pose = self.midpoint_and_angle_to_eef_pose(midpoint, angle)
        #         probs.append((self.predict_grasp_success_probability(midpoint, angle), pose))
        # probs_descending = sorted(probs, reverse=True)
        # print [x for x in probs_descending[:self.num_grasp_candidates_to_generate]]
        # return [x[1] for x in probs_descending[:self.num_grasp_candidates_to_generate]]
        # UNCOMMENT THIS FOR FULL HEIGHT MAP READING
        mps = [(0.23, 0.06), (0.17, 0.06)]  # for levels 2-4
        # mps = [(0.23, 0.0), (0.17, 0.0)]  # for level 1
        dummy = self.stand_in_coord_and_angle_to_eef_pose(mps[count], 0)
        return [dummy]

    def generateSampleAngles(self):
        return [x/float(self.num_grasp_angles_to_try) * math.pi * 2.0 for x in range(self.num_grasp_angles_to_try)]

    def evaluate_against_gt(self):
        try:
            req = rospy.ServiceProxy('/eval_gt', GTEval)
            msg = Bool()
            msg.data = True
            res = req(msg)
            # print res
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        # create confusion matrix here with proportions
        conf = np.asarray([[res.GT_OCC_EST_OCC.data, res.GT_OCC_EST_FREE.data, res.GT_OCC_EST_UNOBS.data],
                           [res.GT_FREE_EST_OCC.data, res.GT_FREE_EST_FREE.data,
                               res.GT_FREE_EST_UNOBS.data],
                           [res.GT_UNOBS_EST_OCC.data, res.GT_UNOBS_EST_FREE.data, res.GT_UNOBS_EST_UNOBS.data]])
        sum_conf = np.sum(conf)
        conf = np.true_divide(conf, sum_conf)
        print "Confusion matrix:\n", conf

    def manage_collision_matrix(self):
        allowed_collision_elements = ['jaco_6_hand_limb', 'jaco_7_finger_mount_index', 'jaco_7_finger_mount_pinkie', 'jaco_7_finger_mount_thumb',
                                      'jaco_8_finger_index', 'jaco_8_finger_pinkie', 'jaco_8_finger_thumb', 'jaco_9_finger_index_tip', 'jaco_9_finger_pinkie_tip', 'jaco_9_finger_thumb_tip']
        scene = PlanningScene()
        acm_revised = AllowedCollisionMatrix()
        try:
            req = rospy.ServiceProxy(
                '/get_planning_scene', GetPlanningScene)
            psc = PlanningSceneComponents()
            psc.components = 128
            res = req(psc)
            scene = res.scene
            acm = res.scene.allowed_collision_matrix
            # print acm
            # find index of obj
            index = acm.entry_names.index(self.object_name)
            # find indices of all ones of interest
            ok_collision_indices = []
            for el in allowed_collision_elements:
                ok_collision_indices.append(acm.entry_names.index(el))
            for i in ok_collision_indices:
                acm.entry_values[index].enabled[i] = True
            # print "revised acm", acm
            acm_revised = acm
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        try:
            req = rospy.ServiceProxy(
                '/apply_planning_scene', ApplyPlanningScene)
            ps = scene
            ps.is_diff = True
            ps.allowed_collision_matrix.entry_values[index].enabled[0] = True
            # ps.allowed_collision_matrix = acm_revised
            print 'acm revised at send', ps.allowed_collision_matrix
            res = req(ps)
            print "Planning scene update request status:", res.success
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def save_finger_joint_states_at_grasp(self):
        js = self.get_joint_states()
        for finger in self.finger_joint_names:
            self.finger_joint_states_at_grasp[finger] = js[finger]

    def supplement_with_correct_finger_joint_angles(self, js):
        js_renewed = {}
        for joint in js:
            js_renewed[joint] = js[joint]
        for finger in self.finger_joint_names:
            js_renewed[finger] = self.finger_joint_states_at_grasp[finger]
        # print 'should be a renewed js with better finger joint angles\n', js
        return js_renewed

    def change_physics(self, action):
        # msg = Empty()
        if action == 'pause':
            try:
                req = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
                res = req()
                print "physics off"
                # print res
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e            
        elif action == 'unpause':
            try:
                req = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
                res = req()
                print "physics on"
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e                     


def main(args):
    rospy.init_node('grasp_data_collection')
    gdc = GraspDataCollection(args[1])
    gdc.save_arm_home_state()

    # ### UNCOMMENT TO RETURN TO NORMAL
    # while not gdc.gm_received or not gdc.hm_received:
    #     print "waiting"
    #     rospy.sleep(0.5)
    midpoints = gdc.generateSampleMidpoints()
    angles = gdc.generateSampleAngles()
    # gdc.change_physics('unpause')
    # quit()
    count = 0

    # ### UNCOMMENT TO RETURN TO NORMAL
    # gdc.manage_collision_matrix()
    # do the actual sequence
    # THIS BLOCK DOES THE ACTUAL ARM MOVING
    # while gdc.current_trial_no < gdc.total_num_trials:
    #     # gdc.save_eef_pose_at_grasp()
    #     # gdc.position_object()
    #     # height_map = gdc.height_map  # TODO make not None
    #     # gdc.pickup()
    #     # gdc.execute_grasp_action('open')
    #     candidate_grasps = gdc.generate_grasp_pose_candidates(
    #         midpoints, angles, count)
    #     print candidate_grasps
    #     gdc.get_nbvs_and_grasp_poses(candidate_grasps)
    #     js_pre, _ = gdc.get_ik('pre')
    #     raw_input("Press to move to pre")
    #     gdc.move_to_state(js_pre)
    #     gdc.save_joint_states_at_pre()
    #     # gdc.eef_tf_save_request()
    #     # rospy.sleep(2)
    #     raw_input("Press to move to grasp")
    #     gdc.move_from_pregrasp_to_grasp(False)
    #     gdc.save_current_eef_pose('grasp')
    #     # gdc.save_finger_joint_states_at_grasp()
    #     # raw_input("press any key to move back")
    #     # rospy.sleep(2)
    #     gdc.actuate_fingers('close')
    #     raw_input("Press to move to raised")
    #     gdc.move_from_grasp_to_raised()
    #     gdc.save_joint_states_at_raised()
    #     # raw_input("Press to try to close fingers again")
    #     # gdc.actuate_fingers('close')
    #     # gdc.move_to_state(gdc.generate_joint_states_presentation_pose())
    #     raw_input("Press to do nbv kinematic calcs")
    #     for rank in range(gdc.num_nbvs_to_request):
    #         for angle in gdc.present_roll_angle_options:
    #             js, outcome = gdc.generate_nbv_pose(angle, rank)
    #             if outcome == 1:
    #                 # gdc.supplement_with_correct_finger_joint_angles(js)
    #                 # print 'should be a renewed js with better finger joint angles\n', js
    #                 raw_input("press to move to nbv")
    #                 gdc.move_to_state(js)
    #                 break
    #         # https://stackoverflow.com/questions/189645/how-to-break-out-of-multiple-loops-in-python
    #         else:
    #             continue
    #         break
    #     gdc.save_current_eef_pose('present')
    #     gdc.capture_and_process_observation()
    #     raw_input("press any key to move back")
    #     rospy.sleep(2)
    #     gdc.move_to_state(gdc.joint_states_at_raised)
    #     rospy.sleep(2)
    #     gdc.move_from_preplace_to_place()
    #     gdc.save_current_eef_pose('place')
    #     rospy.sleep(2)
    #     # gdc.move_to_state(gdc.joint_states_at_grasp)
    #     # gdc.move_from_pregrasp_to_grasp(False)
    #     # gdc.move_to_state(gdc.get_ik('pre'))
    #     # gdc.move_from_pregrasp_to_grasp(True)
    #     gdc.actuate_fingers('open')
    #     gdc.move_from_grasp_to_raised()
    #     # height = gdc.get_object_height()
    #     # gdc.execute_grasp_action('open')
    #     # gdc.save(height_map, ik_pre, height)
    #     gdc.increment_current_trial_no()
    #     gdc.return_arm_home()
    # THIS BLOCK DOES THE ACTUAL ARM MOVING /END

    while gdc.current_trial_no < gdc.total_num_trials:
        candidate_grasps = gdc.generate_grasp_pose_candidates(
            midpoints, angles, count)
        raw_input("Press to calculate nbv")
        gdc.get_nbvs_and_grasp_poses(candidate_grasps)
        js_pre, _ = gdc.get_ik('pre')
        raw_input("Press to move to pre")
        gdc.move_to_state(js_pre)
        raw_input("Press to move to grasp")
        gdc.move_from_pregrasp_to_grasp(False)
        gdc.save_current_eef_pose('grasp')
        # rospy.sleep(5)
        raw_input("Press to do nbv kinematic calcs")
        for rank in range(gdc.num_nbvs_to_request):
            for angle in gdc.present_roll_angle_options:
                js, outcome, obj_present_tf, obj_orig_tf = gdc.generate_nbv_pose(angle, rank)
                if outcome == 1:
                    raw_input("press to move to raised")
                    gdc.move_from_grasp_to_raised()
                    gdc.save_joint_states_at_raised()
                    raw_input("press to move to nbv")
                    gdc.move_to_state(js)
                    gdc.save_current_eef_pose('present')
                    # rospy.sleep(2)
                    # gdc.change_physics('pause')
                    raw_input("press to teleport up")
                    gdc.move_object_to_pose_magically(obj_present_tf)                    
                    raw_input("press to process")
                    gdc.capture_and_process_observation()
                    raw_input("press to teleport down")
                    gdc.move_object_to_pose_magically(obj_orig_tf, offset=True)
                    # gdc.change_physics('unpause')
                    break
            # https://stackoverflow.com/questions/189645/how-to-break-out-of-multiple-loops-in-python
            else:
                continue
            break
        gdc.evaluate_against_gt()
        gdc.move_to_state(gdc.joint_states_at_raised)
        # gdc.return_arm_home()
        gdc.increment_current_trial_no()
        count += 1

        # rospy.spin()
        if rospy.is_shutdown():
            break


if __name__ == '__main__':
    main(sys.argv)
