#!usr/bin/env python

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

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
import math
import pickle
import os
import tf


class GraspSelection:
    def __init__(self):
        self.eef_pose_pub = rospy.Publisher("/eef_pose", Pose, queue_size=1)
        self.hm_sub = rospy.Subscriber("/height_map_image",
                                       Image, self.hmClbk,  queue_size=1)
        self.em_sub = rospy.Subscriber("/elevation_mapping/elevation_map",
                                       GridMap, self.emClbk,  queue_size=1)
        self.bridge = CvBridge()
        self.height_offset_abs = 0.77  # [m]
        self.sample_point_spacing = 0.03  # [m]
        self.sample_points_per_side = 3
        self.height_map_resolution = None  # [m] need to manually verify this
        self.model = None
        self.model_path = "/home/et/ind_study_ws/src/grasped_reconstruction/scripts/grasp_selection.pkl"
        self.grid_map_info = None
        self.rows = None
        self.cols = None
        self.center = None

    def emClbk(self, msg):
        self.grid_map_info = msg.info
        self.height_map_resolution = self.grid_map_info.resolution
        self.rows = self.grid_map_info.length_x * self.height_map_resolution
        self.cols = self.grid_map_info.length_y * self.height_map_resolution
        self.center = Point(self.grid_map_info.pose.position.x,
                            self.grid_map_info.pose.position.y, self.grid_map_info.pose.position.z)

    def hmClbk(self, msg):
        if self.grid_map_info is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            except CvBridgeError as e:
                print(e)
            cv2.imshow('image', cv_image)
            norm_image = cv2.normalize(
                cv_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.selectGrasp(norm_image)

    def publishPose(self, x, y, psi):
        eef_pose = Pose()
        quat = tf.transformations.quaternion_from_euler(
            0, 0, psi * math.pi/180.)
        eef_pose.position.x = x
        eef_pose.position.y = y
        eef_pose.position.z = self.height_offset_abs
        eef_pose.orientation.x = quat[0]
        eef_pose.orientation.y = quat[1]
        eef_pose.orientation.z = quat[2]
        eef_pose.orientation.w = quat[3]
        # self.eef_pose_pub.publish(eef_pose)

    def getXYCoordsFromPixels(self, pixels):
        cx = (pixels[0] - self.rows / 2) * \
            self.height_map_resolution + self.center.x
        cy = (pixels[1] - self.cols / 2) * \
            self.height_map_resolution + self.center.y
        return cx, cy

    def generateSampleMidpoints(self, rows, cols):
        stride = int(self.sample_point_spacing / self.height_map_resolution)
        midpoints = []
        for r in range(stride - 1, rows - stride + 1):
            for c in range(stride - 1, cols - stride + 1):
                midpoints.append((r, c))
        return self.generateGridAroundMidpoints(midpoints, stride)

    def generateGridAroundMidpoints(self, midpoints, stride):
        # row-major
        grid = []
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        for pt in midpoints:
            one_square = []
            for offset in offsets:
                one_square.append([(pt[0] + (stride - 1) * offset[0]),
                                   (pt[1] + (stride - 1) * offset[1])])
            grid.append(one_square)
        return grid

    def selectGrasp(self, cv_image):
        dims = cv_image.shape
        rows = dims[0]
        cols = dims[1]
        sample_grid = self.generateSampleMidpoints(rows, cols)

        angles = np.arange(0, 180, 45)

        match = 0.0001  # hack BEWARE!
        best_angle = None
        best_square = None
        cv2.waitKey(0)
        for angle in angles:
            # Rotate height map image by angle
            rot = cv2.getRotationMatrix2D((rows/2, cols/2), angle, 1)
            image_rot = cv2.warpAffine(
                cv_image, rot, cv_image.shape[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Apply template matching
            for square in sample_grid:
                match_ = self.convolveTemplateWithImage(
                    self.getExampleWeights(), square, image_rot)
                if match > match_:
                    best_angle = angle
                    best_square = square
                    match = match_

        cx, cy = self.getXYCoordsFromPixels(best_square[4])
        print cx, cy, best_angle
        self.publishPose(cx, cy, best_angle)

    def convolveTemplateWithImage(self, template, pixels, height_map):
        sum = 0
        for p, t in zip(pixels, template):
            sum += height_map[p[0], p[1]] * t
        return sum

    # def predictGraspSuccess(self, template, height_map):
    #     if self.model is None:
    #         self.loadModel()
    #     else:

    def getExampleWeights(self):
        return [0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1]

    def loadModel(self):
        # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
        self.model = pickle.load(open(self.model_path, 'rb'))


# model=LogisticRegression()
# model.fit(X_train, Y_train)
# # save the model to disk
# filename='finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))


def main(args):
    rospy.init_node('grasp_selection')
    gs = GraspSelection()
    while True:
        rospy.spin()
        if rospy.is_shutdown():
            break


if __name__ == '__main__':
    main(sys.argv)
