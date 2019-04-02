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
from geometry_msgs.msg import Pose
import math
import pickle
import os


class GraspSelection:

    def __init__(self):
        self.eef_pose_pub = rospy.Publisher("/eef_pose", Pose, queue_size=1)
        self.hm_sub = rospy.Subscriber("/height_map_image",
                                       Image, self.callback,  queue_size=1)
        self.bridge = CvBridge()
        # self.meth = 'cv2.TM_CCOEFF'
        self.sample_point_spacing = 0.03  # [m]
        self.sample_points_per_side = 3
        self.height_map_resolution = 0.01  # [m] need to manually verify this
        self.model = None
        self.model_path = "/home/et/ind_study_ws/src/grasped_reconstruction/scripts/grasp_selection.pkl"

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError as e:
            print(e)

        eef_pose = Pose()
        # eef_pose.position.x =
        # eef_pose.position.y =
        # eef_pose.position.z =
        # eef_pose.orientation.x =
        # eef_pose.orientation.y =
        # eef_pose.orientation.z =
        # eef_pose.orientation.w =s
        # self.eef_pose_pub.publish(eef_pose)
        self.selectGrasp(cv_image)

    def generateSampleMidpoints(self, rows, cols):
        stride = int(self.sample_point_spacing / self.height_map_resolution)
        midpoints = []
        for r in range(stride - 1, rows - stride + 1):
            for c in range(stride - 1, cols - stride + 1):
                midpoints.append((r, c))
                # print (r, c)
        return self.generateGridAroundMidpoints(midpoints, stride)

    def generateGridAroundMidpoints(self, midpoints, stride):
        # row-major
        grid = []
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
        for pt in midpoints:
            print "\n"
            for offset in offsets:
                grid.append(((pt[0] + (stride - 1) * offset[0]),
                             (pt[1] + (stride - 1) * offset[1])))
                print((pt[0] + (stride - 1) * offset[0]),
                      (pt[1] + (stride - 1) * offset[1]))
        return grid

    def selectGrasp(self, cv_image):
        dims = cv_image.shape
        rows = dims[0]
        cols = dims[1]
        sample_grid = self.generateSampleMidpoints(rows, cols)
        # just do non-rotated case for now

        angles = np.arange(0, 180, 10)

        for angle in angles:
            # rotate image by angle
            rot = cv2.getRotationMatrix2D((rows/2, cols/2), angle, 1)
            image_rot = cv2.warpAffine(
                cv_image, rot, cv_image.shape[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # img=img2.copy()
            # method=eval(meth)

            # Apply template Matching
            # res=cv2.matchTemplate(img, template_rot, method)
            # min_val, max_val, min_loc, max_loc=cv2.minMaxLoc(res)

    def loadModel(self):
        # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
        self.model = pickle.load(open(self.model_path, 'rb'))


# model=LogisticRegression()
# model.fit(X_train, Y_train)
# # save the model to disk
# filename='finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))


def main(args):
    gs = GraspSelection()
    rospy.init_node('grasp_selection', anonymous=True)
    while True:
        rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
