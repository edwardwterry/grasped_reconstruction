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
from geometry_msgs.msg import Pose, Point, PoseArray
from grid_map_msgs.msg import GridMap
import math
import pickle
import os
import tf

import yaml


class GTParser:
    def __init__(self, mesh_name):
        self.gt_geom_pub = rospy.Publisher("/gt_geom", PoseArray, queue_size=1)
        self.mesh_name = mesh_name
        self.parseYaml()

    def parseYaml(self):
        with open("object_ground_truth.yaml", 'r') as stream:
            data = yaml.load(stream)
        print data[self.mesh_name]
        pa = PoseArray()
        for prim in data[self.mesh_name]:
            p = Pose()
            p.position.x = prim['origin']['x']
            p.position.y = prim['origin']['y']
            p.position.z = prim['origin']['z']
            p.orientation.x = prim['size']['x']
            p.orientation.y = prim['size']['y']
            p.orientation.z = prim['size']['z']
            pa.poses.append(p)
        print pa
        while (self.gt_geom_pub.get_num_connections()<1):
            pass
        self.gt_geom_pub.publish(pa)


def main(args):
    mesh_name = args[1]
    rospy.init_node('gt_parser')
    gp = GTParser(mesh_name)
    while True:
        rospy.spin()
        if rospy.is_shutdown():
            break


if __name__ == '__main__':
    main(sys.argv)
