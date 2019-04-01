#!usr/bin/env python

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('height_map_1.jpg',0)
img2 = img.copy()
template = cv2.imread('filter.jpg',0)
w, h = template.shape[::-1]

meth = 'cv2.TM_CCOEFF'

angles = np.arange(0, 180, 10)
# print angles

for angle in angles:
    # rotate template by angle
    rot = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
    # print rot
    template_rot = cv2.warpAffine(template, rot, template.shape[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue = 255)
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template_rot,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print max_val

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, (0,255,0), 2)

    plt.subplot(131),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img,cmap = 'gray')
    plt.title('Grasp Location'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(template_rot,cmap = 'gray')
    plt.title('Grasp Orientation'), plt.xticks([]), plt.yticks([])
       
    plt.show()