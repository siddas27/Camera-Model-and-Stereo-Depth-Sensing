import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((6 * 9, 3), np.float32)
# objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# world_space_points = []
# world_space_points.append(objp)
# cams = ["left", "right"]
#
# left_image_points = []
# right_image_points = []

# Step 1


l = cv2.FileStorage('parameters/left_camera_intrinsics.xml', cv2.FileStorage_READ)
l_mtx = l.getNode('camera_matrix').mat()
l_dist = l.getNode('distortion_coefficients').mat()

r = cv2.FileStorage('parameters/right_camera_intrinsics.xml', cv2.FileStorage_READ)
r_mtx = r.getNode('camera_matrix').mat()
r_dist = r.getNode('distortion_coefficients').mat()

sr = cv2.FileStorage("parameters/stereo_rectification.xml", cv2.FileStorage_READ)
R1 = sr.getNode("R1").mat()
R2 = sr.getNode("R2").mat()
P1 = sr.getNode("P1").mat()
P2 = sr.getNode("P2").mat()
Q = sr.getNode("Q").mat()
sr.release()

sc = cv2.FileStorage("parameters/stereo_calibration.xml", cv2.FileStorage_READ)
T = sc.getNode('Translation_vector').mat()
R = sc.getNode('Rotation_matrix').mat()
F = sc.getNode('Fundamental_matrix').mat()
E = sc.getNode('Essential_matrix').mat()
sc.release()

left_image = cv2.imread("images/task_3_and_4/left_0.png")
right_image = cv2.imread("images/task_3_and_4/right_0.png")
gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Step 2.1
left_dst = utils.undistort(l_mtx,l_dist,gray_left_image,None)
right_dst = utils.undistort(r_mtx,r_dist,gray_right_image,None)
# Step 2.2
orb = cv2.ORB()
l_kp = orb.detect(left_dst,None)
#r_orb = cv2.ORB.detect(right_dst,None)

