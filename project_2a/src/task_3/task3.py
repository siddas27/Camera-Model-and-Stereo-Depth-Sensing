import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

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
left_dst = utils.undistort(l_mtx,l_dist,gray_left_image,R1,P1)
right_dst = utils.undistort(r_mtx,r_dist,gray_right_image,R1,P2)
cv2.imwrite("output/task_3/left_undistorted.png",left_dst)
cv2.imwrite("output/task_3/right_undistorted.png",right_dst)

# Step 2.2
orb = cv2.ORB_create()
l_kp = orb.detect(left_dst,None)
l_kp, des1 = orb.compute(left_dst,l_kp)
r_kp = orb.detect(right_dst,None)
r_kp, des2 = orb.compute(right_dst,r_kp)
#r_kp = utils.find_local_maxima(r_kp,1)

# Step 2.3 - find local maxima, draw keypoints and save image
local_maxima_l_kp = utils.find_local_maxima(l_kp,6)

l_kp_image = cv2.drawKeypoints(left_dst,l_kp,None, color=(255,0,0),flags=0)
cv2.imwrite("output/task_3/left_detected_feature_points.png",l_kp_image)
local_maxima_l_kp_image = cv2.drawKeypoints(left_dst,local_maxima_l_kp,None, color=(255,0,0),flags=0)
cv2.imwrite("output/task_3/left_local_maxima_feature_points.png",local_maxima_l_kp_image)
#r_orb = cv2.ORB.detect(right_dst,None)

# Step 3
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatches(left_dst,l_kp,right_dst,r_kp,matches[:30],outImg=None,flags=2)
cv2.imwrite("output/task_3/selected_matches_feature_points.png",img3)

# todo step 4 triangular points
matched_l_kp, matched_r_kp = utils.get_matched_key_points(matches,l_kp,r_kp)
matched_l_kp = np.array(matched_l_kp)
matched_r_kp = np.array(matched_r_kp)
p3d_points =cv2.triangulatePoints(P1,P2,matched_l_kp,matched_r_kp)
# homogenous to non-homogeneous coordinates
nh_3d_points = p3d_points[:-1]/p3d_points[-1]

# todo step 5 plot 3D
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(nh_3d_points[0],nh_3d_points[1],nh_3d_points[2],'gray')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()