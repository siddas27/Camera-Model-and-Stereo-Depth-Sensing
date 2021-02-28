import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

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

left_image = cv2.imread("images/task_3_and_4/left_4.png")
right_image = cv2.imread("images/task_3_and_4/right_4.png")


left_dst = utils.undistort(l_mtx, l_dist, left_image, R1, P1)
right_dst = utils.undistort(r_mtx, r_dist, right_image, R2, P2)

# step 2
window_size =3
stereo = cv2.StereoSGBM_create(minDisparity=-1,
        numDisparities=3*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size*window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size*window_size,
        disp12MaxDiff=12,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=0,
        mode=cv2.STEREO_SGBM_MODE_HH)

disparity = stereo.compute(left_dst,right_dst)

disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

cv2.imshow('ds',disparity)
cv2.waitKey(890)
# cv2.imwrite('output/task_4/left_rectified.png',left_dst)
# cv2.imwrite('output/task_4/right_rectified.png',right_dst)
#
# cv2.imwrite('output/task_4/disparity.png',disparity)
#
# _3dImage = cv2.reprojectImageTo3D(	disparity, Q)
# cv2.imwrite('output/task_4/3d.png',_3dImage)