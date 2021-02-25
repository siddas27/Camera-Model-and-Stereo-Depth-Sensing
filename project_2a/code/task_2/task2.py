import glob
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
world_space_points = []
world_space_points.append(objp)
cams = ["left","right"]

left_image_points = []
right_image_points = []

# Step 1
left_image = cv2.imread('images/task_2/left_0.png')
right_image = cv2.imread('images/task_2/right_0.png')
gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

l = cv2.FileStorage('parameters/leftparameters.xml', cv2.FileStorage_READ)
l_mtx = l.getNode('camera_matrix').mat()
l_dist = l.getNode('distortion_coefficients').mat()
r = cv2.FileStorage('parameters/rightparameters.xml', cv2.FileStorage_READ)
r_mtx = r.getNode('camera_matrix').mat()
r_dist = r.getNode('distortion_coefficients').mat()

# Step 2
ret, left_corners = cv2.findChessboardCorners(gray_left_image, (9, 6), None)
if ret:
    corners2 = cv2.cornerSubPix(gray_left_image, left_corners, (11, 11), (-1, -1), criteria)
    left_image_points.append(left_corners)

    # Draw and display the corners
    cv2.drawChessboardCorners(left_image, (9, 6), corners2, ret)
    cv2.imwrite('output/task_2/left_corner_points_annotation.png', left_image)

ret, right_corners = cv2.findChessboardCorners(gray_right_image, (9, 6), None)
if ret:
    corners2 = cv2.cornerSubPix(gray_right_image, right_corners, (11, 11), (-1, -1), criteria)
    right_image_points.append(right_corners)

    # Draw and display the corners
    cv2.drawChessboardCorners(right_image, (9, 6), corners2, ret)
    cv2.imwrite('output/task_2/right_corner_points_annotation.png', right_image)

# Step 3
ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F	=	cv2.stereoCalibrate(world_space_points,
                                                                                                    left_image_points,
                                                                                                    right_image_points,
                                                                                                    l_mtx,
                                                                                                    l_dist,
                                                                                                    r_mtx,
                                                                                                    r_dist,
                                                                                                    gray_left_image.shape[::-1], None,
                                                                                                    None, None, None, cv2.CALIB_FIX_INTRINSIC,
                                                                                                    criteria)

