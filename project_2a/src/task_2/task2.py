import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
world_space_points = []
world_space_points.append(objp)
cams = ["left", "right"]

left_image_points = []
right_image_points = []

# Step 1
left_image = cv2.imread('images/task_2/left_0.png')
right_image = cv2.imread('images/task_2/right_0.png')
gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

l = cv2.FileStorage('parameters/left_camera_intrinsics.xml', cv2.FileStorage_READ)
l_mtx = l.getNode('camera_matrix').mat()
l_dist = l.getNode('distortion_coefficients').mat()
r = cv2.FileStorage('parameters/right_camera_intrinsics.xml', cv2.FileStorage_READ)
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
ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(world_space_points,
                                                                                              left_image_points,
                                                                                              right_image_points,
                                                                                              l_mtx,
                                                                                              l_dist,
                                                                                              r_mtx,
                                                                                              r_dist,
                                                                                              gray_left_image.shape[
                                                                                              ::-1], None,
                                                                                              None, None, None,
                                                                                              cv2.CALIB_FIX_INTRINSIC,
                                                                                              criteria)

s = cv2.FileStorage("parameters/stereo_calibration.xml",cv2.FileStorage_WRITE)
s.write('Translation_vector',T)
s.write('Rotation_matrix',R)
s.write('Fundamental_matrix',F)
s.write('Essential_matrix',E)
s.release()
# Step 4
R1 = np.eye(3,3)
t1= np.zeros((3,1))
P1 = np.concatenate((R1,t1),axis=1)
wp = np.array(world_space_points)*25.4
#board =
P2 = np.hstack((R,T.reshape(3,1)))


lpoints = cv2.undistortPoints(left_corners,cameraMatrix1,distCoeffs1)
rpoints = cv2.undistortPoints(right_corners,cameraMatrix2,distCoeffs2)

utils.undistort_save(cameraMatrix1,distCoeffs1,left_image,R1,"task_2/left_undistorted_not_rectified")
utils.undistort_save(cameraMatrix2,distCoeffs2,right_image,R,"task_2/right_undistorted_not_rectified")


p3d_points =cv2.triangulatePoints(P1,P2,lpoints,rpoints)
h,  w = left_image.shape[:2]

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1,distCoeffs1,cameraMatrix2,distCoeffs2,(w,h),R,T,R1,R,P1,P2,(w,h))


utils.undistort_save(cameraMatrix1,distCoeffs1,left_image,R1,"task_2/left_undistorted_rectified",P1)
utils.undistort_save(cameraMatrix2,distCoeffs2,right_image,R2,"task_2/right_undistorted_rectified",P2)


sr = cv2.FileStorage("parameters/stereo_rectification.xml", cv2.FileStorage_WRITE)
sr.write("R1",R1)
sr.write("R2",R2)
sr.write("P1",P1)
sr.write("P2",P2)
sr.write("Q",Q)
sr.release()




f =cv2.imread('output/task_2/left_undistorted_rectified.png')
f = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
retn, fcorners = cv2.findChessboardCorners(f, (8, 5), None)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
#ax.plot(l)
utils.pltcam(ax)
utils.pltcam(ax, R2,T)
plt.show()