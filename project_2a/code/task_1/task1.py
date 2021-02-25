# Task 1: Pinhole camera model and calibration
#
# As discussed in class, we use the pinhole camera model to represent the project geometry of a camera, shown.
# Usually, given a camera, the camera intrinsic parameters are unknown but can be calculated with known 3D-to-2D point
# correspondences (i.e., camera calibration). In this task, you are going to calibrate the camera using a few images.
# Please follow these steps.
#
# Step (1): Load the images. Please use the images in the provided resource files. For this task, the folder is "images/task_1".
# Since a stereo camera system is used, there are two sets of images with prefixes of "left_" and "right_", indicating
# which camera took the images. You are going to calibrate each individual camera separately, i.e., if you want to
# calibrate the left camera, use those images with prefixes of "left_". You can use the OpenCV library function
# "imread()" for this step.
import glob
import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


cams = ["left","right"]

for cam in cams:
    image_plane_points = []
    world_space_points = []
    images = glob.glob('images/task_1/'+cam+'*.png')
    test_image = cv2.imread('images/task_1/'+cam+'_2.png')
    gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    for image_file in images:
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step (2): Extract 3D-to-2D point correspondences.
        world_space_points.append(objp)
        ret, chessboard_corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, chessboard_corners, (11, 11), (-1, -1), criteria)
            image_plane_points.append(chessboard_corners)

            if image_file == 'images/task_1/'+cam+'_2.png':
                # Draw and display the corners
                cv2.drawChessboardCorners(image, (9, 6), corners2, ret)
                cv2.imwrite('output/task_1/'+cam+'_2_corner_points_annotation.png', image)
    #         cv2.waitKey(500)
    # cv2.destroyAllWindows()

    # Step (3): Calculate camera intrinsic parameters. Once the 3D-to-2D point correspondences are obtained, call OpenCV
    # library function "calibrateCamera()" to calculate the camera intrinsic matrix and distort coefficients.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_space_points, image_plane_points, gray_test_image.shape[::-1], None, None)
    # Step (4): Undistort the images of calibration board patterns with these parameters using the OpenCV library function
    # "initUndistortRectifyMap()" and "remap()".

    h,  w = test_image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(test_image, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    cv2.imwrite('output/task_1/'+cam+'_2_undistorted.png', dst)
    # Step (5): Save the parameters to a file. You can use OpenCV "FileStorage" class to write the intrinsic matrix.
    s = cv2.FileStorage("parameters/parameters.xml", cv2.FileStorage_WRITE)
    s.write("camera_matrix",mtx)
    s.write("distortion_coefficients",dist)
    s.release()


