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
    images = glob.glob('images/task_2/'+cam+'*.png')
    test_image = cv2.imread('images/task_2/'+cam+'_0.png')
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

            # Draw and display the corners
            cv2.drawChessboardCorners(image, (9, 6), corners2, ret)
            cv2.imwrite('output/task_1/'+image_file.split('.png')[0][-4:]+'_corner_points_annotation.png', image)

    