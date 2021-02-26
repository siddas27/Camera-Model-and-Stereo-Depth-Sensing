import cv2
import numpy as np
import matplotlib.pyplot as plt


def pltcam(ax, R_prime=np.identity(3), t_prime=np.zeros((3, 1))):
    f = 1
    tan_x = 1
    tan_y = 1

    cam_center_local = np.asarray([
        [0, 0, 0], [tan_x, tan_y, 1],
        [tan_x, -tan_y, 1], [0, 0, 0], [tan_x, -tan_y, 1],
        [-tan_x, -tan_y, 1], [0, 0, 0], [-tan_x, -tan_y, 1],
        [-tan_x, tan_y, 1], [0, 0, 0], [-tan_x, tan_y, 1],
        [tan_x, tan_y, 1], [0, 0, 0]
    ]).T

    cam_center_local *= f
    cam_center = np.matmul(R_prime, cam_center_local) + t_prime

    ax.plot(cam_center[0, :], cam_center[1, :], cam_center[2, :],
            color='k', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def undistort(cameraMatrix, distCoeffs, image, R_):
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 0, (w, h))

    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R_, newcameramtx, (w, h), 5)
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    return dst