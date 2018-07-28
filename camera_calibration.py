import numpy as np
import cv2
import os
import sys
import matplotlib.image as mpimg
import pickle
from pathlib import Path

input_img = sys.argv[1]
output_img = sys.argv[2]

# prepare object points
nx = 9 #Number of inside corners in x
ny = 6 #Number of inside corners in y

def transform(img_undist):

	corner_top_left = (img_undist.shape[1]/2-55, img_undist.shape[0]/1.58)
	corner_top_right = (img_undist.shape[1]/2+55, img_undist.shape[0]/1.58)
	corner_bottom_right = (img_undist.shape[1]-190, img_undist.shape[0])
	corner_bottom_left = (190, img_undist.shape[0])
	
	# By increasing the offset, the lines are closer to each other, permetting to detect a higher curvature of the lane
	offset = 300

	# define 4 source points src = np.float32([[,],[,],[,],[,]])
	src = np.float32([ corner_top_left, corner_top_right, corner_bottom_right, corner_bottom_left ])

	# define 4 destination points dst = np.float32([[,],[,],[,],[,]])
	dst = np.float32([[offset,0],[img_undist.shape[1]-offset,0],[img_undist.shape[1]-offset,img_undist.shape[0]],[offset,img_undist.shape[0]]]) 

	# use cv2.getPerspectiveTransform() to get M, the transform matrix
	M = cv2.getPerspectiveTransform(src, dst)

	# use cv2.getPerspectiveTransform() to get Minv, the inverse of transform matrix
	Minv = cv2.getPerspectiveTransform(dst, src)

	# d) use cv2.warpPerspective() to warp your image to a top-down view
	warped = cv2.warpPerspective(img_undist, M, (img_undist.shape[1], img_undist.shape[0]), flags=cv2.INTER_LINEAR)
	
	# Return the resulting image and matrix
	return warped, M, Minv
    

def calibrate_camera(calibration_img_folder):

	# To avoid running calibration calculations every time, datas are stored for later use
	# Check if calibration was already done before
	if os.path.isfile("calibration_data.dat"):
		print("Calibration datas were found, no need to redo camera calibration calculations")
		with open("calibration_data.dat", "rb") as filehandler:
			mtx, dist = pickle.load(filehandler)
	else:
		print("Calibration datas not found, camera calibration must be done")
		
		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d points in real world space
		imgpoints = [] # 2d points in image plane.
		
		# Prepare object points, like (0,0,0), (1,0,0), (2,0,0)...,(8,5,0)
		objp = np.zeros((nx*ny,3), np.float32)
		objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

		# Find 9x6 grid in the different calibration images
		for img in os.listdir(calibration_img_folder):
			img = mpimg.imread(calibration_img_folder + img)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
			
			if ret == True:
				objpoints.append(objp)
				imgpoints.append(corners)

		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		# save camera calibration parameters for future use
		with open("calibration_data.dat", "wb") as filehandler:
			pickle.dump([mtx, dist],filehandler)

	return mtx, dist

def undistort_img(img_dist, mtx, dist):
	return cv2.undistort(img_dist, mtx, dist, None, mtx)

# Arguments expected:
# 1: Path of image to be transformed
# 2: Path to store transformed image
if __name__ == '__main__':

	# Calibrate the camera
	mtx, dist = calibrate_camera("camera_cal/")
	
	img_dist = cv2.imread(input_img)

	# Undistort the image
	img_undist = undistort_img(img_dist, mtx, dist)

	# Create a birds eye view of the image
	bird_view_img, transform_matrix, inverse_transform_matrix = transform(img_undist)
	unwrap = cv2.warpPerspective(bird_view_img, inverse_transform_matrix, (bird_view_img.shape[1], bird_view_img.shape[0])) 
	
	cv2.imwrite(output_img, bird_view_img)
	cv2.imwrite("output_images/test.jpg", unwrap)
