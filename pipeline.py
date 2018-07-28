import camera_calibration
import threshold
import lane_finding
import lane_finding2
import cv2
import sys
import numpy as np

input_img = sys.argv[1]
output_img = sys.argv[2]

first_detection = True
isCalibrated = False

def process_image(distorded_img, one_frame_only = False):
	global first_detection
	global isCalibrated
	global mtx, dist

	# Calibrate the camera to avoid distortion
	if not isCalibrated:
		mtx, dist = camera_calibration.calibrate_camera("camera_cal/")

	# Based on camera calibration parameters, undistort the image
	undistorded_img = camera_calibration.undistort_img(distorded_img, mtx, dist)
	
	# Apply color threshold on picture
	threshold_image = threshold.custom_threshold(undistorded_img)
	
	# Transform the image to see the lanes from a bird view
	bird_view_img, transform_matrix, inverse_transform_matrix = camera_calibration.transform(threshold_image)

	
	if first_detection:
		color_warp, curverad, offset_center = lane_finding.find_lane_pixels(bird_view_img)
		first_detection = False
		isCalibrated = True # Avoid calibrating the camera again for next frame of the video
	else:
		color_warp, curverad, offset_center = lane_finding.search_around_poly(bird_view_img)

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, inverse_transform_matrix, (distorded_img.shape[1], distorded_img.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undistorded_img, 1, newwarp, 0.3, 0)

	# export images at different steps of processing for analysis
	if one_frame_only == True:
		cv2.imwrite(append_it(output_img, "original"), distorded_img)
		cv2.imwrite(append_it(output_img, "undistorded"), undistorded_img)
		cv2.imwrite(append_it(output_img, "threshold"), np.uint8(threshold_image*255))
		cv2.imwrite(append_it(output_img, "birdview"), np.uint8(bird_view_img*255))
		cv2.imwrite(append_it(output_img, "lanefinding"), result)
	else:
		cv2.putText(result, "Curvature: %1.2f" % curverad, (10,70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (85,255,0),4)
		cv2.putText(result, "Offset: %1.2f" % offset_center, (10,130), cv2.FONT_HERSHEY_COMPLEX, 1.5, (85,255,0),4)
	
	return result

# Append a word between the filename and the extension
def append_it(filename, append_word):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + [append_word])



if __name__ == '__main__':

	# Read the image to be processed
	img = cv2.imread(input_img)
	process_image(img, True)


	
