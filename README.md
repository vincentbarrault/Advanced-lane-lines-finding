# **Advanced lane finding**

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Code

The project contains the following python files:
* pipeline.py: Main pipeline calling the different functions processing each frame of the video.
* video_processing.py: Process the video by sending each frame to the pipeline.
* camera_calibration.py: Calibrate the camera, correct the distortion of the image and apply perspective transform to get a bird's eye view of the lanes.
* threshold.py: Apply filters to create a thresholded binary image.
* lane_finding.py: Determine lane pixels, curvature of the lane and vehicle position with respect to the center of the lanes before displaying the lane boundaries on the original image.

The script is run by using python and providing 2 parameters:

      python video_processing.py project_video.mp4 project_video_output.mp4

The first parameter is the path for the input video.
The second parameter is the path where the processed video will be saved.

![](https://github.com/vincentbarrault/Advanced-lane-lines-finding/blob/master/project_video_output.gif?raw=true)

In the same way, for processing a single image, run the following:

      python pipeline.py image_input.mp4 image_output.mp4

## 1. Camera calibration

Located in `camera_calibration.py`, the script uses `cv2.findChessboardCorners` to find corners in an image of a chessboard pattern. The same process is repeated with about 20 chessboards pictures taken from different angles to collect enough `imgpoints` (array of corners found). Object points are stored in a second array, `objpoints`. Both imgpoints are parameters of the camera calibration function c`v2.calibrateCamera` which returns distortion coefficients.

The distortion coefficients are stored in a file (calibration_data.dat) using `pickle` to not recalculate them everytime the script is run.

The distortion correction is then applied to the image with `cv2.undistort`:
![](https://raw.githubusercontent.com/vincentbarrault/Advanced-lane-lines-finding/master/test_images/test3.jpg | width=100)
*distorded image*

![](https://raw.githubusercontent.com/vincentbarrault/Advanced-lane-lines-finding/master/output_images/test3_undistorded.jpg | width=100)
*undistorded image*

## 2. Image threshold

To isolate and filter the lanes in the picture, the R channel from the RGB picture and the S (saturation) channel from the HLS picture are combined together.

    def custom_threshold(image):
	    img = np.copy(image)

	    R = img[:,:,0]
	    G = img[:,:,1]
	    B = img[:,:,2]

	    # Isolate R channel (RGB) from the picture
	    thresh = (200, 255)
	    binary_R = np.zeros_like(R)
	    binary_R[(R > thresh[0]) & (R <= thresh[1])] = 1
	    
	    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	    H = hls[:,:,0]
	    L = hls[:,:,1]
	    S = hls[:,:,2]
	    
	    # Isolate S channel (HLS) from the picture
	    thresh = (110, 255)
	    binary_S = np.zeros_like(S)
	    binary_S[(S > thresh[0]) & (S <= thresh[1])] = 1
	    
	    # Combine both channel to identify lanes
	    color_binary = np.dstack(( np.zeros_like(binary_S), binary_S, binary_R)) * 255
	    
	    combined_binary = np.zeros_like(binary_S)
	    combined_binary[(binary_S == 1) | (binary_R == 1)] = 1
	    
	    return combined_binary

The threshold has been chosen after various tests and the result is as follows:

![](https://raw.githubusercontent.com/vincentbarrault/Advanced-lane-lines-finding/master/output_images/test3_threshold.jpg | width=100)
*Thresholded binary image*

## 3. Perspective transform

The area (source `src` and destination `dst`) on which the transformation is applied were chosen by tuning in an empirical manner the 4 corners of `src` in a trapezoidal shape. This will represent a rectangle when looking down on the road from above.

To calculate the transformation matrix and its inverse, the function `cv2.getPerspectiveTransform` was used. Using `cv2.warpPerspective` with the transformation matrix and the image in parameters, the script returns returns a bird's eye view of the lanes.

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

![](https://github.com/vincentbarrault/Advanced-lane-lines-finding/blob/master/output_images/test3_birdview.jpg?raw=true| width=100)
INCLUDE IMAGE OF BIRDS VIEW

## 4. Lane finding

This part of the script calculates the histogram on the X axis to find which pixels are part of the lines and which belong to the left line and which belong to the right line.

The two highest peaks from the histogram are used as a starting point for determining where the lane lines are, before sliding windows moving upward in the image (further along the road) to determine where the lane lines go.

After finding all pixels belonging to each line through the sliding window method, a polynomial fit is calculated (`np.polyfit`) for each lane to get their polynomial curve and then determine their radius (value displayed in the video). 

A weighted mean list (`array_size = 8`) was also implemented to compensate lane detections (from threshold binary image) which were not good enough. It also smoothen the movement of the lines displayed on the video. Furthermore, if the left and right lines have a slope which is too different (lines not aligned), it means that the values are probably wrong. Those are excluded from the calculations. If this happens in 3 frames in a row, the weighted average list is reset to update it with values which are more actual and not display "old" lines which wouldn't fit the latest frames.

    margin = 0.45

    # Exclude the lines found if the difference of slope between 2 lines is higher than margin
    if not first_frame and (left_fit[1] > right_fit[1] + margin) or (right_fit[1] > left_fit[1] + margin):
        successive_error = successive_error+1
        # Reject values with wrong slope. If it happens 3 times in a row, reset weighted average list
        if(successive_error==3):
            pipeline.first_detection = True;
    else: # lanes found have correct values, add them to weighted average
        first_frame = False
        successive_error = 0
        left_fit_list[index] = left_fit
        right_fit_list[index] = right_fit
        index = (index + 1) % array_size


    # If weighted average list is empty, load the last valid fit values
    if all(element is None for element in left_fit_list):
        left_fit = last_valid_fit_values[0]
        right_fit = last_valid_fit_values[1]
    else: #else make the average of all elements in the list
        if any(elem is None for elem in left_fit_list):
            left_fit = sum(left_fit_list[:(index)])/(index)
            right_fit = sum(right_fit_list[:(index)])/(index)
        else:
            left_fit = sum(left_fit_list)/array_size
            right_fit = sum(right_fit_list)/array_size
        last_valid_fit_values = [left_fit, right_fit] # save the latest valid fit values

The result of the lane boundaries is then warped back to return to the original view of the picture and displayed on top of the original image:

![](https://github.com/vincentbarrault/Advanced-lane-lines-finding/blob/master/output_images/test3_lanefinding.jpg?raw=true | width=100)
*Processed image after lane finding detection*