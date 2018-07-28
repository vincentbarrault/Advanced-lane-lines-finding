import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pipeline
import time

left_fit = None
right_fit = None

array_size = 8
left_fit_list = [None] * array_size
right_fit_list = [None] * array_size
index = 0
successive_error = 0
first_frame = True

# Set of values used
last_valid_fit_values = np.zeros((2,3))#[None] * 6
last_valid_curverad = [None] * 2
last_valid_offset = None
last_valid_line_detection = np.zeros((4,1))

def find_lane_pixels(binary_warped):
    global left_fit
    global right_fit
    global left_fit_list
    global right_fit_list
    global array_size
    global index
	
    # Initialize weighted mean lists
    left_fit_list = [None] * array_size
    right_fit_list = [None] * array_size
    index = 0
    successive_error = 0
    
	# Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    color_warp = draw_line(left_fitx, right_fitx, ploty, binary_warped)

    # calculate curvature of lane
    left_curverad, right_curverad = calculate_curvature(binary_warped.shape[0], leftx, lefty, rightx, righty)
    mean_curverad = (left_curverad+right_curverad)/2
	
	# calculation center offset of the car
    offset = calculate_offset(binary_warped.shape, leftx, lefty, rightx, righty)

    return color_warp, mean_curverad, offset

def search_around_poly(binary_warped):
    global left_fit
    global right_fit
	
    # HYPERPARAMETER
    # width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Set the area of search based on activated x-values within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    color_warp = draw_line(left_fitx, right_fitx, ploty, binary_warped);
	
    # calculate curvature and car position
    left_curverad, right_curverad = calculate_curvature(binary_warped.shape[0], leftx, lefty, rightx, righty)
    mean_curverad = (left_curverad+right_curverad)/2
	
	# calculation center offset of the car
    offset = calculate_offset(binary_warped.shape, leftx, lefty, rightx, righty)
	
    return color_warp, mean_curverad, offset


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    global left_fit
    global right_fit
    global left_fit_list
    global right_fit_list
    global index
    global array_size
    global successive_error
    global last_valid_fit_values
    global last_valid_line_detection
    global first_frame
    global margin
	
	
    # Check if both lanes were detected (even with wrong values)
    #if not (lefty and leftx and rightx and righty):
    """if any(elem is None for elem in lefty) or \
        any(elem is None for elem in leftx) or \
        any(elem is None for elem in righty) or \
        any(elem is None for elem in rightx):"""
    if len(lefty) == 0 or \
        len(leftx) == 0 or \
        len(righty) == 0 or \
        len(rightx) == 0 :
        pipeline.first_detection = True; # At least one lane was not detected, restart lane finding
        [lefty, leftx, rightx, righty] = last_valid_line_detection
    else:
        last_valid_line_detection = [lefty, leftx, rightx, righty]

	
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

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

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    
    return left_fitx, right_fitx, ploty

def draw_line(left_fitx, right_fitx, ploty, warped):
	# Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (150,255, 0))

    return color_warp


def calculate_curvature(image_y_size, leftx, lefty, rightx, righty):
	global last_valid_curverad
	
	# Define y-value where we want radius of curvature
	# We'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = image_y_size
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
	leftx_cr = leftx * xm_per_pix
	lefty_cr = lefty * ym_per_pix
	rightx_cr = rightx * xm_per_pix
	righty_cr = righty * ym_per_pix

	try:
		left_fit_cr = np.polyfit(lefty_cr, leftx_cr, 2)
		right_fit_cr = np.polyfit(righty_cr, rightx_cr, 2)
	except (TypeError):
	    return last_valid_curverad[0], last_valid_curverad[1]
	
    # Calculation of R_curve (radius of curvature)
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	
	last_valid_curverad = [left_curverad, right_curverad]
	
	return left_curverad, right_curverad

def calculate_offset(image_size, leftx, lefty, rightx, righty):
    
    global last_valid_offset
	# Define y-value where we want radius of curvature
	# We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = image_size[0]
	
	# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except(TypeError):
        return last_valid_offset
    
    left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
	
	# Get the abscisse point in the middle of the 2 calculated lines
    midpoint_lanes_x = (left_fitx + right_fitx)/2
    midpoint_image = image_size[1]
	
    offset_pixel = midpoint_image - midpoint_lanes_x
    offset_meter = offset_pixel*3.7/np.abs(left_fitx - right_fitx)	
    last_valid_offset = offset_meter
	
    return offset_meter

def check_curverad(left_curverad, right_curverad):
    if (left_curverad > right_curverad + 10000) or (left_curverad < right_curverad - 10000):
        print("Curve radius difference between both lanes is too high")


def fit_polynomial(binary_warped):
    global left_fit
    global right_fit
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img


if __name__ == '__main__':

	out_img = fit_polynomial(binary_warped)

