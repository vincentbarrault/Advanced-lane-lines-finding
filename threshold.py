import matplotlib.image as mpimg
import numpy as np
import cv2
import sys
import pipeline


def region_of_interest(img, vertices):
    # Applies an image mask 

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    return dir_binary

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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

if __name__ == '__main__':

	input_img = sys.argv[1]
	
	#reading in an image
	image = mpimg.imread(input_img)

	# Choose a Sobel kernel size
	ksize = 3 # Choose a larger odd number to smooth gradient measurements

	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(15, 100))
	grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
	mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 100))
	dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(1, np.pi/2))
	
	if(len(sys.argv) == 3):
		output_img = sys.argv[2]
	else:
		output_img = sys.argv[1]

	# Multiply the output image (8 bit) by 255 to get the correct range
	cv2.imwrite(pipeline.append_it(output_img, "gradx"), np.uint8(gradx*255))
	cv2.imwrite(pipeline.append_it(output_img, "grady"), np.uint8(grady*255))
	cv2.imwrite(pipeline.append_it(output_img, "mag_binary"), np.uint8(mag_binary*255))
	cv2.imwrite(pipeline.append_it(output_img, "dir_binary"), np.uint8(dir_binary*255))