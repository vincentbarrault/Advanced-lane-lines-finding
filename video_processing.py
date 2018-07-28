from moviepy.editor import VideoFileClip
import pipeline
import sys
import cv2

input_video = sys.argv[1]
output_video = sys.argv[2]

def process_video(video):
	return pipeline.process_image(video, False)

if __name__ == '__main__':
	
	#clip1 = VideoFileClip(input_video).subclip(42,50)
	clip1 = VideoFileClip(input_video)
	white_clip = clip1.fl_image(process_video) #NOTE: this function expects color images!!
	white_clip.write_videofile(output_video, audio=False)