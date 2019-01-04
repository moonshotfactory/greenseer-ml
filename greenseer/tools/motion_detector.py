# USAGE
# Based on https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import os
from skimage.measure import compare_ssim as ssim



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-s", "--skip", type=int, default=500, help="skip initial frames")
ap.add_argument("--crop-width", type=int, default=-1, help="crop width of a frame")
ap.add_argument("--crop-height", type=int, default=-1, help="crop height of a frame")
ap.add_argument("--output-dir", default="extracted_images", help="output directory")


args = vars(ap.parse_args())

output_dir = args['output_dir']

if not os.path.isdir(output_dir):
	os.makedirs(output_dir)


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

count = 0
for _ in range(args['skip']):
	vs.read()
	count += 1

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video

saved_hash = []


import skimage.io as skio

def save_to_file(count, output_dir, frame):
	filename = 'frame_{}.jpg'.format(count)
	filepath = os.path.join(output_dir, filename)
	skio.imsave(filepath, frame)

while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"


	original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
	
	if args['crop_width'] > 0:
		frame = frame[:, :args['crop_width'], :]
	if args['crop_height'] > 0:
		frame = frame[:args['crop_height'], :, :]
	
	count += 1


	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None or count % 25 == 0:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

	if text == "Occupied":
		continue
	
	if not saved_hash:
		save_to_file(count, output_dir, original)
		saved_hash = [gray]
		continue

	import numpy as np

	is_duplicate = False
	for _gray in saved_hash: 
		score = (gray - _gray).mean()
		yy = ssim(gray, _gray)
		print(yy, score)
		if score <  100:
			is_duplicate = True
			break
	if not is_duplicate:
		save_to_file(count, output_dir, original)
		saved_hash.append(gray)

	# cv2.waitKey(0)

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()