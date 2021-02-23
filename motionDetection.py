import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
import imutils
from imutils.video import VideoStream
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import PowerTransformer
import math

def frameDeltaMotionDetection(args, vs):
	previousFrame = None
	while True:
		frame = vs.read()
		if args.get("video", None) == None:
			frame = frame
		else:
			frame = frame[1]

		#end of video
		if frame is None:
			cv2.destroyAllWindows()
			break

		frame = imutils.resize(frame, width=500, height=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (25, 25), 0)

		#base state, first frame of video
		if previousFrame is None:
			previousFrame = gray
			continue

		frameDelta = cv2.absdiff(previousFrame, gray)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

		thresh = cv2.dilate(thresh, None, iterations=10)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)

		for c in cnts:
			if cv2.contourArea(c) < args["min_area"]:
				continue

			#c = cv2.approxPolyDP(c, 0.2*cv2.arcLength(c, True), True)

			# try to fill contour
			#cv2.fillPoly(c, pts =[c], color=(255,255,255))
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imshow("Video", frame)
		cv2.imshow("Threshold (binary)", thresh)
		cv2.imshow("Frame Delta", frameDelta)

		previousFrame = gray

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			cv2.destroyAllWindows()
			break

def SIFTMotionDetection(args, vs):
	sift = cv2.SIFT_create() #maybe play around with parameters to try and make better
	bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
	previousFramesN = []
	prevKPN = []
	prevDesN = []
	previousFramesM = []
	prevKPM = []
	prevDesM = []
	bufferN = 10 #number of frames a point must be new for it to be detected as motion
	bufferM = 1 #number of frames a point must be moving for it to be detected as motion
	vectorMarginOfError = 1 #num pixels the components of a movement vector must be greater than to be significant
	percentDifferenceMarginOfError = 1 #value that the difference between x and y components of 2 vectors must less than to be significantly similar
	while True:
		t1 = time.time()
		frame = vs.read()
		if args.get("video", None) == None:
			frame = frame
		else:
			frame = frame[1]

		#end of video
		if frame is None:
			break

		frame = imutils.resize(frame, width=500, height=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (25, 25), 0)

		kp, des = sift.detectAndCompute(gray,None)
		cv2.drawKeypoints(frame, kp, frame)

		if prevDesN != [] and previousFramesN != []:
			newPoints = [] #point, count
			oldPoints = [] #point, mvtVector, count
			# for each frame in past buffer frames
			for i in range(len(prevKPN)):
				#feature matching
				matches = bf.match(des, prevDesN[i])
				matchesIdx = [match.queryIdx for match in matches] #index of pt in kp
				# matches = sorted(matches, key = lambda x:x.distance)
				# img = cv2.drawMatches(frame, kp, previousFrame, prevKP, matches, previousFrame, flags=2)
				# plt.imshow(img),plt.show()
				# for each KP in that frame
				for j in range(len(kp)):
					#compare to KPs of current frame
					#new KP => object comes into frame/something moving(?)
					if j not in matchesIdx:
						#if comparing to first frame of buffer, all new points are new
						if i == 0:
							newPoints.append([kp[j], 1])
						#comparing to other frames, check if different KPs are already in newPoints and increment counter
						#only increments counter of those already in list
						#forces points to be "new" for buffer frames to be detected
						elif kp[j] in [point[0] for point in newPoints]:
							newPoints[[point[0] for point in newPoints].index(kp[j])][1] += 1

				#not new for all buffer frames
				badPoints = []
				for j in range(len(newPoints)):
					if newPoints[j][1] != i+1:
						badPoints.append(newPoints[j])
				for point in badPoints:
					newPoints.remove(point)

			prevKPM.append(kp)
			prevDesM.append(des)

			for i in range(1, len(prevKPM)):
				matches = bf.match(prevDesM[i], prevDesM[i-1])
				matchesIdx = [match.queryIdx for match in matches]
				#old points moving
				for j in range(len(prevKPM[i])):
					if j in matchesIdx:
						mvtVector = ((prevKPM[i][j].pt[0] - prevKPM[i-1][matches[matchesIdx.index(j)].trainIdx].pt[0]), (prevKPM[i][j].pt[1] - prevKPM[i-1][matches[matchesIdx.index(j)].trainIdx].pt[1]))
						# magnitude = math.sqrt((kp[j].pt[0] - prevKP[i][matches[matchesIdx.index(j)].trainIdx].pt[0]) ** 2) + ((kp[j].pt[1] - prevKP[i][matches[matchesIdx.index(j)].trainIdx].pt[1]) ** 2)
						# unitMvtVector = ((kp[j].pt[0] - prevKP[i][matches[matchesIdx.index(j)].trainIdx].pt[0])/magnitude, (kp[j].pt[1] - prevKP[i][matches[matchesIdx.index(j)].trainIdx].pt[1])/magnitude)

						#make sure movement vector is nonzero
						if abs(mvtVector[0]) > vectorMarginOfError and abs(mvtVector[1]) > vectorMarginOfError:
							#no movements have been recorded yet
							if i == 1:
								oldPoints.append([prevKPM[i][j], mvtVector, 1])
							#comparing to other frames, ignores points that aren't already in list (moving for less than buffer frames)
							elif prevKPM[i-1][matches[matchesIdx.index(j)].trainIdx] in [point[0] for point in oldPoints]:
								oldPoints[[point[0] for point in oldPoints].index(prevKPM[i-1][matches[matchesIdx.index(j)].trainIdx])][2] += 1
								oldPoints[[point[0] for point in oldPoints].index(prevKPM[i-1][matches[matchesIdx.index(j)].trainIdx])][1] = mvtVector
								oldPoints[[point[0] for point in oldPoints].index(prevKPM[i-1][matches[matchesIdx.index(j)].trainIdx])][0] = prevKPM[i][j]

				#not moving for all buffer frames
				badVectors = []
				for j in range(len(oldPoints)):
					if oldPoints[j][2] != i:
						badVectors.append(oldPoints[j])
				for point in badVectors:
					oldPoints.remove(point)

			#draws a bounding box around new points
			#only helpful if bufferM is large
			#relative to the time a moving object spends on screen
			if newPoints != []:
				(x, y, w, h) = cv2.boundingRect(np.array([point[0].pt for point in newPoints], dtype=np.int32))
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
				#if w > 1 and h > 1:
					#box = frame[x:x+w, y:y+h]
					#cv2.imshow("Motion", box)
					#cv2.waitKey(0)

			#groups points into clusters based on the movement vector, and draws boxes around them
			similarClusters = []
			if oldPoints != []:
				cluster = []
				oldPoints = sorted(oldPoints, key = lambda x:x[1][0]) #sort by mvtVector x component
				for i in range(1, len(oldPoints)):
					#see if x components and y components of two vectors are within some margin of error
					if abs((oldPoints[i][1][0]-oldPoints[i-1][1][0])/oldPoints[i-1][1][0]) < percentDifferenceMarginOfError and abs((oldPoints[i][1][1]-oldPoints[i-1][1][1])/oldPoints[i-1][1][1]) < percentDifferenceMarginOfError:
						if cluster == []:
							cluster.append(oldPoints[i-1][0])
						cluster.append(oldPoints[i][0])
					else:
						if len(cluster) > 1:
							similarClusters.append(cluster)
							cluster = []

			if similarClusters != []:
				for cluster in similarClusters:
					(x, y, w, h) = cv2.boundingRect(np.array([point.pt for point in cluster], dtype=np.int32))
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
					# if w > 1 and h > 1:
					# 	box = frame[x:x+w, y:y+h]
					# 	cv2.imshow("Motion", box)
					# 	cv2.waitKey(0)

		if len(prevKPN) < bufferN and len(prevDesN) < bufferN:
			prevKPN.append(kp)
			prevDesN.append(des)
		else:
			prevKPN.pop(0)
			prevKPN.append(kp)
			prevDesN.pop(0)
			prevDesN.append(des)

		if len(prevKPM) > bufferM and len(prevDesM) > bufferM:
			prevKPM.pop(0)
			prevDesM.pop(0)

		#base state, first frame of video
		if len(previousFramesN) == 0:
			previousFramesN.append(gray)

		if len(previousFramesM) == 0:
			previousFramesM.append(gray)

		t2 = time.time()
		cv2.putText(frame, "%2.2f FPS" % (1/(t2-t1)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

		cv2.imshow("Video", frame)

		if len(previousFramesN) < bufferN:
			previousFramesN.append(gray)
		else:
			previousFramesN.pop(0)
			previousFramesN.append(gray)

		if len(previousFramesM) < bufferM:
			previousFramesM.append(gray)
		else:
			previousFramesM.pop(0)
			previousFramesM.append(gray)

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			cv2.destroyAllWindows()
			break

def backgroundRemovalMotionDetection(args, vs):
	previousFrame = None
	fgbg = cv2.createBackgroundSubtractorMOG2()
	while True:
		frame = vs.read()
		if args.get("video", None) == None:
			frame = frame
		else:
			frame = frame[1]

		#end of video
		if frame is None:
			cv2.destroyAllWindows()
			break

		frame = imutils.resize(frame, width=500, height=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (25, 25), 0)

		#base state, first frame of video
		if previousFrame is None:
			previousFrame = gray
			continue

		fgmask = fgbg.apply(frame)

		cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)

		for c in cnts:
			if cv2.contourArea(c) < args["min_area"]:
				continue
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imshow("Video", frame)
		cv2.imshow("Mask", fgmask)

		previousFrame = gray

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			cv2.destroyAllWindows()
			break

def main():
	#get video
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", help="path to the video file")
	ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
	args = vars(ap.parse_args())

	#no video, using webcam instead
	if args.get("video", None) is None:
		vs = VideoStream(src=0).start()
		frameDeltaMotionDetection(args, vs)
		SIFTMotionDetection(args, vs)
		backgroundRemovalMotionDetection(args,vs)
		time.sleep(2.0)
	else:
		vs1 = cv2.VideoCapture(args["video"])
		vs2 = cv2.VideoCapture(args["video"])
		vs3 = cv2.VideoCapture(args["video"])
		frameDeltaMotionDetection(args, vs1)
		SIFTMotionDetection(args, vs2)
		backgroundRemovalMotionDetection(args,vs3)

main()
