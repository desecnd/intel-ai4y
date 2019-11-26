import os
import cv2
import sys
import time
import numpy as np 
from keras.models import load_model

from opencv_inf import OpencvInference
from ngraph_inf import NgraphInference

from train_model import alphabet, imageIntoData

# --- Load deep learning network 
# --- for hand recognition
# --- This model detects 22 hand keypoints 
# --- based on hand image 
# --- Model is taken from:
# --- https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/
# select one of the available inference engines for hand detection
# pass True as the second argument to create an engine that calculates an average latency of inferences for this engine
# hand_detection_engine = OpencvInference("keypoint_hand_model/pose_deploy.prototxt", "keypoint_hand_model/pose_iter_102000.caffemodel")
hand_detection_engine = NgraphInference('keypoint_hand_model/keypoint.onnx', True)
nPoints = 22
requiredProbability = 0.1  

# --- Set Skeleton 'Bones' (edges) based on 22 hand keypoints
# --- returned by DNN model
SKELETON_BONES = [ 
	[0,1], [1,2], [2,5], [5,9], [9,13], [13,17],[17,0], # Palm
	[2,3], [3,4], # Thumb
	[5,6], [6,7], [7,8], # Index Finger
	[9, 10], [10, 11], [11, 12], # Middle Finger
	[13, 14], [14,15], [15, 16], # Ring Finger
	[17, 18], [18,19], [19, 20] # Little Finger 
]

# --- Load last compiled (using trainModel.py script)
# --- gesture recognition model
model = load_model('gesture_recognition_model.h5')

def drawSkeleton(image, keypoints, drawGesture=False):
	# -- Draw Skeleton based on detected keypoints
	for edge in SKELETON_BONES:
		# -- edge = [ firstKeypointID, secondKeypointID ] tells us which keypoints we should connect
		keyA, keyB = edge
		
		# -- Both not None type
		if keypoints[keyA] and keypoints[keyB]:
			# -- Draw line between them on translated sheet, and not translated sheet
			
			if drawGesture == True:
				cv2.line(image, keypoints[keyA], keypoints[keyB], 0, 10, lineType=cv2.LINE_AA)
			else:
				# -- Draw colored skeleton on hand image 
				cv2.line(image, keypoints[keyA], keypoints[keyB], (255,0 ,0), 3, lineType=cv2.LINE_AA)
				cv2.circle(image, keypoints[keyA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.circle(image, keypoints[keyB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

def recenterHandKeypoints(points, img_rows, img_cols):
	# -- From points we take only which we have found
	foundPoints = [ p for p in points if p ]
	if not len(foundPoints):
		foundPoints.append([0,0])

	# -- Calculate Centroid of hand keypoints
	xCoordSum, yCoordSum = 0, 0
	for p in foundPoints:
		xCoordSum += p[0]
		yCoordSum += p[1]

	centroid = [xCoordSum // len(foundPoints), yCoordSum // len(foundPoints)]

	# -- Calculate Translation Vector, which if we add to every point, we will translate
	# -- our hand skeleton such that its centroid will land on middle of hand area
	# -- A -> B = [B.x - A.x, B.y - A.y]
	centerVector = [img_cols // 2 - centroid[0], img_rows // 2 - centroid[1]]

	# -- Get centered point entry for every keypoint index
	centeredPoints  =  [ (p[0] + centerVector[0], p[1] + centerVector[1]) if p else None for p in points ]

	return centeredPoints


def rescaleHandGesture(recenteredGesture):
	# -- Not made yet
	rescaledGesture = np.copy(recenteredGesture)
	return rescaledGesture


def processSnapshot(hand):
	# -- Copy current hand sector
	skeleton = np.copy(hand)

	hand_rows, hand_cols, = hand.shape[:2]

	# -- Blank canvas for drawing gestures
	dataoutRaw = np.zeros((hand_rows, hand_cols, 1))
	dataoutRaw[:,:,:] = 255
	rawCentered = np.copy(dataoutRaw)

	# -- Get Blob from hand image
	blob = cv2.dnn.blobFromImage(hand, 1.0/255 , (hand_rows, hand_cols), (0,0,0), swapRB=False, crop=False)

	# -- Detect keypoints in a hand
	netOutput = hand_detection_engine.infer(blob)
	points = []

	# -- Scan DNN output for every keypoint
	for keypointID in range(nPoints):
		# -- Get probability map for current keypoint
		probabilityMap = netOutput[0, keypointID, :, :]
		probabilityMap = cv2.resize(probabilityMap, (hand_rows, hand_cols))

		# -- Find most probable coords for current keypoint
		mapMinProbability, mapMaxProbability, mapMinPoint, mapMaxPoint = cv2.minMaxLoc(probabilityMap)

		# -- if probability that current coords are our keypoint exceeds 
		# -- set value we add this point to list  
		if mapMaxProbability >= requiredProbability:
			points.append((int(mapMaxPoint[0]), int(mapMaxPoint[1])))
		else:
			points.append(None)

	centeredPoints = recenterHandKeypoints(points, hand_rows, hand_cols)

	drawSkeleton(skeleton, points)
	drawSkeleton(rawCentered, centeredPoints, drawGesture=True)  

	rawRescaled = rescaleHandGesture(rawCentered)

	# -- Extract training data from translated hand skeleton
	inputData = imageIntoData(rawRescaled, resize=True)
	# -- Predict current gesture skeleton, and print this prediction with given probability 
	res = model.predict(inputData)[0]
	y = np.argmax(res)
	predictedLetter = alphabet[y]

	cv2.imshow('Skeleton on hand', skeleton)
	# cv2.imshow('Raw Centered', rawCentered)

	return (predictedLetter, res[y])  
