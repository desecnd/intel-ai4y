import os
import cv2
import sys
import time
import numpy as np 
from keras.models import load_model

from opencv_inf import OpencvInference
# from ngraph_inf import NgraphInference

from train_model import alphabet, imageIntoData

# --- Load deep learning network 
# --- for hand recognition
# --- This model detects 22 hand keypoints 
# --- based on hand image 
# --- Model is taken from:
# --- https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/
# select one of the available inference engines for hand detection
# pass True as the second argument to create an engine that calculates an average latency of inferences for this engine
hand_detection_engine = OpencvInference("keypoint_hand_model/pose_deploy.prototxt", "keypoint_hand_model/pose_iter_102000.caffemodel")
# hand_detection_engine = NgraphInference('keypoint_hand_model/keypoint.onnx', True)
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

def getCenteredKeypoints(points, imgRows, imgCols):
	# -- From points we take only which we have found
	foundPoints = [ p for p in points if p ]

	if len(foundPoints) < 2:
		return [ p for p in points ]

	minRow = imgRows + 1 
	maxRow = -1 
	minCol = imgCols + 1 
	maxCol = -1 

	for x,y in foundPoints:
		minRow = min(minRow, y)
		maxRow = max(maxRow, y)
		minCol = min(minCol, x)
		maxCol = max(maxCol, x)

	centerBoxRow = minRow + (maxRow - minRow)//2
	centerBoxCol = minCol + (maxCol - minCol)//2
	centerVector = (imgCols//2 - centerBoxCol, imgRows//2 - centerBoxRow)

	centeredPoints  =  [ (p[0] + centerVector[0], p[1] + centerVector[1]) if p else None for p in points ]
	return centeredPoints

def getTransformedKeypoints(points, imgRows, imgCols):
	# -- Not made yet
	foundPoints = [ p for p in points if p ]

	if len(foundPoints) < 2:
		return [p for p in points]

	minRow = imgRows + 1 
	maxRow = -1 
	minCol = imgCols + 1 
	maxCol = -1 

	for x,y in foundPoints:
		minRow = min(minRow, y)
		maxRow = max(maxRow, y)
		minCol = min(minCol, x)
		maxCol = max(maxCol, x)

	boxRows = maxRow - minRow 
	boxCols = maxCol - minCol
	
	minRowDist = min(minRow, imgRows - maxRow) 
	minColDist = min(minCol, imgCols - maxCol)
	
	border = 5

	if minRowDist < minColDist:
		scalar = (boxRows + 2*(minRowDist - border)) / boxRows 
	else: 
		scalar = (boxCols + 2*(minColDist - border)) / boxCols 

	centerRow = imgRows // 2
	centerCol = imgCols // 2
	
	transformedPoints  =  [ (centerCol + int(scalar*(p[0] - centerCol)), centerRow + int(scalar*(p[1] - centerRow))) if p else None for p in points ]
	return transformedPoints


def drawHandGesture(hand):
	# -- Copy current hand sector
	skeleton = np.copy(hand)

	handRows, handCols, = hand.shape[:2]

	# -- Blank canvas for drawing gestures
	whiteClean = np.zeros((handRows, handCols, 1))
	whiteClean[:,:,:] = 255

	# -- Get Blob from hand image
	blob = cv2.dnn.blobFromImage(hand, 1.0/255 , (handRows, handCols), (0,0,0), swapRB=False, crop=False)

	# -- Detect keypoints in a hand
	netOutput = hand_detection_engine.infer(blob)
	points = []

	# -- Scan DNN output for every keypoint
	for keypointID in range(nPoints):
		# -- Get probability map for current keypoint
		probabilityMap = netOutput[0, keypointID, :, :]
		probabilityMap = cv2.resize(probabilityMap, (handRows, handCols))

		# -- Find most probable coords for current keypoint
		mapMinProbability, mapMaxProbability, mapMinPoint, mapMaxPoint = cv2.minMaxLoc(probabilityMap)

		# -- if probability that current coords are our keypoint exceeds 
		# -- set value we add this point to list  
		if mapMaxProbability >= requiredProbability:
			points.append((int(mapMaxPoint[0]), int(mapMaxPoint[1])))
		else:
			points.append(None)

	centeredPoints = getCenteredKeypoints(points, handRows, handCols)
	transformedPoints = getTransformedKeypoints(centeredPoints, handRows, handCols)

	handGesture = np.copy(whiteClean)

	drawSkeleton(skeleton, points)
	drawSkeleton(handGesture, transformedPoints, drawGesture=True)  

	return (skeleton, handGesture)


def predictGesture(handGesture):
	# -- Extract training data from translated hand skeleton
	inputData = imageIntoData(handGesture, resize=True)
	# -- Predict current gesture skeleton, and print this prediction with given probability 
	res = model.predict(inputData)[0]
	y = np.argmax(res)
	predictedLetter = alphabet[y]

	return (predictedLetter, res[y])  
