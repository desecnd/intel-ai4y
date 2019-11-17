# ---- Authors: Pawel Wozniak, Ewelina Tyma

# --- Import all necessary modules
import os
import cv2
import sys
import time
import numpy as np 
from keras.models import load_model

from opencv_inf import OpencvInference
from ngraph_inf import NgraphInference

# --- Options for user: 
extendedOutput = False # change to 'True' to show more hand recognition layers

# --- Load deep learning network 
# --- for hand recognition
# --- This model detects 22 hand keypoints 
# --- based on hand image 
# --- Model is taken from:
# --- https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/

# select one of the available inference engines for hand detection
# pass True as the second argument to create an engine that calculates an average latency of inferences for this engine
# hand_detection_engine = OpencvInference("keypoint_hand_model/pose_deploy.prototxt", "keypoint_hand_model/pose_iter_102000.caffemodel")
hand_detection_engine = NgraphInference('keypoint_hand_model/keypoint.onnx')

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

# --- List of used letters in alphabet
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
	'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z']

# --- Load last compiled (using trainModel.py script)
# --- gesture recognition model
model = load_model('gesture_recognition_model.h5')

# --- Collecting Data for base mode
# --- You can choose to collect data for 
# --- future model compilation while using recognition tool
collectingMode = False
if input("Do you want to collect gestures data? (yes/no): ").replace(" ","") == 'yes':
	collectingMode = True

# --- Create next sample folder if collecting database mode is selected 
path = ""
if collectingMode:

	# Choose letter for collecting
	letter = input('Choose letter which gestures you want to collect: ')
	while not (len(letter) == 1 and (letter in alphabet)):
		letter = input('Something gone wrong, choose again: ')	

	# Search for last training folder
	path = "gestures_database/"+letter+"/"
	last = -1
	for r,d,f in os.walk(path):
		for folder in d:
			last = max(last, (int(folder)))

	# Create next data folder for current session 
	path += str(last + 1).zfill(3) 
	os.mkdir(path)
	path += "/"
	
# --- Training image Parameters
training_rows = 28
training_cols = 28

# --- Set resolution and select webcam  
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- Check if camera is opened properly
if not cap.isOpened():
	print("Could not open video device")
	exit()

# --- Set hand area parameters

hand_rows = 280
hand_cols = 280

# --- UpperLeft BottomRight corner coords for hand frame
ulx = 20
uly = 150 
brx = ulx + hand_cols
bry = uly + hand_rows

# --- Estimate time per frame for current webcam
print ("Estimating webcam time per frame...")
frames = 20
s = time.time()
for i in range(frames):
	ret, frame = cap.read()
e = time.time()
timePerFrame = (e - s)/frames

if extendedOutput:
	print("Estimated time per frame: ", timePerFrame)

# --- Set starting variables 
iteration = 0
predictions = 0
dumpedRecords = 0
predictedMessage = "Press 's' to take hand snapshot"
recordingON = False

# --- Start webcam video 
while(True):
	# -- Start time for frame time measure
	s = time.time()

	# -- Get next frame from webcam and cut out hand sector	
	ret, frame = cap.read()
	hand = frame[uly:bry, ulx:brx,:]
	
	# -- Draw Frame for hand sector
	cv2.rectangle(frame, (ulx, uly), (brx, bry), (0, 0, 255), 2)  	

	# -- Show last prediction
	cv2.rectangle(frame, (0,0), (frame.shape[1], 40), (0,0,0), cv2.FILLED)
	cv2.putText(frame, predictedMessage, (20, 25),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, 1)


	# -- Show hand area and user's webcam view
	cv2.imshow('video', frame)

	# -- Get next user input
	userChoice = cv2.waitKey(1)

	# -- RECOGNITION PHASE
	# -- if current mode is recording, or 
	# -- user has choosen option 's' (screenshot)
	# -- its first taken webcam frame
	if recordingON or ('s' == chr(userChoice & 255)):
		if predictions == 0:
			predictedMessage =	""
		
		# -- Copy current hand sector
		skeleton = np.copy(hand)
		handSnapshot = np.copy(hand)

		# -- Blank canvas for drawing gestures
		dataoutRaw = np.zeros((hand_rows, hand_cols, 1))
		dataoutRaw[:,:,:] = 255
		dataoutTranslated = np.copy(dataoutRaw)

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
		translationVector = [hand_cols // 2 - centroid[0], hand_rows // 2 - centroid[1]]
		
		# -- Draw Skeleton based on detected keypoints
		for edge in SKELETON_BONES:
			# -- edge = [ firstKeypointID, secondKeypointID ] tells us which keypoints we should connect
			keyA, keyB = edge

			# -- Both not None type
			if points[keyA] and points[keyB]:
				# -- 2 points translated by our Translation Vector
				transPointA = ( points[keyA][0] + translationVector[0], points[keyA][1] + translationVector[1] )
				transPointB = ( points[keyB][0] + translationVector[0], points[keyB][1] + translationVector[1] )

				# -- Draw line between them on translated sheet, and not translated sheet
				cv2.line(dataoutTranslated, transPointA, transPointB, 0, 10, lineType=cv2.LINE_AA)
				cv2.line(dataoutRaw, points[keyA], points[keyB], 0, 10, lineType=cv2.LINE_AA)
				
				# -- Draw colored skeleton on hand image 
				cv2.line(skeleton, points[keyA], points[keyB], (255,0 ,0), 3, lineType=cv2.LINE_AA)
				cv2.circle(skeleton, points[keyA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.circle(skeleton, points[keyB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


		# -- Extract training data from translated hand skeleton
		resized = cv2.resize(dataoutTranslated, (training_rows, training_cols))
		X = np.asarray(resized)
		X = X.reshape(1, training_rows, training_cols, 1)
		X = X.astype('float32')
		X /= 255

		
		# -- Predict current gesture skeleton, and print this prediction with given probability 
		res = model.predict(X)[0]
		y = np.argmax(res)
		predictedLetter = alphabet[y]
	
		predictionMessage = "{} - {:3}% sure".format(predictedLetter, int(res[y] * 100))
		cv2.rectangle(handSnapshot, (0,0), (hand_cols, 40), (0,0,0), cv2.FILLED)
		cv2.putText(handSnapshot, predictionMessage , (20, 25),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, 1)
		print("User pressed 's' - taking hand area snapshot:  ", predictionMessage)

		predictions += 1
		predictedMessage += predictedLetter

		# -- Output every sheet
		if extendedOutput:
			cv2.imshow('Raw gesture drawing', dataoutRaw)
			cv2.imshow('Resized data training', resized)
			cv2.imshow('Translated gesture skeleton', dataoutTranslated)
		cv2.imshow('Skeleton on hand', skeleton)
		cv2.imshow('Hand snapshot', handSnapshot) 
		
	# -- if user pressed 'ESC' leave main loop
	if userChoice == 27:
		print ("User pressed 'ESC', thanks for using our software, see you next time")
		break

	# -- If selected mode is collecting data 
	# -- we can press 'd' (dump) to write current translated gesture
	# -- to file as data for future model compilation   
	if collectingMode and (recordingON or 'd' == chr(userChoice & 255)):
		newRecord = "record" + str(dumpedRecords).zfill(4)
		cv2.imwrite(path + newRecord + ".jpg", dataoutTranslated)
		dumpedRecords += 1
		print("User pressed 'd' - Dumped new", letter, "letter gesture database record ", newRecord, "to: ", path)

	# -- If user pressed 'r' he enters (leaves)
	# -- recording mode in which every (current) frame
	# -- is scanned for keypoints then dumped as database record  
	if 'r' == chr(userChoice & 255):
		if recordingON: 
			print("User pressed 'r' - Stopping recording mode,")	
			recordingON = False	
		else:
			print("User pressed 'r' - Started recording mode")
			recordingON = True

	# -- if user pressed space, add space to predicted message
	if ' ' == chr(userChoice & 255):
		print("User pressed ' ' - Adding space to predicted message")
		predictedMessage += ' '							
	
	# -- if user pressed 'c', last character in message is cleaned     
	if 'x' == chr(userChoice & 255):
		print("User pressed 'x' - erasing last message letter")
		if predictions > 0 and len(predictedMessage) > 0:
			predictedMessage = predictedMessage[:-1]
	
	# -- if user pressed 'x', erase whole message
	if 'c' == chr(userChoice & 255):
		print("User pressed 'c' - cleaning whole message")
		predictedMessage = ""

	# -- Measure time taken for 1 frame processing
	e = time.time()
	timeSpent = e - s; 

	# -- Read amount of frames needed to equalize frame loss 
	framesToRead = int(timeSpent / timePerFrame)
	for i in range(framesToRead):
		ret, frame = cap.read()
	
	iteration += 1

# --- Close all windows and exit script
cap.release()
cv2.destroyAllWindows()
