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

from train_model import training_rows, training_cols
import hand_processing

# --- Collecting Data for base mode
# --- You can choose to collect data for 
# --- future model compilation while using recognition tool
collectingMode = False
if input("Do you want to collect gestures data? (yes/no): ").replace(" ","") == 'yes':
	collectingMode = True

	# -- get letter for training
	letter = input('Choose letter which gestures you want to collect: ')
	while not (len(letter) == 1 and (letter in alphabet)):
		letter = input('Something gone wrong, choose again: ')	

	path = hand_processing.createNewLetterSession(letter)

# --- Set resolution and select webcam  
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- Check if camera is opened properly
if not cap.isOpened():
	print("Could not open video device")
	exit()

# --- Set hand area parameters

hand_rows = training_rows * 10
hand_cols = training_cols * 10

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
	
		handSnapshot = np.copy(hand)
		predictedLetter, prob = hand_processing.processSnapshot(hand)  
		
		predictionMessage = "{} - {:3}% sure".format(predictedLetter, int(prob * 100))
		
		cv2.rectangle(handSnapshot, (0,0), (hand_cols, 40), (0,0,0), cv2.FILLED)
		cv2.putText(handSnapshot, predictionMessage , (20, 25),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, 1)

		print("User pressed 's' - taking hand area snapshot:  ", predictionMessage)
		predictions += 1
		predictedMessage += predictedLetter
		
		cv2.imshow('Letter Prediction', handSnapshot)

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

	# print(timeSpent)

	# -- Read amount of frames needed to equalize frame loss 
	framesToRead = int(timeSpent / timePerFrame)
	for i in range(framesToRead):
		ret, frame = cap.read()
	
	iteration += 1

# --- Close all windows and exit script
cap.release()
cv2.destroyAllWindows()
