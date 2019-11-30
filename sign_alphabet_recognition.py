# ---- Authors: Pawel Wozniak, Ewelina Tyma

# --- Import all necessary modules
import os
import cv2
import sys
import time
import numpy as np 
import argparse
from keras.models import load_model
from subprocess import Popen

from opencv_inf import OpencvInference
from ngraph_inf import NgraphInference

import train_model
import hand_processing
import prefix_queries
from utils import apputil

def parseArguments():
	parser = argparse.ArgumentParser(description="Sign alphabet recognition main script", usage = """
		program has to be run with --dictionary flag, which is path to file with keywords for words prediction	
	""")
	parser.add_argument("--dictionary", required=False, help="path to dictionary file")
	return parser.parse_args()

def checkDictionaryPath(dictionaryPath):
	try:
		f = open(dictionaryPath)
	except IOError:
		print("\nDictionary file: ", dictionaryPath, " does not exist, try --help flag for more information\n")
	finally:				
		f.close()
	
args = parseArguments()
dictionaryPath = "dictionaries/default.txt"
if args.dictionary:
	dictionaryPath = args.dictionary

checkDictionaryPath(dictionaryPath)

processHandle = prefix_queries.runProcess(dictionaryPath)

# --- Collecting Data for base mode
# --- You can choose to collect data for 
# --- future model compilation while using recognition tool
collectingMode = False
if input("Do you want to collect gestures data? (yes/[no]): ").replace(" ","") == 'yes':
	collectingMode = True

	# -- get letter for training
	letter = input('Choose letter which gestures you want to collect: ')
	while not (len(letter) == 1 and (letter in train_model.alphabet)):
		letter = input('Something gone wrong, choose again: ')	

	path = train_model.createNewLetterSession(letter)

# --- Set resolution and select webcam  
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- Check if camera is opened properly
if not cap.isOpened():
	print("Could not open video device")
	exit()

# --- Set hand area parameters

hand_rows = train_model.training_rows * 10
hand_cols = train_model.training_cols * 10

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
wordSuggestions = []

apputil.openWindows(hand_cols, hand_rows)

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
	cv2.putText(frame, predictedMessage[-25:], (20, 25),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, 1)


	# -- Show hand area and user's webcam view
	cv2.imshow('video', frame)

	# -- Get next user input
	userChoice = chr(cv2.waitKey(1) & 255)

	# -- If user pressed 'r' he enters (leaves)
	# -- recording mode in which every (current) frame
	# -- is scanned for keypoints then dumped as database record  
	if userChoice == 'r':
		if recordingON: 
			print("User pressed 'r' - Stopping recording mode,")	
			recordingON = False	
		else:
			print("User pressed 'r' - Started recording mode")
			recordingON = True

	# -- if user pressed 'q' leave main loop
	elif userChoice == 'q':
		print ("User pressed 'q', thanks for using our software, see you next time")
		break

	# -- if user pressed space, add space to predicted message
	elif userChoice == ' ':
		print("User pressed ' ' - Adding space to predicted message")
		predictedMessage += ' '							

	# -- if user pressed 'c', last character in message is cleaned     
	elif userChoice == 'x': 
		print("User pressed 'x' - erasing last message letter")
		if predictions > 0 and len(predictedMessage) > 0:
			predictedMessage = predictedMessage[:-1]

	# -- if user pressed 'x', erase whole message
	elif userChoice == 'c':
		print("User pressed 'c' - cleaning whole message")
		predictedMessage = ""

	elif userChoice > '0' and userChoice <= str(len(wordSuggestions)) and predictions > 0:
		print("User pressed '1-" + str(len(wordSuggestions)) +  "' predicting word")
		index = int(userChoice) - 1

		predictedMessage = apputil.append_word(predictedMessage, wordSuggestions[index])

	# -- RECOGNITION PHASE
	# -- if current mode is recording, or 
	# -- user has choosen option 's' (screenshot)
	# -- its first taken webcam frame
	if recordingON or userChoice == 's':
		if predictions == 0:
			predictedMessage =	""
	
		handSnapshot = np.copy(hand)
		skeleton, handGesture = hand_processing.drawHandGesture(hand)

		predictedLetter, prob = hand_processing.predictGesture(handGesture)  
		
		predictionMessage = "{} - {:3}% sure".format(predictedLetter, int(prob * 100))
		
		cv2.rectangle(handSnapshot, (0,0), (hand_cols, 40), (0,0,0), cv2.FILLED)
		cv2.putText(handSnapshot, predictionMessage , (20, 25),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, 1)

		print("User pressed 's' - taking hand area snapshot:  ", predictionMessage)
		predictions += 1
		predictedMessage += predictedLetter
		
		suggestions = np.zeros(hand.shape)
		cv2.putText(suggestions, "Suggestions:", (20, 25),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, 1)
		
		lastPrefix = predictedMessage.split()[-1]
		wordSuggestions = prefix_queries.queryProcess(processHandle, lastPrefix)
		
		for i in range(1, len(wordSuggestions) + 1):
			cv2.putText(suggestions, str(i)+". "+wordSuggestions[i - 1], (20, 45 * i + 25),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, 1)

		cv2.imshow('Letter Prediction', handSnapshot)
		cv2.imshow('Skeleton on hand', skeleton)
		cv2.imshow('Hand Gesture', handGesture)
		cv2.imshow('Word Suggestions', suggestions)

	# -- If selected mode is collecting data 
	# -- we can press 'd' (dump) to write current translated gesture
	# -- to file as data for future model compilation   
	if collectingMode and predictions > 0 and (recordingON or userChoice == 'd'):
		newRecord = "record" + str(dumpedRecords).zfill(4)
		cv2.imwrite(path + newRecord + ".jpg", handGesture)
		dumpedRecords += 1
		print("User pressed 'd' - Dumped new", letter, "letter gesture database record ", newRecord, "to: ", path)


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
prefix_queries.closeProcess(processHandle)
