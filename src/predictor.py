import cv2
import time
import numpy
from keras.models import load_model
from ngraph_inf import NgraphInference
from train_model import alphabet, imageIntoData

class Predictor:
    def __init__(self, config):
        self.config = config
        self.engine = NgraphInference('keypoint_hand_model/keypoint.onnx', True)
        self.model = load_model('gesture_recognition_model.h5')
        self.nPoints = 22
        self.requiredProbability = 0.1

        # --- Set Skeleton 'Bones' (edges) based on 22 hand keypoints
        # --- returned by DNN model
        self.SKELETON_BONES = [ 
            [0,1], [1,2], [2,5], [5,9], [9,13], [13,17],[17,0], # Palm
            [2,3], [3,4], # Thumb
            [5,6], [6,7], [7,8], # Index Finger
            [9, 10], [10, 11], [11, 12], # Middle Finger
            [13, 14], [14,15], [15, 16], # Ring Finger
            [17, 18], [18,19], [19, 20] # Little Finger 
        ]

    def predict(self, hand):
        # -- Get Blob from hand image
        blob = cv2.dnn.blobFromImage(hand, 1.0/255, (self.config.hand_rows, self.config.hand_cols), 
                                                    (0,0,0), swapRB=False, crop=False)

        netOutput = self.engine.infer(blob)
        points = self.extract_keypoints(netOutput)
        centeredPoints = self.getCenteredKeypoints(points, self.config.hand_rows, self.config.hand_cols)
        transformedPoints = self.getTransformedKeypoints(centeredPoints, self.config.hand_rows, self.config.hand_cols)

        whiteClean = numpy.zeros((self.config.hand_rows, self.config.hand_cols, 1))
        whiteClean[:,:,:] = 255
        handGesture = numpy.copy(whiteClean)
        self.drawSkeleton(handGesture, transformedPoints, drawGesture=True)

        skeleton = numpy.copy(hand)
        self.drawSkeleton(skeleton, points)

        predictedLetter, prob = self.predictGesture(handGesture)

        return (predictedLetter, prob, skeleton, handGesture)

    def extract_keypoints(self, netOutput):
        points = []

        for keypointID in range(self.nPoints):
            # -- Get probability map for current keypoint
            probabilityMap = netOutput[0, keypointID, :, :]
            probabilityMap = cv2.resize(probabilityMap, (self.config.hand_rows, self.config.hand_cols))

            # -- Find most probable coords for current keypoint
            mapMinProbability, mapMaxProbability, mapMinPoint, mapMaxPoint = cv2.minMaxLoc(probabilityMap)

            # -- if probability that current coords are our keypoint exceeds 
            # -- set value we add this point to list  
            if mapMaxProbability >= self.requiredProbability:
                points.append((int(mapMaxPoint[0]), int(mapMaxPoint[1])))
            else:
                points.append(None)

        return points

    def getCenteredKeypoints(self, points, imgRows, imgCols):
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

    def getTransformedKeypoints(self, points, imgRows, imgCols):
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

    def drawSkeleton(self, image, keypoints, drawGesture=False):
        # -- Draw Skeleton based on detected keypoints
        for edge in self.SKELETON_BONES:
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

    def predictGesture(self, handGesture):
        # -- Extract training data from translated hand skeleton
        inputData = imageIntoData(handGesture, resize=True)
        # -- Predict current gesture skeleton, and print this prediction with given probability 
        res = self.model.predict(inputData)[0]
        y = np.argmax(res)
        predictedLetter = alphabet[y]

        return (predictedLetter, res[y])  
