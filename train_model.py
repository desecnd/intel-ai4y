# ---- Authors: Pawel Wozniak Ewelina Tyma

# --- Import all necessary modules
import os
import cv2
import keras
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# --- Training parameters
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
	'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z']
rows = 28
cols = 28
batch_size = 128
epochs=10

# --- DNN Model Definition
# --- network topology is not fully created by us 
# --- credits for inspiration:
# --- https://github.com/acl21/Alphabet_Recognition_Gestures/blob/master/cnn_model_builder.py
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(rows,cols,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(alphabet), activation='softmax'))


# --- Compile created model
model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=keras.optimizers.Adadelta(),
	metrics=['accuracy'])

# --- Train model based on 'gestures' local database
X = []
y = []

# --- Get all record files for given letter and save them in X, y lists
for i in range(len(alphabet)):
	letter = alphabet[i]
	path = "gestures_database/" + letter + "/"
		
	# -- Get all record files in 'letter' gestures subdirectory
	for roots, dirs, samples in os.walk(path):
		for sampleFile in samples:
			fullPath = os.path.join(roots, sampleFile)
			sample = cv2.imread(fullPath)

			# -- Save sample image and correct prediction  
			X.append(cv2.cvtColor(cv2.resize(sample, (rows, cols)), cv2.COLOR_BGR2GRAY))
			y.append(i)

# --- Transform given (rows, cols) grayscale images 
# --- for DNN inputs
X = np.asarray(X)
X = X.reshape(X.shape[0], rows, cols, 1)

# --- Normalization [0-255] -> [0,1]
X = X.astype('float32')
X /= 255

# --- Transform predictions into binary matrix 
y = keras.utils.to_categorical(y, len(alphabet))

# --- Split database for 75% of training samples and 25% of testing samples 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0xdeadbeef)

# --- Train Mdodel for epochs times by bath_size amount of samples once
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# --- Check model accuracy
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# --- Save trained model for use in test script
model.save('gesture_recognition_model.h5')
