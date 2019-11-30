import cv2
from numpy import zeros

def openWindows(cols, rows):
    blank_img = zeros((cols, rows))
    cv2.imshow('video', zeros((100, 100)))
    cv2.imshow('Hand Gesture', blank_img)
    cv2.imshow('Skeleton on hand', blank_img)
    cv2.imshow('Letter Prediction', blank_img)
    cv2.imshow('Word Suggestions', blank_img)
    cv2.moveWindow('video', 500, 125)
    cv2.moveWindow('Hand Gesture', 0, 0)
    cv2.moveWindow('Skeleton on hand', 0, 450)
    cv2.moveWindow('Letter Prediction', 1200, 0)
    cv2.moveWindow('Word Suggestions', 1200, 450)