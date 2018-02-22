import numpy as np
import cv2
from scipy import ndimage
import math


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def preprocess(img):
	img=255-np.array(img).reshape(28,28).astype(np.uint8)
	(thresh, gray) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	while np.sum(gray[0]) == 0:
		gray = gray[1:]

	while np.sum(gray[:,0]) == 0:
		gray = np.delete(gray,0,1)

	while np.sum(gray[-1]) == 0:
		gray = gray[:-1]

	while np.sum(gray[:,-1]) == 0:
		gray = np.delete(gray,-1,1)

	rows,cols = gray.shape

	if rows > cols:
		factor = 20.0/rows
		rows = 20
		cols = int(round(cols*factor))
		gray = cv2.resize(gray, (cols,rows))
	else:
		factor = 20.0/cols
		cols = 20
		rows = int(round(rows*factor))
		gray = cv2.resize(gray, (cols, rows))

	colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
	rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
	gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

	shiftx,shifty = getBestShift(gray)
	shifted = shift(gray,shiftx,shifty)
	gray = shifted

	img = gray.reshape(1,28,28).astype(np.float32)

	img-= int(33.3952)
	img/= int(78.6662)
	return img
