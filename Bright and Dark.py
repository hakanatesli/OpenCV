import cv2 as cv
import numpy as np

img = cv.imread('Data/8.jpg',0)

B = np.ones(img.shape, dtype='uint8')*75

added = cv.add(img,B)
subtract = cv.subtract(img,B)

cv.imshow('added',added)
cv.waitKey(0)

cv.imshow('subtract',subtract)
cv.waitKey(0)

cv.destroyAllWindows()