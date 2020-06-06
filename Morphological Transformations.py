import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Data/7.jpg',0)

kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
dilation = cv.dilate(img,kernel,iterations = 1)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

titles = ['image','erosion','dilation','opening','closing','gradient','tophat','blackhat']
images = [img, erosion, dilation, opening, closing, gradient,tophat,blackhat]

for i in range(8):
    cv.imshow(titles[i],images[i])
    cv.waitKey(0)

cv.destroyAllWindows()