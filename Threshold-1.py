import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
frame = cv.imread('Data/13.jpg',0)

ret,thresh1 = cv.threshold(frame,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(frame,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(frame,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(frame,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(frame,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [frame, thresh1, thresh2, thresh3, thresh4, thresh5]

plt.figure(figsize=(16,16))

for i in range(6):
    plt.subplot(3,2,i+1),plt.imshow(images[i],cmap = 'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()