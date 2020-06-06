import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('Data/13.jpg',0)

blur = cv.medianBlur(img,5)
ret,th1 = cv.threshold(blur,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

plt.figure(figsize=(16,16))

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],cmap = 'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()