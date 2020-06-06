import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Data/13.jpg',0)
rows, cols = img.shape

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum1 = 20*np.log(np.abs(fshift))

ret, magnitude_spectrum2 = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(magnitude_spectrum2,cv.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv.dilate(opening,kernel,iterations=3)
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

ret, markers = cv.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0

dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum3 = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

crow,ccol = rows//2 , cols//2
fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)


plt.figure(figsize=(16,16))

plt.subplot(3,3,1),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,2),plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,3),plt.imshow(magnitude_spectrum3, cmap = 'gray')
plt.title('Magnitude Spectrum3'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,4),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,5),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,6),plt.imshow(magnitude_spectrum2)
plt.title('Magnitude Spectrum2'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,7),plt.imshow(unknown)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(3,3,8),plt.imshow(markers)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.show()

