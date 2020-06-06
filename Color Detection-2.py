import cv2
import numpy as np

frame=cv2.imread("Data/13.jpg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur1 = cv2.GaussianBlur(gray, (11, 11), 0)
edge = cv2.Canny(blur1,0,37,apertureSize = 3)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_white = np.array([0,0,186], dtype=np.uint8)
upper_white = np.array([179,17,255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_white, upper_white)
res = cv2.bitwise_and(edge,edge, mask= mask)

cv2.imshow('frame',frame)
cv2.waitKey()
cv2.imshow('mask',mask)
cv2.waitKey()
cv2.imshow('res',res)
cv2.waitKey()
cv2.destroyAllWindows()