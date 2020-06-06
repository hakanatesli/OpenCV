import cv2 
import numpy as np 

img = cv2.imread('Data/23.jpg') 

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

edges = cv2.Canny(gray,50,150,apertureSize = 3) 

lines = cv2.HoughLinesP(edges,1,np.pi/180,50,None,minLineLength=50,maxLineGap=10)

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)
    
cv2.imshow('deneme',img)
cv2.waitKey(0)
cv2.destroyAllWindows()