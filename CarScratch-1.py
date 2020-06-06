import numpy as np
import cv2 
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
           
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines_probabilistic(img):
    linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(line_img, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

region_ratios = [
    (0, 666),
    (0,0),
    (961, 3),
    (941, 195),
    (910,234 ),
    (771, 666),
]

kernel = np.ones((5,5),np.uint8)
kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

region_points = [(ratio[0], ratio[1]) for ratio in region_ratios]
vertices = np.array([region_points], dtype=np.int32)

img = cv2.imread('Data/23.jpg')
cv2.imshow('frame',img)
cv2.waitKey(0)
gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray_image',gray_image)
cv2.waitKey(0)
blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('blackhat',blackhat)
cv2.waitKey(0)
sharpened = cv2.filter2D(blackhat,-1,kernel_sharpening)
cv2.imshow('sharpened',sharpened)
cv2.waitKey(0)
edge_image = cv2.Canny(sharpened,50,150,apertureSize = 3)
cv2.imshow('edge_image',edge_image)
cv2.waitKey(0)
masked_edge_image = region_of_interest(edge_image, vertices)
cv2.imshow('masked_edge_image',masked_edge_image)
cv2.waitKey(0)
lines = hough_lines_probabilistic(masked_edge_image)
cv2.imshow('lines',lines)
cv2.waitKey(0)
result = weighted_img(lines, img)
cv2.imshow('result',result)
cv2.waitKey(0)

cv2.destroyAllWindows()