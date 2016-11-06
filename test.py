import cv2
import numpy as np

img = cv2.imread('256_ObjectCategories/001.ak47/001_0002.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

#img=cv2.drawKeypoints(gray,kp)

#cv2.imwrite('sift_keypoints.jpg',img)

kp, des = sift.detectAndCompute(gray,None)
print type(des)
print len(des[0])
