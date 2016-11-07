import cv2
import numpy as np
import scipy as sp
from sys import getsizeof

data = np.random.random(100)
#print data
bins = np.linspace(0, 1, 10)
#print bins
digitized = np.digitize(data, bins)
#print digitized
bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
#print bin_means
digitized_mean = np.digitize(data, bin_means)
#print digitized_mean

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 5, 10], [11, 3, 5]], np.int32)
print x[x[:2] == 5]
x = []
print x
    
try:
    img = cv2.imread("256_ObjectCategories\\009.bear\\009_0044.jpg")
    print getsizeof(img)

    #resized_img = sp.misc.imresize(image, 0.5)
    #print getsizeof(resized_img)

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)

    img=cv2.drawKeypoints(gray,kp)
    cv2.imwrite('C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\sift_keypoints.jpg',img)

    img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('C:\\Users\\Madhur\\Desktop\\Python\\ML\\Project\\sift_keypoints_rich.jpg',img)

    kp, des = sift.detectAndCompute(gray,None)

    #print kp
    #print des

except Exception as msg:
    print "Exception in image : "  + str(msg)
    
