import cv2
import numpy as np
import os
import csv
from sklearn.externals import joblib

'''Below function is currently not in use.
def kmean():
    return 1
'''

descriptor_list = []
feature_list = []
exceptions = open("ExceptionList","wb")  #File to list images whose features extraction failed
dataset_dir = "256_ObjectCategories"
feature_dir = "256_ObjectFeatures"

for dir_name in os.listdir(dataset_dir):

    label = dir_name[dir_name.find('.')+1:]     #To eliminate sequence number from dir name
    work_dir = os.path.join(dataset_dir, dir_name)

    #Creating Feature Directory
    if not os.path.exists(os.path.join(feature_dir, dir_name)):
        os.makedirs(os.path.join(feature_dir, dir_name))

    #if label.startswith('a'):   #Temporary : Just extracting features for image category starting with 'a'
    for images in os.listdir(work_dir):
        
        try:
            img_path = os.path.join(dataset_dir, dir_name, images)
            print img_path

            #Open the image and convert to grayscale
            img = cv2.imread(img_path)
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #Detect SIFT keypoints and load the feature Descriptors
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            
            #Storing feature descriptor for each image in a file
            #np.save(os.path.join(feature_dir, dir_name, images), des)    #Old - Required more space
            joblib.dump(des, os.path.join(feature_dir, dir_name, images+str('.desc')), compress = 3)
            
            #Vector Quantization using Averaging of columns
            quantized = np.empty(128)
            for col in np.arange(128):
                quantized[col] = des[:,col].sum()/len(des)

            #Adding quantized vector rounded to 3 decimals to our final feature list
            row = np.around(quantized, decimals=3).tolist()
            row.append(label)
            feature_list.extend([row])

        except Exception as exp:
            msg = "Image : " + img_path + "\nException : " + str(exp) + "\n"
            print msg, quantized, "\nDone Exception\n"
            exceptions.write(msg)

#Creating a consolidated quantized feature file
features_file = open("features.csv","wb")
writer = csv.writer(features_file)
writer.writerows(feature_list)

#Closing all open file
features_file.close()
exceptions.close()
