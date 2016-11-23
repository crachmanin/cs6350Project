from scipy.cluster.vq import vq, kmeans, whiten
from matplotlib.pyplot import plot, draw, show
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

feature_dir = "256_ObjectFeatures"
histogram_dir = "Histograms"
all_descriptors = []

#Creating random train and test splits from the data
print "Creating Random training and testing set"
train_descriptors = []
test_descriptors = []

for dir_name in os.listdir(feature_dir):

    label = dir_name[dir_name.find('.')+1:]     #To eliminate sequence number from dir name
    work_dir = os.path.join(feature_dir, dir_name)
    counter = 0
    total_imgs = len(os.listdir(work_dir))

    train_idx = np.random.randint(total_imgs, size=int(total_imgs * 0.60))
        
    for files in os.listdir(work_dir):
        #file_descriptors = np.load(os.path.join(feature_dir, dir_name, files))
        file_descriptors = joblib.load(os.path.join(feature_dir, dir_name, files))
        if counter in train_idx:
            train_descriptors.append((label, file_descriptors))
        else:
            test_descriptors.append((label, file_descriptors))
        counter += 1

all_descriptors = train_descriptors[:]

print "Converting to Stacked Numpy Array"
l_count = {}
arr_descriptors = all_descriptors[0][1]
for label, desc in all_descriptors[1:]:
    if label not in l_count.keys():
        l_count[label] = 0
    if l_count[label] < 50:
        print label
        arr_descriptors = np.vstack((arr_descriptors, desc))

#K-mean clustering

for k in [20, 75, 100, 128, 160, 200, 250, 300]:
    print "Calculating K-mean for k :", k
    cbook, distortion = kmeans(arr_descriptors, k, 1)
    
    # Calculate the histogram of features for training and testing data
    print "Calculating the histogram of features for training data"
    
    im_train = np.zeros((len(train_descriptors), k), "float32")
    train_label = []
    for i in xrange(len(train_descriptors)):
        words, distance = vq(train_descriptors[i][1],cbook)
        train_label.append(train_descriptors[i][0])
        for w in words:
            im_train[i][w] += 1
            
    train_label = np.asarray(train_label, dtype=np.str)
    im_train = im_train.astype(np.str)
    
    train_set = np.insert(im_train, len(im_train[0,:]), train_label, axis=1)
    
    print "Calculating the histogram of features for testing data"
    
    im_test = np.zeros((len(test_descriptors), k), "float32")
    test_label = []
    for i in xrange(len(test_descriptors)):
        words, distance = vq(test_descriptors[i][1],cbook)
        test_label.append(test_descriptors[i][0])
        for w in words:
            im_test[i][w] += 1
    
    test_label = np.asarray(test_label, dtype=np.str)
    im_test = im_test.astype(np.str)
    
    test_set = np.insert(im_test, len(im_test[0,:]), test_label, axis=1)
    
    print "Printing Histogram to file"
    if not os.path.exists(histogram_dir):
        os.mkdir(histogram_dir)
    
    print train_set
    
    joblib.dump((k, cbook, train_set),
                os.path.join(histogram_dir, str(k)+"_train.var"), compress = 3)
    joblib.dump((k, cbook, test_set),
                os.path.join(histogram_dir, str(k)+"_test.var"), compress =3)
