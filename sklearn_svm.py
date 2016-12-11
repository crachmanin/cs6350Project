from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
import scipy as sp
import os

def main():
    
    ''' This thing is for training with the clustered dataset'''
    histogram_dir = "Histograms"
    print "Enter the cluster value to check with : "
    cluster_size = input()

    k, cbook, train_set = joblib.load(os.path.join(histogram_dir, str(cluster_size) + "_train.var"))
    k, cbook, test_set = joblib.load(os.path.join(histogram_dir, str(cluster_size) + "_test.var"))
    '''

    train_set = sp.genfromtxt("train.csv", delimiter=",", dtype=str)
    test_set = sp.genfromtxt("test.csv", delimiter=",", dtype=str)
    '''
    label = np.unique(train_set[:,-1])
    classifiers = {}

    #Adding the bias term
    bias = 1
    train_set = np.insert(train_set, 0, bias, axis = 1)
    test_set = np.insert(test_set, 0, bias, axis = 1)

    print "Training the classifier : "           
    lsvm = LinearSVC()
    t_svm = lsvm.fit(train_set[:,:-1].astype(np.float), train_set[:,-1])

    mistakes = 0.0

    print "Predicting the labels"
    for idx in range(len(test_set)):
        label = t_svm.predict(test_set[idx,:-1].reshape(1,-1).astype(np.float))
        if test_set[idx,-1] != label[0]:
            mistakes += 1
        #print test_set[idx,-1], label[0]

    print "\n\nAccuracy : ", 1-(mistakes/len(test_set))
                
if __name__ == "__main__":
    main()
