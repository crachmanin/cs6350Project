import numpy as np
import scipy as sp
import os
from sys import argv
from random import shuffle
from random import uniform
from sklearn.externals import joblib


def train_svm(train_set, epoch_size = 1, c = 1, gamma = 0.01):

    label_idx = len(train_set[0,:])-1
    weights = np.zeros(label_idx)
    t = 0
     
    for e in range(epoch_size):
        np.random.shuffle(train_set)
        for i in range(len(train_set)):
            d_prod = np.dot(weights.transpose(), train_set[i,:label_idx])
            #print("Dot Product : ", d_prod)
            t += 1
            gamma_t = gamma/(1 + (gamma * t/c))

            weights *= (1-gamma_t)
            if d_prod * train_set[i][label_idx] <= 1:
                weights += (gamma_t * c * train_set[i][label_idx]
                             * train_set[i,:label_idx])

        #print(weights)

    return weights[:]

def classify(test_set, labels, classifiers):
    
    #label_idx = len(test_set[0,:])-1
    mistakes = 0.0

    for i in range(len(test_set)):
        max_margin = 0.0
        label = None
        for key in classifiers.keys():
            d_prod = np.dot(classifiers[key].T, test_set[i,:])
            
            if d_prod > max_margin:
                max_margin = d_prod
                label = key

        #print label, labels[i]
        if label != labels[i]:
            mistakes += 1

    return float(mistakes/len(test_set))

def f_score(test_set, weights):
    label_idx = len(test_set[0,:])-1
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(test_set)):
        d_prod = np.dot(weights, test_set[i,:label_idx])

        if d_prod * test_set[i][label_idx] < 0:
            if test_set[i][label_idx] < 0:
                fp += 1
            else:
                fn += 1
        else:
            if test_set[i][label_idx] >= 0:
                tp += 1

    p = tp / (tp + fp) if (tp + fp) != 0 else 100
    r = tp / (tp + fn) if (tp + fn) != 0 else 100
    f = (2*p*r) / (p+r)

    print ("Precision :", p, " :: Recall :", r, " :: F-Score :", f)

    return f
            
def main(args):
    if len(args)<=1:
        histogram_dir = "Histograms"
        print "Enter the cluster value to check with : "
        cluster_size = input()

        k, cbook, train_set = joblib.load(os.path.join(histogram_dir, str(cluster_size) + "_train.var"))
        k, cbook, test_set = joblib.load(os.path.join(histogram_dir, str(cluster_size) + "_test.var"))
        
        label = np.unique(train_set[:,-1])
        classifiers = {}
               
        epoch = 20
        
        #Adding the bias term
        bias = 1
        train_set = np.insert(train_set, 0, bias, axis = 1)
        test_set = np.insert(test_set, 0, bias, axis = 1)
        
        
        #Training the multiclass SVM classifier for each label
        for lab in label:
            new_train_set = np.array(train_set, copy=True)
            new_train_set[:,-1] = np.asarray([new_train_set[:,-1] == lab]).astype(np.int)
            new_train_set = new_train_set.astype(np.float)
            classifiers[lab] = train_svm(new_train_set, epoch)
            
        for key in classifiers.keys():
            print key, "::", classifiers[key]
            
        print "Accuracy : ", 1 - classify(test_set[:,:-1].astype(np.float), test_set[:,-1], classifiers)
             
    #Below code is currently not used.    
    else:
            
        train_set = sp.genfromtxt(args[1], delimiter = " ")
        train_labels = sp.genfromtxt(args[2], delimiter = " ")
        test_set = sp.genfromtxt(args[3], delimiter = " ")
        test_labels = sp.genfromtxt(args[4], delimiter = " ")
    
        bias = 1
        bias_train_set = np.insert(train_set, 0, bias, axis = 1)
        train_set = np.insert(bias_train_set, len(bias_train_set[0,:]),
                              train_labels, axis = 1)
        bias_test_set = np.insert(test_set, 0, bias, axis = 1)
        test_set = np.insert(bias_test_set, len(bias_test_set[0,:]),
                              test_labels, axis = 1)
    
        epoch = 20
        if args[5] == '1':
            weights = train_svm(train_set[:], epoch)
            accuracy = classify(train_set, weights)
            print("Accuracy on train set : ", 1 - accuracy)
    
            accuracy = classify(test_set, weights)
            print("Accuracy on test set : ", 1 - accuracy)
            
        elif args[5] == '2':
            '''Cross Validation Logic'''
            cv_folds = 5
            avg_accuracy = 0
            best_accuracy = 0
            best_c = 0
            best_gamma = 0
            best_weights = []
            train_set_splits = np.array_split(train_set, cv_folds)
    
            print("C \t Gamma\t Avg_Accuracy")
    
            for c in [2, 1, -1, -2, -3, -4, -5]:
                for gamma in [0.01, 0.25, 0.50, 0.99]:
                    accuracy = 0
                    for i in range(cv_folds):
                        cv_train = []
                        
                        #Creating Train and Test splits
                        for j in range(cv_folds):
                            if i != j:
                                if cv_train == []:
                                    cv_train = train_set_splits[j]
                                else:
                                    cv_train = np.vstack((cv_train, train_set_splits[j]))
                                
                            else:
                                cv_test = train_set_splits[j]
                        
                        weights = train_svm(cv_train, epoch, 2**c, gamma)
                        accuracy += 1 - classify(cv_test, weights)
    
                    avg_accuracy = accuracy / cv_folds
                    print("%d \t %.2f %.3f" %(2**c, gamma, avg_accuracy))
    
                    if avg_accuracy > best_accuracy:
                        best_weights = weights
                        best_c = 2 ** c
                        best_gamma = gamma
                        best_accuracy = avg_accuracy
    
            print("\n\nBest Hyperparameters :: C :", best_c,
                  " | Gamma :", best_gamma, " | Accuracy :", best_accuracy)
            weights = train_svm(train_set, epoch, best_c, best_gamma)
            print("Accuracy of Best weight on Train Set : ",
                  1 - classify(train_set, weights))
            f_score(train_set, weights)
            print("Accuracy of Best weight on Test Set : ",
                  1 - classify(test_set, weights))
            f_score(test_set, weights)
            
                
if __name__ == "__main__":
    main(argv)
