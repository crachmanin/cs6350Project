from sklearn import svm
import numpy as np

whole_set = np.loadtxt("all.csv", delimiter=",")
np.random.shuffle(whole_set)
num_points = whole_set.shape[0]
training, test = whole_set[:num_points*4/5,:], whole_set[num_points*4/5:,:]

train_features = training[:,:-1]
train_labels = training[:,-1]

test_features = test[:,:-1]
test_labels = test[:,-1]

lin_clf = svm.LinearSVC()
lin_clf = svm.LinearSVC()
lin_clf.fit(test_features, test_labels) 

preds = lin_clf.predict(test_features)

mistakes = np.count_nonzero(preds - test_labels)
accuracy = 1 - float(mistakes)/len(preds)

print accuracy
