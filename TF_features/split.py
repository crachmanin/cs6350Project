import numpy as np
from collections import defaultdict
import random

d1 = defaultdict(list)
whole_set = np.loadtxt("10Categories.csv", delimiter=",")

train = []
test = []

for idx in range(whole_set.shape[0]):
    d1[whole_set[idx,-1]].append(whole_set[idx, :])

for label in d1:
    test_idxs = set(random.sample(range(len(d1[label])), 20))
    for j, point in enumerate(d1[label]):
        if j in test_idxs:
            test.append(point)
        else:
            train.append(point)


np_train = np.vstack(train)
np_test = np.vstack(test)

np.savetxt("tensor_train.csv", np_train, fmt="%.10e")
np.savetxt("tensor_test.csv", np_test, fmt="%.10e")
