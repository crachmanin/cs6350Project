import scipy as sp
import numpy as np
import os
import csv

data = sp.genfromtxt("features.csv",delimiter =",")
digitized = data[:]
all_buckets = []

for col in np.arange(128):
    min_value = data[:,col].min()
    max_value = data[:,col].max()

    #Initializing the bucket and fssss
    bucket = np.linspace(min_value, max_value, 20)
    digitized[:,col] = np.digitize(data[:,col], bucket)

    #Writing to bucket.config file
    all_buckets.extend([bucket.tolist()])

#Create a list of the newly bucketed feature and add label
features = digitized.astype(int).tolist()
features_fp = open("features.csv","r")

idx = 0
for f in features_fp:
    row = f.strip("\n").split(",")
    features[idx][-1] = row[-1]
    idx += 1
features_fp.close()

#Write the bucketed features to a file
bucketed_features_fp = open("bucketed_features.csv","wb")
writer = csv.writer(bucketed_features_fp)
writer.writerows(features)
bucketed_features_fp.close()

#Generating training (80%) and testing (20%) data
consolidated = np.asarray(features, dtype=str)
ran_array = np.random.randint(len(features), size = int(len(features) * 0.2))
test_data = consolidated[ran_array,:]
train_data = np.delete(consolidated, ran_array, axis=0)

#Writing test data to file
test_fp = open("test.csv", "wb")
test_writer = csv.writer(test_fp)
test_writer.writerows(test_data.tolist())
test_fp.close()
train_fp = open("train.csv", "wb")
train_writer = csv.writer(train_fp)
train_writer.writerows(train_data.tolist())
train_fp.close()

#Creating a config file to store bucket info for future reference
config_fp = open("buckets.config","wb")
config_writer = csv.writer(config_fp)
config_writer.writerows(all_buckets)
config_fp.close()
