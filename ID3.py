from sys import argv
from math import log, sqrt
import numpy as np
import scipy as sp

#Defining the tree node class
class ID3Tree:
    'Nodes of the ID3 algorithm'
    
    def __init__(self, t_set, f_used, depth, link_val=None):
        global tree_depth
        self.attribute = None
        self.depth = depth
        self.properties = []
        self.child = []
        self.label = None
        self.f_used = f_used
        self.t_set = t_set
        self.values = {}
        self.link_val = link_val
        tree_depth = (self.depth if self.depth >= tree_depth else tree_depth)
        self.findBestFeature()
        
        
    def __repr__(self, level=0):
        ret = "\n" + "\t"*level+repr(self.attribute)+" - "+repr(self.label) + "   " +repr(self.depth)
        for child in self.child:
            ret += child.__repr__(level+1)
        return ret
        
    def setTreeNode (self, attribute, label=None):
        self.attribute = attribute
        self.f_used.append(attribute)
        self.label = label

    def findBestFeature(self):
        global set_label
        global label_idx
        global depth_limit
        l = None
        depth = 0;
        max_gain = 0
        max_gain_feat = None
        
        if len(self.t_set) > 0:
            t_entropy = entropy(self.t_set)
            if t_entropy == 0:
                self.label = self.t_set[0][label_idx]
            else:
                for f in np.arange(128):
                    if f not in self.f_used:
                        gain = calcInfoGain(t_entropy, self.t_set, f)
                        if max_gain < gain:
                            max_gain = gain
                            max_gain_feat = f
                            
                self.setTreeNode(max_gain_feat)
                print("Node created : ", max_gain_feat)
                                
                if self.depth+1 > depth_limit:
                    self.label = findMaxLabel(self.t_set)
                
                else:
                    for value in np.unique(self.t_set[:,max_gain_feat]):        #20 is the size of the bucket and each feature has value ranged 1..20
                        n_set = generateSubSet(self.t_set, max_gain_feat, value)
                        c_node = ID3Tree(n_set, self.f_used[:], self.depth+1, value)
                        self.values[value] = c_node
                        self.child.append(c_node)
                        if c_node.label is None:
                            c_node.label = findMaxLabel(self.t_set)
                    self.t_set = []
            
        
    def classify (self, test_set):
        global tree_depth
        
        if self.child:
            value = test_set[self.attribute]
            if value in self.values.keys():
                c_node = self.values[value]
            else:
                return self.label
            tree_depth = self.depth
            return c_node.classify(test_set)
        else:
            tree_depth = (self.depth if tree_depth<self.depth else tree_depth)
            return self.label
            

#Defining the entropy function
#This function calculates the entropy of the training set
def entropy(t_set):
    global set_label
    global label_idx
    lbl = list(set_label)
    unique_labels = len(set_label)
    total_data = len(t_set)
    lbl_list = np.asarray(list(set_label))
    ent_list = np.zeros(unique_labels)
    entropy = 0

    for i in np.arange(unique_labels):
        ent_list[i] = np.sum(t_set[:,label_idx] == lbl_list[i])

    for i in np.arange(unique_labels):
        if ent_list[i] > 0:
            entropy += (-1*(ent_list[i]/total_data)*log((ent_list[i]/total_data), 2))
                
    return entropy;

    
def generateSubSet(t_set, f_no, value):
    try:
        if len(t_set) > 1:
            subset  = t_set[t_set[:,f_no] == value]
        elif t_set[f_no] == value:
            subset = t_set
        else:
            subset = []
    except Exception as msg:
        print("Exception : ", msg)
        print("Set", t_set, "\n", f_no, value, "\nSubset", subset, "\n")
    return subset;
    

#Defining the info gain function
#This function calls the "entropy function" for each attribute and calculates the information gain
def calcInfoGain(l_entropy, t_set, feature_no):
    global label
    total = 0
    a_entropy = 0
    gain = 0
    if len(t_set) > 0:
        attrib_values = set()

        #Calculating entropy for each element in the feature
        for av in np.unique(t_set[:,feature_no]):

            n_set = generateSubSet(t_set, feature_no, av)
            a_entropy += len(n_set)/len(t_set) * entropy(n_set)
            
        gain = l_entropy - a_entropy
    return gain;
    

#Function to train the ID3 tree using the training data    
def train(training_set, f_added, user_limit=16000):
    global depth_limit 
    depth_limit = (user_limit if user_limit > 0 else depth_limit)
    tree = ID3Tree(training_set, f_added, 0)
    return tree


#Function to test the accuracy of the ID3 Tree
def test_accuracy(tree, test_set):
    global label_idx
    correct = 0
    for n in np.arange(len(test_set)):
        tree_label = tree.classify(test_set[n,:])
        if test_set[n,label_idx] == tree_label:
            correct += 1
    accuracy = correct/len(test_set)
        
    return accuracy
    

#Finding the label which occurs most 
def findMaxLabel(t_set):
    global set_label
    global label_idx
    lbl_list = np.asarray(list(set_label))
    ent_list = np.zeros(len(lbl_list))

    for i in np.arange(len(lbl_list)):
        ent_list[i] = np.sum(t_set[:,label_idx] == lbl_list[i])

    return lbl_list[ent_list.argmax()]
    
    
#Function for checking accuracy with cross validation.
#Needs to be a new module once completed.
def crossValidation (main_training_set, main_test_set, method=5):
    global tree_depth
    global replace_data
    depth_set = {1,2,3,4,5,10,15,20}
    max_mean = 0
    max_method = 0
    data_sets = [[],[],[],[],[],[]]
    hyper_param = 0
    l_dataset = [[],[],[],[],[],[]]
    og_train = main_training_set[:]
    og_test = main_test_set[:]
    
    for i in range(0,6):
        f = open(argv[4]+"training_0"+str(i)+".data", "r")
        for line in f:
            d_entry = line.strip("\n").split(",")
            data_sets[i].append(d_entry)
            l_dataset[i].append(tuple(d_entry))
    
    if int(method) > 0 and int(method) <=3:
        for meth in range(1, 4):
            mean = 0
            std_dev = 0
            count = 0
            acc = []
            
            for i in range(0,6):
                training_set = []
                testing_set = []
                            
                for j in range(0,6):
                    if i==j:
                        testing_set.extend(l_dataset[j])
                    else:
                        training_set.extend(l_dataset[j])
                
                if meth != 3:
                    #train_set = handleMissingFeature(training_set[:], meth)
                    #test_set = handleMissingFeature(testing_set[:], meth)
                    cv_tree = train(handleMissingFeature(training_set[:], training_set[:], meth), [])
                    acc.append(test_accuracy(cv_tree, handleMissingFeature(training_set[:], testing_set[:], meth)))

                #Building the ID3 tree for each of the k-1 datasets and training on the remaining 1 dataset
                cv_tree = train(training_set, [])
                acc.append(test_accuracy(cv_tree, testing_set))
            
            mean = sum([x for x in acc])/len(acc)
            std_dev = sqrt(sum([(a - mean)**2 for a in acc])/len(acc))
            
            print("\nMethod : ", meth)
            print("Average accuracy using Method ", meth, ": ", mean)
            print("Standard Deviation at Method ", meth, ": ", std_dev)
        
            if mean > max_mean:
                max_mean = mean
                max_method = meth
        
        print("\nSubQuestion c:")
        print("--------------")
        
        print("\nBest Method : ", max_method, " Best Accuracy : ", max_mean)
        
        cm_tree = train(handleMissingFeature(og_train, og_test, max_method), [])
        print("\nAccuracy after training overall set : ", test_accuracy(cm_tree, handleMissingFeature(main_training_set[:], main_test_set[:], max_method)))    
    
    else: 
        print("SubQuestion a")
        print("-------------")
        
        for depth in depth_set:            
            mean = 0
            std_dev = 0
            count = 0
            acc = []
            tree_depth = 0
                    
            for i in range(0,6):
                training_set = []
                test_set = []
                            
                for j in range(0,6):
                    if i==j:
                        test_set.extend(data_sets[j])
                    else:
                        training_set.extend(data_sets[j])
                
                #Building the ID3 tree for each of the k-1 datasets and training on the remaining 1 dataset
                cv_tree = train(training_set, [], depth)
                acc.append(test_accuracy(cv_tree, test_set))
                
            mean = sum([x for x in acc])/len(acc)
            std_dev = sqrt(sum([(a - mean)**2 for a in acc])/len(acc))
            
            print("\nDepth Limit Set : ", depth)
            print("Average accuracy at Depth ", depth, ": ", mean)
            print("Standard Deviation at Depth ", depth, ": ", std_dev)
            
            if mean > max_mean:
                max_mean = mean
                hyper_param  = depth
        
        print("\nSubQuestion b:")
        print("--------------")
        
        print("\nBest Depth : ", hyper_param, " Best Accuracy : ", max_mean)
                    
        cm_tree = train(main_training_set, [], hyper_param)
        #cm_tree.__repr__
        print("\nAccuracy after training overall set : ", test_accuracy(cm_tree, main_test_set))

    
#Function to handle missing data in the training set`
def handleMissingFeature(training_set, replace_set, method):
    global label_idx
    global feature
    global set_label
    
    t_set = []
    r_set = []
    
    max_feat = None
    max_count = 0
    l_missing = set()
    l_label = list(set_label)
    
    m_feat_count = [0] * len(set_label)
    m_feat_val = [''] * len(set_label)
    
    #finding the feature with the missing data
    for i in range(0, len(training_set)):
        t_set.append(list(training_set[i]))
        for j in range(0, label_idx):        
            if (t_set[i][j] is None or t_set[i][j] == '?'):
                l_missing.add(j+1)
    
    if method == 1:
        for f in l_missing:
        
            for x in range(2, len(feature[str(f)]),2):
                attrib_value = feature[str(f)][x]
                attrib_count = 0
            
                for data in range(0, len(t_set)):
                    if t_set[data][f-1] == attrib_value and t_set[data][f-1] != '?':
                        attrib_count += 1
                
                if max_count < attrib_count:
                    max_count = attrib_count
                    max_feat = attrib_value
        
            
            for data in range(0, len(replace_set)):
                r_set.append(list(replace_set[data]))
                r_set[data][f-1] = (max_feat if r_set[data][f-1]=='?' else r_set[data][f-1])

            
    if method == 2:
        for f in l_missing:
            for x in range(2, len(feature[str(f)]),2):
                attrib_value = feature[str(f)][x]
                l_feat_count = [0] * len(set_label)
    
                for data in range(0, len(t_set)):
                    if t_set[data][f-1] == attrib_value and t_set[data][f-1] != '?':
                        for idx in range(0,len(l_label)):
                            l_feat_count[idx] += (1 if t_set[data][label_idx] == l_label[idx] else 0)
                
                for idx in range(0,len(l_label)):
                    if(l_feat_count[idx] > m_feat_count[idx]):
                        m_feat_count[idx] = l_feat_count[idx]
                        m_feat_val[idx] = attrib_value
        
            for data in range(0, len(replace_set)):
                r_set.append(list(replace_set[data]))
                for idx in range(0,len(l_label)):
                    r_set[data][f-1] = (m_feat_val[idx] if r_set[data][f-1]=='?' else r_set[data][f-1])

    return r_set[:]
    

"""Where the magic happens."""

training_set = sp.genfromtxt("train.csv", dtype=str, delimiter=",")
label = training_set[:,-1]
'''
training_set = sp.genfromtxt("bf_test.csv", dtype=str, delimiter=",")
label = training_set[:,-1]
'''
test_set = sp.genfromtxt("test.csv", dtype=str, delimiter=",")


label_idx = 128
set_label = set(label)
depth_limit = 10
tree_depth = 0
f_added = []

tree = train(training_set, f_added, depth_limit)
print("Max Tree Depth : ", tree_depth)
print("Error Rate : ", (1- test_accuracy(tree, test_set)))

print("\n----------------------------------------------------------------------------------------------------\n")
print("Tree Representation : \n")
print(tree.__repr__)
