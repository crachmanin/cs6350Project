import numpy as np
from collections import Counter, defaultdict, deque
import os
import math
import glob
import timeit

unique_vals = []

class Node:
    def __init__(self, feature_num, entropy, rows, used_columns, classification):
        self.feature_num = feature_num
        self.entropy = entropy
        self.rows = rows
        self.used_columns = used_columns
        self.classification = classification
        self.children = {}


    def __str__(self):
        return str(self.feature_num) + "\n" + str(self.children.keys()) + " " + str(self.classification)


def gen_unique(dataset):
    global unique_vals
    unique_vals = []
    for j in xrange(dataset.shape[1]):
        unique_vals.append(set(dataset[:, j]))


def entropy(dataset, rows):
    num_columns = dataset.shape[1]
    c1 = Counter()
    for i in rows:
        c1[dataset[i][num_columns - 1]] += 1
    total = float(sum(c1.values()))
    return -1 * sum([float(val)/total * math.log(float(val)/total, 2) for val in c1.values()])


def ig(base_entropy, dataset, column_num, filtered_rows):
    row_lists = defaultdict(list)
    for i in filtered_rows:
        row_lists[dataset[i][column_num]].append(i)
    weighted_entropies = [(entropy(dataset, rows) * len(rows)) for rows in row_lists.values()]
    return base_entropy - sum(weighted_entropies)/len(filtered_rows)


def max_gain_idx(gains):
    curr_max = 0
    max_idx = -1
    for item in gains:
        if item[1] > curr_max:
            curr_max = item[1]
            max_idx = item[0]
    return max_idx


def get_avg_of_rows(dataset, rows):
    c1 = Counter()
    num_columns = dataset.shape[1]
    for i in rows:
        c1[dataset[i][num_columns - 1]] += 1
    return c1.most_common(1)[0][0]


def get_avg_by_feature(dataset, column_num, value):
    c1 = Counter()
    num_columns = dataset.shape[1]
    for i in xrange(dataset.shape[0]):
        if dataset[i][column_num] == value:
            c1[dataset[i][num_columns - 1]] += 1
    if not c1:
        return get_avg_of_rows(dataset, xrange(dataset.shape[0]))
    return c1.most_common(1)[0][0]


def find_root(dataset, rows, used_columns, curr_depth, max_depth):
    base_entropy = entropy(dataset, rows)
    if base_entropy == 0:
        return Node(-1, 0, [], used_columns, dataset[rows[0], dataset.shape[1] - 1])
    if curr_depth == max_depth:
        avg = get_avg_of_rows(dataset, rows)
        return Node(-1, 0, [], used_columns, avg)
    free_columns = set(range(dataset.shape[1] - 1)).difference(used_columns)
    gains = [(j, ig(base_entropy, dataset, j, rows)) for j in free_columns]
    max_column = max_gain_idx(gains)
    new_used_columns = set(used_columns)
    new_used_columns.add(max_column)
    root = Node(max_column, base_entropy, rows, new_used_columns, None)
    return root


def id3(node, dataset, curr_depth, max_depth):
    if node.classification or curr_depth == max_depth:
        return
    row_lists = defaultdict(list)
    for i in node.rows:
        row_lists[dataset[i][node.feature_num]].append(i)
    for val in unique_vals[node.feature_num]:
        if val not in row_lists:
            avg = get_avg_by_feature(dataset, node.feature_num, val)
            node.children[val] = Node(-1, 0, [], node.used_columns, avg)
        else:
            node.children[val] = find_root(dataset, row_lists[val],
                    node.used_columns, curr_depth + 1, max_depth)
    for child in node.children.values():
        id3(child, dataset, curr_depth + 1, max_depth)


def BFS(root):
    q1 = deque()
    q1.append((root, 0))
    depth = -1
    while q1:
        curr, depth = q1.popleft()
        for child in curr.children.values():
            q1.append((child, depth + 1))
    return depth


def get_depth(data):
    gen_unique(data)
    root = find_root(data, range(data.shape[0]), set(), 1, -1)
    id3(root, data, 1, -1)
    return BFS(root)


def traverse(train, test, row_num, node):
    if node.classification:
        return node.classification
    label = test[row_num][node.feature_num]
    if label not in node.children:
        return get_avg_of_rows(train, node.rows)
    return traverse(train, test, row_num, node.children[label])


def test_tree(train, test, root):
    correct = 0
    for i in xrange(test.shape[0]):
        tree_label = traverse(train, test, i, root)
        if tree_label == test[i][test.shape[1] - 1]:
            correct += 1
    return float(correct)/float(test.shape[0])


def run_test(train, test, max_depth):
    gen_unique(train)
    root = find_root(train, range(train.shape[0]), set(), 0, max_depth)
    id3(root, train, 0, max_depth)
    return test_tree(train, test, root)


def correct_data(dataset, method_num):
    last_column = dataset.shape[1] - 1
    if method_num == 1:
        for j in xrange(dataset.shape[1]):
            c1 = Counter(dataset[:, j])
            replace_val = c1.most_common(1)[0][0]
            for i in xrange(dataset.shape[0]):
                if dataset[i][j] == '?':
                    dataset[i][j] = replace_val
    elif method_num == 2:
        row_lists = defaultdict(list)
        for i in xrange(dataset.shape[0]):
            row_lists[dataset[i][last_column]].append(i)
        for j in xrange(dataset.shape[1]):
            counters = defaultdict(Counter)
            replace_vals = {}
            for label in row_lists.keys():
                for row in row_lists[label]:
                    counters[label][dataset[row][j]] += 1
                replace_val = counters[label].most_common(1)[0][0]
                if replace_val == '?':
                    replace_val = counters[label].most_common(2)[1][0]
                replace_vals[label] = replace_val
            for i in xrange(dataset.shape[0]):
                if dataset[i][j] == '?':
                    label = dataset[i][last_column]
                    dataset[i][j] = replace_vals[label]


def cross_validate1(directory):
    files = [file1 for file1 in os.listdir(directory) if '.data' in file1]
    heights = [1, 2, 3, 4, 5, 10, 15, 20]
    accuracies = defaultdict(list)
    for i in xrange(len(files)):
        train_files = []
        test_file = ""
        for j in xrange(len(files)):
            if i == j:
                test_file = files[j]
            else:
                train_files.append(files[j])
        train_datasets = [np.genfromtxt(os.sep.join([directory, file_name]), delimiter=',', dtype='c') for file_name in train_files]
        train = np.vstack(train_datasets)
        test = np.genfromtxt(os.sep.join([directory, test_file]), delimiter=',', dtype='c')
        for height in heights:
            accuracies[height].append(run_test(train, test, height))
    result = dict.fromkeys(heights)
    for height in result:
        avg = np.mean(accuracies[height])
        sd = np.std(accuracies[height])
        result[height] = (avg, sd)
    return result


def cross_validate2(directory):
    files = [file1 for file1 in os.listdir(directory) if '.data' in file1]
    method_nums = [1, 2, 3]
    accuracies = defaultdict(list)
    for i in xrange(len(files)):
        train_files = []
        test_file = ""
        for j in xrange(len(files)):
            if i == j:
                test_file = files[j]
            else:
                train_files.append(files[j])
        train_datasets = [np.genfromtxt(os.sep.join([directory, file_name]), delimiter=',', dtype='c') for file_name in train_files]
        train = np.vstack(train_datasets)
        test = np.genfromtxt(os.sep.join([directory, test_file]), delimiter=',', dtype='c')
        for method_num in method_nums:
            corrected_train = train.copy()
            correct_data(corrected_train, method_num)
            corrected_test = test.copy()
            correct_data(corrected_test, method_num)
            accuracies[method_num].append(run_test(corrected_train, corrected_test, -1))
    result = dict.fromkeys(method_nums)
    for method_num in result:
        avg = np.mean(accuracies[method_num])
        sd = np.std(accuracies[method_num])
        result[method_num] = (avg, sd)
    return result


def main():
    start = timeit.default_timer()

    training_A = np.genfromtxt(os.sep.join(["train.csv"]), delimiter=',', dtype='c')
    test_A = np.genfromtxt(os.sep.join(['test.csv']), delimiter=',', dtype='c')
    training_error = 1 - run_test(training_A, training_A, 10)
    test_error = 1 - run_test(training_A, test_A, 10)

    stop = timeit.default_timer()

    print stop - start

    #depth = get_depth(training_A)
    #print "!!!!!!!!!!!!!!SETTING A!!!!!!!!!!!!!!!!"
    print "training error = " + str(training_error)
    print "test error = " + str(test_error)
    #print "max depth = " + str(depth)
    #cv_dir_A = os.sep.join(['datasets', 'SettingA', 'CVSplits'])
    #cv_results_A = cross_validate1(cv_dir_A)
    #for depth in cv_results_A:
        #print "depth = " + str(depth) + ", accuracy = " + str(cv_results_A[depth][0]) + ", std deviation = " + str(cv_results_A[depth][1])

    #print "Best depth was 2"
    #best_test_error_AA = run_test(training_A, test_A, 2)
    #print "best test accuracy AA= " + str(best_test_error_AA)

    #training_B = np.genfromtxt(os.sep.join(['datasets', 'SettingB', 'training.data']), delimiter=',', dtype='c')
    #test_B = np.genfromtxt(os.sep.join(['datasets', 'SettingB', 'test.data']), delimiter=',', dtype='c')
    #training_error_BB = 1 - run_test(training_B, training_B, -1)
    #test_error_BB = 1 - run_test(training_B, test_B, -1)
    #training_error_BA = 1 - run_test(training_B, training_A, -1)
    #test_error_BA = 1 - run_test(training_B, test_A, -1)
    #depth = get_depth(training_B)
    #print
    #print "!!!!!!!!!!!!!!SETTING B!!!!!!!!!!!!!!!!"
    #print "training error BB = " + str(training_error_BB)
    #print "test error BB= " + str(test_error_BB)
    #print "training error BA = " + str(training_error_BA)
    #print "test error BA= " + str(test_error_BA)
    #print "max depth = " + str(depth)
    #cv_dir_B = os.sep.join(['datasets', 'SettingB', 'CVSplits'])
    #cv_results_B = cross_validate1(cv_dir_B,)
    #for depth in cv_results_B:
        #print "depth = " + str(depth) + ", accuracy = " + str(cv_results_B[depth][0]) + ", std deviation = " + str(cv_results_B[depth][1])

    #print "Best depth was 1"
    #best_test_error_BB = run_test(training_B, test_B, 1)
    #print "best test accuracy BB= " + str(best_test_error_BB)

    #print
    #print "!!!!!!!!!!!!!!SETTING C!!!!!!!!!!!!!!!!"
    #cv_dir_C = os.sep.join(['datasets', 'SettingC', 'CVSplits'])
    #cv_results_C = cross_validate2(cv_dir_C)
    #for method_num in cv_results_C:
        #print "method_num = " + str(method_num) + ", accuracy = " + str(cv_results_C[method_num][0]) + ", std deviation = " + str(cv_results_C[method_num][1])

    #training_C = np.genfromtxt(os.sep.join(['datasets', 'SettingC', 'training.data']), delimiter=',', dtype='c')
    #test_C = np.genfromtxt(os.sep.join(['datasets', 'SettingC', 'test.data']), delimiter=',', dtype='c')
    #correct_data(training_C, 3)
    #print "Best results with special feature"
    #best_test_error_CC = run_test(training_C, test_C, -1)
    #print "best test accuracy CC= " + str(best_test_error_CC)

if __name__ == "__main__":
    main()
