import random
from random import seed
from random import randrange
import numpy


def singleton(cls):
    _instance = {}

    def inner(whole=0):
        if cls not in _instance:
            _instance[cls] = cls(whole)
        return _instance[cls]
    return inner


class DataSet:
    def __init__(self, cols_name: list, data: list):
        self.cols_name = cols_name
        self.data = data
    
    def suffle_col(self, col):
        arr = numpy.array(self.data.copy())
        sample_col = arr[:, col]
        random.shuffle(sample_col)
        return arr.tolist()        


@singleton
class ProcedureRecorder:
    def __init__(self, whole=0):
        print("start")
        self.counter = 0
        self.whole = whole

    def tell(self):
        print("{:.3f}%".format(self.counter / self.whole * 100))

    def count(self, progress):
        self.counter += progress

    def reset(self, whole):
        self.whole = whole
        self.counter = 0
        print("reset ", self.whole)


# Convert string column to float
def str_column_to_float(dataset: list, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Split a dataset into k folds
def cross_validation_split(dataset: list, n_folds: int):
    """
    :param dataset:
    :param n_folds:
    :return: 被切割成n_folds分的数据集
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds) + 1
    for i in range(n_folds-1):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    dataset_split.append(dataset_copy)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual: list, predicted: list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            if actual[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if actual[i] == 1:
                fp += 1
            else:
                fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# Split a dataset based on an attribute and an attribute value
def test_split(index: int, value: int, dataset: list):
    """
    :param index: 切分依据的列
    :param value: 切分依据的值
    :param dataset:
    :return: 切分后的两个数据集
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def l_score(d: dict):
    return d['score']


# Select the best split point for a dataset
def get_split(dataset, split_func, record_choice):
    # 标签的种类集合
    class_values = list(set(row[-1] for row in dataset))
    # 记录划分的属性，划分用的属性值，划分后的gini系数，划分后的样本集合
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    choice = list()
    # 遍历所有属性
    for index in range(len(dataset[0]) - 1):
        # 遍历属性所有取值
        for value in list(set(row[index] for row in dataset)):
            groups = test_split(index, value, dataset)
            score = split_func(groups, class_values)
            if record_choice:
                choice.append({"index": index, "value": value, "score": score})
            if score < b_score:  # gini系数选最小的，信息增益和增益率选最大的。
                b_index, b_value, b_score, b_groups = index, value, score, groups
    if record_choice:
        choice = sorted(choice, key=l_score)
        # choice = choice[::len(choice)//100]
    return {'index': b_index, 'value': b_value, 'choice': choice, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    pro_rec = ProcedureRecorder()
    pro_rec.count(len(group))
    pro_rec.tell()
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, split_func):
    left, right = node['groups']
    del (node['groups'])

    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if 0 < max_depth <= depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    left_class_values = list(set(row[-1] for row in left))
    if len(left) <= min_size or len(left_class_values) == 1:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, split_func, True)
        split(node['left'], max_depth, min_size, depth + 1, split_func)
    # process right child
    right_class_values = list(set(row[-1] for row in right))
    if len(right) <= min_size or len(right_class_values) == 1:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, split_func, True)
        split(node['right'], max_depth, min_size, depth + 1, split_func)


# Build a decision tree
def build_tree(train, split_func, max_depth=-1, min_size=1):
    pro_rec = ProcedureRecorder()
    pro_rec.reset(len(train))
    root = get_split(train, split_func, True)
    split(root, max_depth, min_size, 1, split_func)
    return root


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    pro_rec = ProcedureRecorder(len(dataset))
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, tree = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]  # 测试集的正确标签
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, test, split_func, max_depth=-1, min_size=1):
    tree = build_tree(train, split_func, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions, tree


# name columns
def name_cols(cols_name: list, tree):
    if not isinstance(tree, dict):
        return
    tree['index'] = cols_name[tree['index']]
    for c in tree['choice']:
        c['index'] = cols_name[c['index']]
    name_cols(cols_name, tree['right'])
    name_cols(cols_name, tree['left'])
    return
