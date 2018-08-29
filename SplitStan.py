import math


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    """
    :param groups: 切分为2份的数据集
    :param classes: 标签的种类集合
    :return: 划分的Gini系数
    """
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def entropy(p):
    if p == 0:
        return 0
    return p * math.log(p, 2)


def info_gain(groups, classes):
    """
        :param groups: 切分为2份的数据集
        :param classes: 标签的种类集合
        :return: 划分的信息增益
    """
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted entropy for each group
    info = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score -= entropy(p)
        # weight the group score by its relative size
        info += score * (size / n_instances)
    # entropy for the whole group
    left, right = groups
    whole_group = left + right
    size = float(len(whole_group))
    fore_info = 0.0
    # avoid divide by zero
    if size != 0:
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in whole_group].count(class_val) / size
            fore_info -= entropy(p)
    return fore_info-info


def gain_ratio(groups, classes):
    """
        :param groups: 切分为2份的数据集
        :param classes: 标签的种类集合
        :return: 划分的信息增益率
    """
    gain = info_gain(groups, classes)
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    split_info = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        split_info -= entropy(size/n_instances)
    return gain/(split_info+1)
