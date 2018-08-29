import numpy
import math
import random


def find_most_value(l):
    """
    :param l: 数据列表
    :return: l中出现最多次的值
    """
    s = set(l)
    l = list(l)
    most_time = 0
    most_value = 0
    for i in s:
        if l.count(i) > most_time:
            most_time = l.count(i)
            most_value = i
    return most_value


def build_tree(data, attribute_names, model_type):
    """
    :param data: 用来建树的数据集，最后一列为分类标签
    :param attribute_names: 记录这次个节点可以用来分类的属性在总的数据里为第几个
    :param model_type: 建树使用的模型: 0-ID3, 1-C4.5, 2-CART
    :return: 这一层建立的子树
    :describe: 递归建树
    """
    data = numpy.array(data)
    row, col = data.shape
    # 属性值用完了 or 所有结果都是一样的
    if col == 1 or len(set(data[:, col-1])) == 1:
        d = {'best_attri': -1, 'best_attri_name': -1}
        tree = [d]
        if col == 1:  # 属性值用完了, 找出数据集中最多的标签作为预测结果
            tree[0]['predict_res'] = find_most_value(data[:, col-1])
        else:  # 标签都一样, 把这个标签作为预测结果
            tree[0]['predict_res'] = data[0][col-1]
        return tree
    best_attribute = get_best_attribute(data, model_type)
    d = {'best_attri': best_attribute, 'best_attri_name': attribute_names[best_attribute], 'predict_res': find_most_value(data[:, col-1])}
    tree = [d]
    values_in_best_attribute = list(set(data[:, best_attribute]))
    tree[0]['attri_value_list'] = values_in_best_attribute
    for i in values_in_best_attribute:
        sub_attribute_names = attribute_names.copy()
        sub_attribute_names.pop(best_attribute)
        sub_tree = build_tree(split_data(data, best_attribute, i), sub_attribute_names, model_type)
        tree.append(sub_tree)
    return tree


def split_data(data, attribute, value):
    """
    :param data: 要分割的元数据集
    :param attribute: 分割数据集依据的属性
    :param value: 属性值为value的行都取出来构成分出来的子数据集
    :return: 分割出来的子数据集
    """
    subset = data.copy()
    subset = numpy.delete(subset, attribute, 1)  # 依据的属性列不用再出现在子数据集里了
    di = 0
    for i in range(len(data)):
        if data[i][attribute] != value:
            subset = numpy.delete(subset, di, 0)
        else:
            di += 1
    return subset


def get_best_attribute(data, type):
    """
    :param data: 要获取最佳属性的数据集
    :param type: 使用的模型: 0-ID3, 1-C4.5, 2-CART
    :return: 最佳属性在data的列号
    """
    row, col = data.shape
    models = [cal_id3, cal_c45, cal_cart]

    best_attribute = 0
    best_attribute_value = models[type](data[:, 0], data[:, col-1])
    for i in range(1, col-1):  # 最后一列是分类结果
        temp_value = models[type](data[:, i], data[:, col-1])  # 选择使用的模型id3, c45, cart
        if temp_value > best_attribute_value:
            best_attribute = i
            best_attribute_value = temp_value
    return best_attribute


# 计算一个属性的ID3值
def cal_id3(a_col, r_col):
    """
    :param a_col: 属性列
    :param r_col: 结果标签列
    :return: 该属性列的ID3值
    """
    a_col = list(a_col.copy())
    r_col = list(r_col.copy())
    origin_entropy = entropy(r_col)
    col_size = len(r_col)
    labels = set(a_col)
    res = 0
    for label in labels:
        prop = a_col.count(label)/col_size
        sub_col = []
        for i in range(col_size):
            if a_col[i] == label:
                sub_col.append(r_col[i])
        condition_entropy = entropy(sub_col)
        res += prop*condition_entropy
    # print('origin:', origin_entropy, 'condition:', res)
    return origin_entropy - res


# 计算一个属性的C4.5值
def cal_c45(a_col, r_col):
    """
    :param a_col: 属性列
    :param r_col: 结果标签列
    :return: 该属性的C4.5值
    """
    a_col = list(a_col.copy())
    r_col = list(r_col.copy())
    origin_entropy = entropy(r_col)
    col_size = len(r_col)
    labels = set(a_col)
    res = 0
    for label in labels:
        prop = a_col.count(label)/col_size
        sub_col = []
        for i in range(col_size):
            if a_col[i] == label:
                sub_col.append(r_col[i])
        condition_entropy = entropy(sub_col)
        res += prop*condition_entropy
    splitinfo = entropy(a_col) + 0.00001  # 属性值都一样，熵为0
    # print('origin:', origin_entropy, 'condition:', res, 'splitInfo', splitinfo)
    return (origin_entropy - res) / splitinfo


# 计算一个列数据的熵
def entropy(tag_col):
    """
    :param tag_col: 数据列，可以是属性值也可以是结果标签
    :return: 这列数据的熵
    """
    col_size = len(tag_col)
    value_type = set(tag_col)
    res = 0
    for t in value_type:
        cnt = tag_col.count(t)
        res -= (cnt/col_size)*math.log(cnt/col_size, 2)
    return res


# 计算一个属性的cart值
def cal_cart(a_col, r_col):
    """
    :param a_col: 属性列
    :param r_col: 结果标签列
    :return: 该属性的CART值
    """
    a_col = list(a_col.copy())
    r_col = list(r_col.copy())
    col_size = len(r_col)
    labels = set(a_col)
    res = 0
    for label in labels:
        prop = a_col.count(label) / col_size
        sub_col = []
        for i in range(col_size):
            if a_col[i] == label:
                sub_col.append(r_col[i])
        condition_gini = gini(sub_col)
        res += prop * condition_gini
    # print('gini:', res)
    return -res  # gini系数选择小的值为优


# 计算一个属性列的gini系数
def gini(tag_col):
    """
    :param tag_col: 一列结果标签
    :return: 计算出的gini系数
    """
    col_size = len(tag_col)
    value_type = set(tag_col)
    res = 0
    for t in value_type:
        cnt = tag_col.count(t)
        res += (cnt / col_size) * (cnt / col_size)
    return 1 - res


def travel_tree(tree, blank_space):
    """
    :param tree: 要遍历的树，以列表形式
    :param blank_space: 输出时前面的空格
    :return: none
    """
    print(blank_space, tree[0])
    if tree[0]['best_attri'] == -1:
        return
    blank_space += '\t'
    for i in range(1, len(tree)):
        travel_tree(tree[i], blank_space)


def make_decision(tree, item):
    """
    :param tree: 用于决策的决策树
    :param item: 需要判断的一行数据
    :return: 决策的结果
    """
    if tree[0]['best_attri'] == -1:
        return tree[0]['predict_res']
    for i in range(len(tree[0]['attri_value_list'])):
        if item[tree[0]['best_attri']] == tree[0]['attri_value_list'][i]:
            sub_item = item.copy()
            sub_item.pop(tree[0]['best_attri'])
            res = make_decision(tree[i+1], sub_item)  # 递归决策
            return res
    return tree[0]['predict_res']  # 没有对应的叶节点


def train(data, model_type):
    """
    :param data: 训练数据集
    :param model_type: 使用的模型: 0-ID3, 1-C4.5, 2-CART
    :return: 返回训练完成得到的决策树
    """
    # 开始训练
    attribute_names = list(range(len(data[0]) - 1))
    data = numpy.array(data)
    tree = build_tree(data, attribute_names, model_type)
    return tree


def valid(tree, data):
    """
    :param tree: 用来决策的决策树
    :param data: 验证集数据
    :return: 验证集判决的accuracy, precision, recall, f1
    """
    # 开始验证
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    size = len(data)
    for line in data:
        res = make_decision(tree, line[0:len(line) - 1])
        # print(res, line[len(line) - 1])
        if res > 0 and line[len(line) - 1] > 0:
            TP += 1
        elif res < 0 and line[len(line) - 1] < 0:
            TN += 1
        elif res > 0 and line[len(line) - 1] < 0:
            FP += 1
        else:
            FN += 1
    a = (TP+TN) / (TP + TN + FP + FN)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f1 = (2 * p * r) / (p + r)
    return a, p, r, f1


# 连续数据的离散化
def data_discrete(data, col, split_number):
    """
    :param data: 数据集
    :param col: 需要离散化的列号
    :return: none
    """
    min_value = min(data[:, col])
    max_value = max(data[:, col])
    # split_number = 4  # 连续值区间被切割的份数
    delta = (max_value - min_value) / split_number  # 连续值的判断区间
    for i in range(len(data[:, col])):
        data[i, col] = (data[i, col]-min_value) // delta


# print('小数据集')
# print('请输入使用模型：0-ID3, 1-C4.5, 2-CART')
# model_type = input()
# model_type = int(model_type)
# f = open('small-train.csv', 'r')
# data = []
# data_list = []
# for line in f.readlines():
#     row = []
#     t_row = line.split(',')
#     for t in t_row:
#         row.append(int(t))
#     data.append(row)
# data = numpy.array(data)
# print(data)
# data_discrete(data, 2)
# print(data)
# data = data.tolist()
# tree = train(data, model_type)
# travel_tree(tree, '')

print('使用S折交叉验证，训练集请命名为‘train.csv’')
print('请输入使用模型：0-ID3, 1-C4.5, 2-CART')
f = open('train.csv', 'r')
data = []
data_list = []
# 读取数据
for line in f.readlines():
    row = []
    t_row = line.split(',')
    for t in t_row:
        row.append(int(t))
    data.append(row)
split_number = 3
data = numpy.array(data)
data_discrete(data, 0, split_number)
data = data.tolist()
data_for_test = data.copy()
Sfold = 4  # S折的折数
# 把训练集数据分成S份
val_size = len(data)//Sfold
for k in range(Sfold-1):
    t_data = []
    for i in range(val_size):
        t_data.append(data.pop(random.randint(0, len(data)-1)))
    data_list.append(t_data)
data_list.append(data)
# S折交叉验证
for model_type in range(3):  # 三种模型进行对比
    print('使用模型:', model_type)
    val_a = 0
    val_p = 0
    val_r = 0
    val_f1 = 0
    # S份中取出一份作为验证集，其余构成训练集
    for k in range(Sfold):
        train_data = []
        for i in range(Sfold):
            if i != k:
                train_data += data_list[i]
        val_data = data_list[k]  # 剩下的一份作为验证集
        tree = train(train_data, model_type)  # 训练得到决策树
        val_res = valid(tree, val_data)  # 获取评判标准
        val_a += val_res[0]
        val_p += val_res[1]
        val_r += val_res[2]
        val_f1 += val_res[3]
    print('accuracy:', val_a/Sfold, 'precision:', val_p/Sfold, 'recall:', val_r/Sfold, 'f1:', val_f1/Sfold)  # 输出平均值
# 分类测试集
print('请选择模型：')
test_model_type = input()
test_model_type = int(test_model_type)
tree_for_test = train(data_for_test, test_model_type)  # 训练得到决策树
tfile = open('test.csv', 'r')
rfile = open('15352049_chenxinyu.txt', 'w')
for line in tfile.readlines():
    row = []
    t_row = line.split(',')
    t_row.pop(len(t_row)-1)
    for t in t_row:
        row.append(int(t))
    rfile.write(str(make_decision(tree_for_test, row)) + '\n')
