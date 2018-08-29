import numpy

# data = numpy.array([[1, 2], [3, 4]])
# data = numpy.array(data)
# # 清零第1行
# data[1] = numpy.zeros(2)
# # 清零第1列
# data[:, 1] = numpy.zeros(2)

# # 获取不重复的值
# data = numpy.array([[1, 1], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
# print(data[:, 1])
# unique = set(data[:, 1])
# print(data)
# print(unique)
# print(len(unique))
# data = list(unique)
# print(data)

# # 删除矩阵的行或列
# data = numpy.array([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]])
# data1 = numpy.delete(data, 0, axis=1)
# print(data1)
# #
# data1 = numpy.delete(data, [1, 3], axis=1)
# print(data1)

# 获取列向量顺便变成行向量表示
# data = numpy.array([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]])
# print(data[:, 1])

# # 计数重复值的个数
# data = [1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]
# data = numpy.array(data)
# dataset = set(data)
# for d in dataset:
#     print(d, ':', data.count(d))

# # 文件读取
# f = open('train.csv', 'r')
# data = []
# for line in f.readlines():
#     row = []
#     t_row = line.split(',')
#     for t in t_row:
#         row.append(int(t))
#     data.append(row)
# print(data)

# 合并list
a = [1, 2, 3, 4, 5]
b = [2, 3, 4, 5, 6]
print(a+b)