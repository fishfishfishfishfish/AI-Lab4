from csv import reader
import utils
import SplitStan
import json


# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    datalist = list(lines)
    # 去掉index和id
    for row in datalist:
        row.pop(0)
        row.pop(0)
    cols_name = datalist.pop(0)
    # convert string attributes to integers
    for i in range(len(datalist[0])):
        utils.str_column_to_float(datalist, i)
    return utils.DataSet(cols_name, datalist)


DS = load_csv("all.csv")
pro_rec = utils.ProcedureRecorder(len(DS.data))
tree = utils.build_tree(DS.data, SplitStan.gain_ratio)
utils.name_cols(DS.cols_name, tree)
# 修改utils 108/118/121行

js_obj = json.dumps(tree)
file_obj = open('ShowTree//app//C45_decision_tree.json', 'w')
file_obj.write(js_obj)
file_obj.close()
