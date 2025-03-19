import numpy as np

def calcGini(feature, label, index):
    '''
    计算基尼系数
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:基尼系数，类型float
    '''

    #********* Begin *********#
    def GiniD(input):
        input_set = set(input)
        input_dict = {}
        for item in input_set:
            input_dict[item] = 0
        for item in input:
            input_dict[item] += 1

        item_num = len(input)
        gini = 0
        for value in input_dict.values():
            gini += -((value / item_num)**2)
        return gini+1

    feature_label_concat = [[feature[i][index], label[i]] for i in range(len(feature))]
    feature_arr = np.array([feature[i][index] for i in range(len(feature))])
    feature_set = set(feature_arr)

    feature_dict = {}
    for item in feature_arr:
        if item not in feature_dict:
            feature_dict[item] = 1
        else:
            feature_dict[item] += 1

    gini_list = []
    for feat in feature_set:
        input_label_list = []
        for i in feature_label_concat:
            if i[0] == feat:
                input_label_list.append(i[1])
        gini_list.append(GiniD(input_label_list))

    gini_sum = 0
    i = 0
    for feat in feature_set:
        gini_sum += feature_dict[feat]/len(label)*gini_list[i]
        i += 1
    return gini_sum
    #********* End *********#