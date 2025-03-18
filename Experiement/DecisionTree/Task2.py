import numpy as np


def calcInfoGain(feature, label, index):
    '''
    计算信息增益
    :param feature:测试用例中字典里的feature，类型为ndarray
    :param label:测试用例中字典里的label，类型为ndarray
    :param index:测试用例中字典里的index，即feature部分特征列的索引。该索引指的是feature中第几个特征，如index:0表示使用第一个特征来计算信息增益。
    :return:信息增益，类型float
    '''

    #*********** Begin ***********#
    def dictSum(feature: dict):
        sum = 0
        for value in feature.values():
            sum += value
        return sum

    def calculateSingleAntropy(probability):
        return -probability * np.log2(probability)

    def totalEntropy(label):
        label_dict = {}
        for l in label:
            if l in label_dict:
                label_dict[l] += 1
            else:
                label_dict[l] = 1
        labelSum = dictSum(label)
        entropySum = 0
        for value in label_dict.values():
            entropySum += calculateSingleAntropy(value / labelSum)
        return entropySum


    # def entropy(feature_dict: dict):
    #     featureSum = dictSum(feature_dict)
    #     entropySum = 0
    #     for value in feature_dict.values():
    #         entropySum += calculateSingleAntropy(value / featureSum)
    #     return entropySum


    # 统计每个特征个数
    feature_dict = {}
    for feat in feature:
        if feat in feature_dict:
            feature_dict[feat[index]] += 1
        else:
            feature_dict[feat[index]] = 1

    label_dict = {}
    for l in label:
        if l in label_dict:
            label_dict[l] += 1
        else:
            label_dict[l] = 1

    feature_label_dict = {}
    calFeature = feature[0:][index]

    # 计算总的熵
    total_entropy = totalEntropy(label)

    # 计算条件熵
    for i in range(len(calFeature)):
        if (calFeature[i], label[i]) in feature_label_dict:
            feature_label_dict[(calFeature[i], label[i])] += 1
        else:
            feature_label_dict[(calFeature[i], label[i])] = 1

    feature_dict_sum = dictSum(feature_dict)
    entropy_dict = {}
    for key in feature_dict.keys():
        entropy_sum = 0
        for value in label_dict.keys():
            if (key, value) in feature_label_dict:
                entropy_sum += calculateSingleAntropy(value / feature_dict_sum[key])
        entropy_dict[key] = entropy_sum

    condition_entropy = 0
    for key, value in feature_label_dict.items():
        condition_entropy += feature_dict[key]/feature_dict_sum * entropy_dict[key]

    return total_entropy - condition_entropy


    #*********** End *************#