import numpy as np

class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}

    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''

        #********* Begin *********#
        # 进行拉普拉斯平滑
        label_set = set(label)
        feature_num = feature.shape[1]

        # 计算每个标签的概率
        for Label in label_set:
            self.label_prob[Label] = (len(label[label == Label]) + 1) / (len(label) + len(label_set))

        for Label in label_set:
            # 初始化标签对应的字典
            self.condition_prob[Label] = {}
            for i in range(feature_num):
                specific_feature = feature[:, i]  # 取出第i个特征
                total_feature_lens = len(specific_feature)
                feature_set = set(specific_feature)
                label_count = {}
                # 先将所有标签的计数器初始化为0
                for Feature in feature_set:
                    label_count[Feature] = 0
                # 挑出当前标签对应的特征
                specific_feature_with_label = specific_feature[label[:] == Label]
                # 计算所有标签的数目
                feat_label_set = set(specific_feature_with_label)
                for Feature in feat_label_set:
                    # 计算当前标签对应的特征数目
                    label_count[Feature] = len(specific_feature_with_label[specific_feature_with_label[:] == Feature])
                # 计算当前标签对应的各个特征的概率
                for Feature in feature_set:
                    label_count[Feature] = (label_count[Feature] + 1) / (total_feature_lens + len(feature_set))
                # 将当前标签对应的概率字典存入condition_prob
                self.condition_prob[Label][i] = label_count
        #********* End *********#


    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''

        result = []
        # 对每条测试数据都进行预测
        for i, f in enumerate(feature):
            # 可能的类别的概率
            prob = np.zeros(len(self.label_prob.keys()))
            ii = 0
            for label, label_prob in self.label_prob.items():
                # 计算概率
                prob[ii] = label_prob
                for j in range(len(feature[0])):
                    prob[ii] *= self.condition_prob[label][j][f[j]]
                ii += 1
            # 取概率最大的类别作为结果
            result.append(list(self.label_prob.keys())[np.argmax(prob)])
        return np.array(result)



