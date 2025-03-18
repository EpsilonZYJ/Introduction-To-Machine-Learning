# encoding=utf8
import numpy as np


# 构建感知机算法
class Perceptron(object):
    def __init__(self, learning_rate=0.01, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, data, label):
        '''
        input:data(ndarray):训练数据特征
              label(ndarray):训练数据标签
        output:w(ndarray):训练好的权重
               b(ndarry):训练好的偏置
        '''
        # 编写感知机训练方法，w为权重，b为偏置
        self.w = np.array([1.] * data.shape[1])
        self.b = np.array([1.])
        # ********* Begin *********#
        for i in range(self.w.shape[0]):
            self.w[i] = 0
        t = 0
        while t < self.max_iter:
            for i in range(data.shape[0]):
                if label[i]*(np.dot(self.w, data[i]) + self.b) < 0:
                    self.w = self.w + self.lr*label[i]*data[i]
                    t = t + 1
        # ********* End *********#

    def predict(self, data):
        '''
        input:data(ndarray):测试数据特征
        output:predict(ndarray):预测标签
        '''
        # ********* Begin *********#
        predict = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            if np.dot(self.w, data[i]) < 0:
                predict[i] = -1
            else:
                predict[i] = 1
        # ********* End *********#
        return predict