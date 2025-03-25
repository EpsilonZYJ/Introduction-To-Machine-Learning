# -*- coding: utf-8 -*-

import numpy as np
import warnings
warnings.filterwarnings("ignore")

def sigmoid(x):
    '''
    sigmoid函数
    :param x: 转换前的输入
    :return: 转换后的概率
    '''
    return 1/(1+np.exp(-x))


def fit(x,y,eta=1e-3,n_iters=10000):
    '''
    训练逻辑回归模型
    :param x: 训练集特征数据，类型为ndarray
    :param y: 训练集标签，类型为ndarray
    :param eta: 学习率，类型为float
    :param n_iters: 训练轮数，类型为int
    :return: 模型参数，类型为ndarray
    '''
    #   请在此添加实现代码   #
    #********** Begin *********#
    def predict_single_y(w, x):
        return sigmoid(np.dot(w, x))

    w = np.zeros(len(x[0]) + 1)
    x_impl = np.zeros((len(x), len(x[0]) + 1))
    x_impl[:, 0] = 1
    x_impl[:, 1:] = x
    it = 0

    while it < n_iters:
        for i in range(len(x_impl)):
            y_hat = predict_single_y(w, x_impl[i])
            w = w - eta * (y_hat - y[i]) * x_impl[i]
        it += 1
    return w[1:]
    #********** End **********#
