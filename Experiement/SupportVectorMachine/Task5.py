# encoding=utf8
import numpy as np


class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        '''
        input:max_iter(int):最大训练轮数
              kernel(str):核函数，等于'linear'表示线性，等于'poly'表示多项式
        '''
        self.max_iter = max_iter
        self._kernel = kernel

    # 初始化模型
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        # 将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 1.0

    # ********* Begin *********#
    # kkt条件
    def _kkt(self, i):
        if self.alpha[i] == 0:
            return self._g(i) * self.Y[i] >= 1
        elif 0 < self.alpha[i] < self.C:
            return self._g(i) * self.Y[i] == 1
        else:
            return self._g(i) * self.Y[i] <= 1
    # g(x)预测值，输入xi（X[i]）
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r
    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1)**2
        else:
            return 0
    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]
    # 初始alpha
    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        not_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(not_satisfy_list)
        for i in index_list:
            if self._kkt(i):
                continue
            E1 = self.E[i]
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j
    # 选择参数
    def _select_alpha(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha
    # 训练
    def fit(self, features, labels):
        self.init_args(features, labels)
        for _ in range(self.max_iter):
            index1, index2 = self._init_alpha()
            if self.Y[index1] == self.Y[index2]:
                L = max(0, self.alpha[index1] + self.alpha[index2] - self.C)
                H = min(self.C, self.alpha[index1] + self.alpha[index2])
            else:
                L = max(0, self.alpha[index2] - self.alpha[index1])
                H = min(self.C, self.C + self.alpha[index2] - self.alpha[index1])

            E1 = self.E[index1]
            E2 = self.E[index2]
            eta = self.kernel(self.X[index1], self.X[index1]) + \
                  self.kernel(self.X[index2], self.X[index2]) - \
                  2 * self.kernel(self.X[index1], self.X[index2])
            if eta <= 0:
                continue
            alpha2_new = self.alpha[index2] + self.Y[index2] * (E2 - E1) / eta
            alpha2_new = self._select_alpha(alpha2_new, L, H)
            alpha1_new = self.alpha[index1] + self.Y[index1] * self.Y[index2] * (self.alpha[index2] - alpha2_new)

            b1_new = -E1 - self.Y[index1] * self.kernel(self.X[index1], self.X[index1]) * (alpha1_new - self.alpha[index1]) - \
                self.Y[index2] * self.kernel(self.X[index2], self.X[index1]) * (alpha2_new - self.alpha[index2]) + self.b
            b2_new = -E2 - self.Y[index1] * self.kernel(self.X[index1], self.X[index2]) * (alpha1_new - self.alpha[index1]) - \
                self.Y[index2] * self.kernel(self.X[index2], self.X[index2]) * (alpha2_new - self.alpha[index2]) + self.b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2
            self.alpha[index1] = alpha1_new
            self.alpha[index2] = alpha2_new
            self.b = b_new
            self.E[index1] = self._E(index1)
            self.E[index2] = self._E(index2)
    # ********* End *********#
    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w
