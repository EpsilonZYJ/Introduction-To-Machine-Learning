import numpy as np

class DecisionStump:
    def __init__(self):
        """
        初始化决策树桩

        feature_index: 最优特征索引
        threshold:  分割阈值
        left_value: 左子树的值
        right_value: 右子树的值
        """
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def _entropy(self, y):
        """
        计算信息熵

        :param y: 数据集标签
        :return: 对应的信息熵
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        label_set = set(y)  # 标签集合
        total = len(y)  # 数据集大小

        # 计算每个标签的出现次数
        cnt = []
        for label in label_set:
            cnt.append(len(y[y == label]))

        # 计算每个标签的概率
        cnt = np.array(cnt)
        prob = cnt / total
        # 计算信息熵
        entropy = -np.sum(prob * np.log2(prob + 1e-12)) # 加上一个小的常数避免log(0)
        return entropy

    def _split_entropy(self, feature, y, threshold):
        """
        计算划分后的加权信息熵

        :param feature: 数据集某一特征
        :param y: 数据集标签
        :param threshold: 划分阈值
        :return: 当前划分下的加权信息熵
        """
        if not isinstance(feature, np.ndarray):
            X = np.array(feature)
        if not isinstance(feature, np.ndarray):
            y = np.array(feature)

        # 获得按照阈值划分后的样本
        left_indices = feature <= threshold
        right_indices = feature > threshold
        left_label = y[left_indices]
        right_label = y[right_indices]

        # 计算分割后的信息熵
        left_entropy = self._entropy(left_label)
        right_entropy = self._entropy(right_label)

        # 权重计算
        total = len(y)
        left_weight = len(left_label) / total
        right_weight = len(right_label) / total

        return left_weight * left_entropy + right_weight * right_entropy

    def _get_candidate_thresholds(self, feature):
        """
        获取划分阈值

        :param feature: 所有数据集在此特征上的值
        :return: 得到所有可能的划分阈值
        """
        # 先得到唯一特征值并排序
        feature = list(set(feature))
        feature = np.array(feature)
        feature.sort()
        if len(feature) < 2:
            return []
        # 计算所有可能的划分阈值
        thresholds = (feature[: -1] + feature[1: ]) / 2
        return thresholds

    def _classification(self, y):
        """
        获取当前y的最好的分类值，按照最大出现次数来分类

        :param y: 当前输入数据的标签
        :return: 根据当前y的每种标签的个数来确定最好的分类值
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.shape[0] == 0:
            raise ValueError("训练进行分类时出现异常分类样本数目")
        label_set = set(y)
        best_label = None
        best_count = -1
        for label in label_set:
            cur_count = len(y[y == label])
            if cur_count > best_count:
                best_label = label
                best_count = cur_count
        return best_label

    def fit(self, X, y):
        """
        训练决策树桩

        :param X: 训练集的输入特征
        :param y: 训练集的标签
        :return:
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.ndarray(y)

        best_gain = -np.inf

        # 遍历所有的特征
        for feature_index in range(X.shape[1]):
            feature = X[:, feature_index]
            # 获取所有的可能分割的阈值
            thresholds = self._get_candidate_thresholds(feature)

            # 对所有可能的阈值进行分割，并且计算信息增益
            for threshold in thresholds:
                info_gain = self._entropy(y) - self._split_entropy(feature, y, threshold)
                if info_gain > best_gain:
                    best_gain = info_gain
                    self.feature_index = feature_index
                    self.threshold = threshold

        # 确认左右子树的值
        left_label = y[X[:, self.feature_index] <= self.threshold]
        right_label = y[X[:, self.feature_index] > self.threshold]
        self.left_value = self._classification(left_label)
        self.right_value = self._classification(right_label)

        # if self.left_value == self.right_value:
        #     raise ValueError("当前数据集无法进行划分")

    def train(self, X, y, sample_weights):
        """
        适用于AdaBoost的训练函数

        :param X: 训练集特征
        :param y: 训练集标签
        :param sample_weights: 样本权重
        :return:
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(sample_weights, np.ndarray):
            sample_weights = np.array(sample_weights)

        best_err = 1.0
        # 遍历所有的特征
        for feature_index in range(X.shape[1]):
            feature = X[:, feature_index]
            # 获取所有的可能分割的阈值
            thresholds = self._get_candidate_thresholds(feature)

            # 对所有可能的阈值进行分割，并且计算加权错误率
            for threshold in thresholds:
                # 计算加权错误率
                prediction = np.ones(X.shape[0])
                prediction[feature <= threshold] = 0
                err = np.sum(sample_weights * (prediction != y))

                # 按照更低错误率进行更新
                if min(err, 1-err) < best_err:
                    best_err = err
                    self.feature_index = feature_index
                    self.threshold = threshold
        self.left_value = self._classification(y[X[:, self.feature_index] <= self.threshold])
        self.right_value = self._classification(y[X[:, self.feature_index] > self.threshold])
        # if self.left_value == self.right_value:
        #     raise ValueError("当前数据集无法进行划分")

    def predict(self, X):
        """
        预测函数

        :param X: 测试数据集
        :return: 预测值
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if X[i, self.feature_index] <= self.threshold:
                y_pred[i] = self.left_value
            else:
                y_pred[i] = self.right_value
        return y_pred
