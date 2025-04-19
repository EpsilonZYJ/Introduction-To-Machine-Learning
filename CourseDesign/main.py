import numpy as np
from Utils import *
from AdaBoost import *
from Utils.data_processor import k_fold
from sklearn.preprocessing import StandardScaler

def main(is_debug=True):
    set_debug_mode(True)
    train_feature = load_feature_data('data.csv')
    train_label = load_label_data('targets.csv')
    print(train_feature.shape)
    print(train_label.shape)
    print(train_feature)
    print(train_label)
    # 数据标准化
    scaler = StandardScaler()
    scaler.fit(train_feature)
    train_feature = scaler.transform(train_feature)
    print(train_feature)
    print(train_label)
    # 数据随机打乱
    n_samples = train_feature.shape[0]
    indices = np.random.permutation(n_samples)
    train_feature = train_feature[indices]
    train_label = train_label[indices]

    # 逻辑回归
    # lr_model = LogisticRegression(n_iters=10000, learning_rate=1e-5)
    # lr_model.train(train_feature, train_label, sample_weights=1/np.ones(train_label.shape[0]))
    # 预测
    # y_predict = lr_model.predict(train_feature)
    # print(accuracy(train_label, y_predict))

    # 逻辑回归
    # lr_model = LogisticRegression(n_iters=1000, learning_rate=1e-1)
    # lr_model.train(train_feature, train_label, sample_weights=np.ones(train_label.shape[0])/train_label.shape[0])
    # 预测
    # y_predict = lr_model.predict(train_feature)
    # print(accuracy(train_label, y_predict))

    # 决策树桩
    # stump_model = DecisionStump()
    # stump_model.fit(train_feature, train_label)
    # 预测
    # y_predict = stump_model.predict(train_feature)
    # print(accuracy(train_label, y_predict))

    # AdaBoost
    # ada_model = AdaBoost(n_estimators=100, base_estimator=LogisticRegression(learning_rate=1e-1, n_iters=100))
    # ada_model = AdaBoost(n_estimators=10, base_estimator=DecisionStump())
    # ada_model.fit(train_feature, train_label)
    # 预测
    # y_predict = ada_model.predict(train_feature, label_decode=True)
    # print(accuracy(train_label, y_predict))

    # k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=10, base_model=LogisticRegression(n_iters=500, learning_rate=1e-1))
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=10, base_model=LogisticRegression(n_iters=100, learning_rate=1e-1))

if __name__ == '__main__':
    main(True)
    # sk()
