from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from Utils import get_k_fold_data, Debug, class_balance, accuracy, save_data, load_label_data, load_feature_data, set_debug_mode
from AdaBoost import pos_neg_label, binary_label
import numpy as np
from sklearn.preprocessing import StandardScaler

@Debug
def k_fold_sklearn(k, X_train, y_train, n_estimators, isClassBalanced=True, isShuffled=False, **base_model_params):
    """
    k折交叉验证，训练模型并保存结果

    :param k: 折数
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param n_estimators: 基学习器的数量
    :param base_model_class: 基学习器的类
    :param isClassBalanced: 是否进行类别平衡
    :param isShuffled: 是否打乱数据
    :param base_model_params: 传给基学习器的参数
    :return: 总的训练集准确率和验证集准确率
    """
    train_acc_sum, valid_acc_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators)
        train_feature = data[0]
        train_label = data[1]

        # 处理类别不平衡
        if isClassBalanced:
            train_feature, train_label = class_balance(train_feature, train_label)

        if isShuffled:
            # 打乱数据
            indices = np.random.permutation(train_feature.shape[0])
            train_feature = train_feature[indices]
            train_label = train_label[indices]

        # 处理标签
        train_label = pos_neg_label(train_label)

        # 训练模型
        model.fit(train_feature, train_label)

        train_pred = model.predict(data[0])
        valid_pred = model.predict(data[2])
        # 处理标签
        train_pred = binary_label(train_pred)
        valid_pred = binary_label(valid_pred)

        train_acc_sum += accuracy(data[1], train_pred)
        valid_acc_sum += accuracy(data[3], valid_pred)
        print(f'Fold {i + 1}, Accumulated accuracy: train acc: {train_acc_sum / (i + 1):.4f}, valid acc: {valid_acc_sum / (i + 1):.4f}')
        # print(valid_pred)
        # print(data[3])
        save_data(f'./experiments/base{n_estimators}_fold{i + 1}.csv', valid_pred, data[4])
    return train_acc_sum / k, valid_acc_sum / k

def train(isClassBalanced=True, isStandard=False, **base_params):
    """
    训练模型，对指定模型进行训练并输出10折交叉验证的结果

    :param base_model_class: 基学习器类
    :param base_params: 传递给基学习器的参数
    :param isClassBalanced: 是否进行训练集的类别平衡
    :param isStandard: 是否进行数据标准化
    :return:
    """

    print("[Preparing] Loading data...")
    train_feature = load_feature_data('data.csv')
    train_label = load_label_data('targets.csv')

    # 数据标准化
    if isStandard:
        scaler = StandardScaler()
        scaler.fit(train_feature)
        train_feature = scaler.transform(train_feature)

    print("[Preparing] Data loaded successfully.")
    print("[Running] Training model...")
    print("[Running] Base model params: ", base_params)

    print("[Training] Training model, estimators: 1")
    k_fold_sklearn(k=10, X_train=train_feature, y_train=train_label, n_estimators=1, isClassBalanced=isClassBalanced, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 5")
    k_fold_sklearn(k=10, X_train=train_feature, y_train=train_label, n_estimators=5, isClassBalanced=isClassBalanced, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 10")
    k_fold_sklearn(k=10, X_train=train_feature, y_train=train_label, n_estimators=10, isClassBalanced=isClassBalanced, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 100")
    k_fold_sklearn(k=10, X_train=train_feature, y_train=train_label, n_estimators=100, isClassBalanced=isClassBalanced,**base_params)
    print("[Training] Done!")

if __name__ == '__main__':
    set_debug_mode(False)
    train(isClassBalanced=True, isStandard=False)