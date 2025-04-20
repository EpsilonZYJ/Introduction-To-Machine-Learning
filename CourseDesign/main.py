import sys
from Utils import *
from AdaBoost import *
from Utils.data_processor import k_fold, class_balance
from sklearn.preprocessing import StandardScaler

def train(base_model_class, isClassBalanced=True, isStandard=False, **base_params):
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
        train_feature, train_label = class_balance(train_feature, train_label)

    print("[Preparing] Data loaded successfully.")
    print("[Running] Training model...")
    print("[Running] Training model with base model class: ", base_model_class.__name__)
    print("[Running] Base model params: ", base_params)

    print("[Training] Training model, estimators: 1")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=1, base_model_class=base_model_class, isClassBalanced=isClassBalanced, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 5")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=5, base_model_class=base_model_class, isClassBalanced=isClassBalanced, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 10")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=10, base_model_class=base_model_class, isClassBalanced=isClassBalanced, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 100")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=100, base_model_class=base_model_class, isClassBalanced=isClassBalanced,**base_params)
    print("[Training] Done!")


def main(is_debug=True):
    set_debug_mode(False)
    if len(sys.argv) > 1:
        isDecisionStump = sys.argv[1]
        if isDecisionStump == '1':
            train(DecisionStumpClassifier)
        elif isDecisionStump == '0':
            train(LogisticRegressionClassifier, n_iters=1000, learning_rate=1e-1)
        else:
            raise ValueError("Invalid argument. Use 1 for DecisionStump and 0 for LogisticRegression.")
    else:
        train(base_model_class=DecisionStumpClassifier, isClassBalanced=False, isStandard=True)

    # 数据标准化

    # 数据随机打乱
    # n_samples = train_feature.shape[0]
    # indices = np.random.permutation(n_samples)
    # train_feature = train_feature[indices]
    # train_label = train_label[indices]

    # 逻辑回归
    # lr_model = LogisticRegression(n_iters=100, learning_rate=1e-1)
    # lr_model.fit(train_feature, train_label, sample_weight=1/np.ones(train_label.shape[0]))
    # 预测
    # y_predict = lr_model.predict(train_feature)
    # print(accuracy(train_label, y_predict))
    # print(len(train_label[train_label == 1]))
    # print(len(train_label[train_label == 0]))

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
    # ada_model = AdaBoostClassifier(n_estimators=10, estimator=LogisticRegression(learning_rate=1e-1, n_iters=100))
    # ada_model = AdaBoostClassifier(n_estimators=10, estimator=DecisionStump())
    # ada_model = AdaBoostClassifier(learning_rate=1.0, n_estimators=10, estimator=LogisticRegression(learning_rate=1e-1, n_iters=100))
    # ada_model = AdaBoostClassifier(n_estimators=10, estimator=DecisionTreeClassifier(max_depth=1))
    # ada_model = AdaBoostClassifier(n_estimators=10, estimator=LR())
    # ada_model.fit(train_feature, train_label)
    # print(len(ada_model.estimators_))
    # 预测
    # y_predict = ada_model.predict(train_feature, is_boolean_label=True)
    # print(accuracy(train_label, y_predict))

    # ada_model = ABC(n_estimators=10, estimator=DecisionTreeClassifier(max_depth=1))
    # ada_model = AdaBoostClassifier(n_estimators=10, estimator=LR())
    # ada_model.fit(train_feature, train_label)
    # print(len(ada_model.estimators_))
    # 预测
    # y_predict = ada_model.predict(train_feature)
    # print(accuracy(train_label, y_predict))


if __name__ == '__main__':
    main(True)
    # sk()
