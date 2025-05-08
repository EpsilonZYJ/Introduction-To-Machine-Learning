import sys
from Utils import *
from AdaBoost import *
from Utils.data_processor import k_fold
from sklearn.preprocessing import StandardScaler

def train(base_model_class, feature_datapath='data.csv', label_datapath='targets.csv', isClassBalanced=True, isStandard=False, isShuffled=False, **base_params):
    """
    训练模型，对指定模型进行训练并输出10折交叉验证的结果

    :param label_datapath: 标签数据路径
    :param feature_datapath: 特征数据路径
    :param isShuffled: 是否打乱数据
    :param base_model_class: 基学习器类
    :param base_params: 传递给基学习器的参数
    :param isClassBalanced: 是否进行训练集的类别平衡
    :param isStandard: 是否进行数据标准化
    :return:
    """

    print("[Preparing] Loading data...")
    train_feature = load_feature_data(feature_datapath)
    train_label = load_label_data(label_datapath)

    # 数据标准化
    if isStandard:
        scaler = StandardScaler()
        scaler.fit(train_feature)
        train_feature = scaler.transform(train_feature)

    print("[Preparing] Data loaded successfully.")
    print("[Running] Training model...")
    print("[Running] Training model with base model class: ", base_model_class.__name__)
    print("[Running] Base model params: ", base_params)

    print("[Training] Training model, estimators: 1")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=1, base_model_class=base_model_class, isClassBalanced=isClassBalanced, isShuffled=isShuffled, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 5")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=5, base_model_class=base_model_class, isClassBalanced=isClassBalanced, isShuffled=isShuffled, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 10")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=10, base_model_class=base_model_class, isClassBalanced=isClassBalanced, isShuffled=isShuffled, **base_params)
    print("[Training] Done!")

    print("[Training] Training model, estimators: 100")
    k_fold(k=10, X_train=train_feature, y_train=train_label, n_estimators=100, base_model_class=base_model_class, isClassBalanced=isClassBalanced,isShuffled=isShuffled, **base_params)
    print("[Training] Done!")

def main(is_debug=True):
    set_debug_mode(is_debug)
    if len(sys.argv) > 1:
        isDecisionStump = sys.argv[3]
        feature_datapath = sys.argv[1]
        label_datapath = sys.argv[2]
        if isDecisionStump == '1':
            train(feature_datapath=feature_datapath, label_datapath=label_datapath, base_model_class=DecisionStumpClassifier, isClassBalanced=True, isStandard=False, isShuffled=True)
        elif isDecisionStump == '0':
            train(feature_datapath=feature_datapath, label_datapath=label_datapath, base_model_class=LogisticRegressionClassifier, isClassBalanced=True, isStandard=True, isShuffled=True, n_iterations=1000, learning_rate=1e-2)
        else:
            raise ValueError("Invalid argument. Use 1 for DecisionStump and 0 for LogisticRegression.")
    else:
        train(base_model_class=DecisionStumpClassifier, isClassBalanced=True, isStandard=False, isShuffled=True)
        # train(base_model_class=LogisticRegressionClassifier, isClassBalanced=True, isStandard=True, isShuffled=True, n_iterations=1000, learning_rate=1e-1)

if __name__ == '__main__':
    main(False)
