from Utils import *
from AdaBoost import *

def main(is_debug=True):
    set_debug_mode(True)
    train_feature = load_feature_data('data.csv')
    train_label = load_label_data('targets.csv')
    print(train_feature.shape)
    print(train_label.shape)
    print(train_feature)
    print(train_label)
    # 逻辑回归
    lr_model = LogisticRegression(n_iters=10000, learning_rate=1e-8)
    lr_model.fit(train_feature, train_label)
    # 预测
    y_predict = lr_model.predict(train_feature)
    print(accuracy(train_label, y_predict))

if __name__ == '__main__':
    main(True)
