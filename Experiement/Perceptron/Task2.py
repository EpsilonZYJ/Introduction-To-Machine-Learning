#encoding=utf8
import os


if os.path.exists('./step2/result.csv'):
    os.remove('./step2/result.csv')

#********* Begin *********#
from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np

#获取训练数据
train_data = pd.read_csv('./step2/train_data.csv')
#获取训练标签
train_label = pd.read_csv('./step2/train_label.csv')
train_label = train_label['target']
#获取测试数据
test_data = pd.read_csv('./step2/test_data.csv')

X_train = np.array(train_data)
Y_train = np.array(train_label)
X_test = np.array(test_data)


clf = Perceptron(eta0=0.1, max_iter=200)
clf.fit(X_train, Y_train)
result = clf.predict(X_test)

result = pd.DataFrame({'result': result})
result.to_csv('./step2/result.csv', index=False)
#********* End *********#
