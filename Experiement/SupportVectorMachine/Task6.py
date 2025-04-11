#encoding=utf8
from sklearn.svm import SVC

def svm_classifier(train_data,train_label,test_data):
    '''
    input:train_data(ndarray):训练样本
          train_label(ndarray):训练标签
          test_data(ndarray):测试样本
    output:predict(ndarray):预测结果
    '''
    #********* Begin *********#
    model = SVC()
    model.fit(train_data, train_label)
    predict = model.predict(test_data)
    #********* End *********#
    return predict
