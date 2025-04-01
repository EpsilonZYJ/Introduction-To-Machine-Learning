from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


def news_predict(train_sample, train_label, test_sample):
    '''
    训练模型并进行预测，返回预测结果
    :param train_sample:原始训练集中的新闻文本，类型为ndarray
    :param train_label:训练集中新闻文本对应的主题标签，类型为ndarray
    :param test_sample:原始测试集中的新闻文本，类型为ndarray
    :return 预测结果，类型为ndarray
    '''

    #********* Begin *********#
    vec = CountVectorizer()
    tfidf = TfidfTransformer()

    X_train = tfidf.fit_transform(vec.fit_transform(train_sample))
    Y = train_label

    X_test = tfidf.transform(vec.transform(test_sample))

    model = MultinomialNB(alpha=1e-10)
    model.fit(X_train, Y)
    result = model.predict(X_test)
    return result

    #********* End *********#
