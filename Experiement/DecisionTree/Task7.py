#********* Begin *********#
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

train_df = pd.read_csv('./step7/train_data.csv').as_matrix()
train_label = pd.read_csv('./step7/train_label.csv').as_matrix()
test_df = pd.read_csv('./step7/test_data.csv').as_matrix()

clf = DecisionTreeClassifier(criterion='gini', max_depth=100)
clf.fit(train_df, train_label)
result = clf.predict(test_df)
result = pd.DataFrame(result)
result.to_csv('./step7/predict.csv', index=False)

#********* End *********#