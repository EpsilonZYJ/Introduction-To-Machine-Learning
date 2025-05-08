# 机器学习导论结课大作业
## 运行环境
- 依赖包安装可使用以下命令
```shell
pip install -r requirementx.txt
```
- 如果装不上，请保证python版本<=3.9，numpy版本<=1.20，其余版本可以兼容
- 如果还是运行不了，可以使用最新版本的依赖包，但是请在evaluate.py中作如下修改
```python
import numpy as np
target = np.genfromtxt('targets.csv')
base_list = [1, 5, 10, 100]

for base_num in base_list:
    acc = []
    for i in range(1, 11):
        # fold = np.genfromtxt('experiments/base%d_fold%d.csv' % (base_num, i), delimiter=',', dtype=np.int)
        fold = np.genfromtxt('experiments/base%d_fold%d.csv' % (base_num, i), delimiter=',', dtype=int)
        accuracy = sum(target[fold[:, 0] - 1] == fold[:, 1]) / fold.shape[0]
        acc.append(accuracy)
    print(np.array(acc).mean())
```
  否则evaluate.py会报错（最新版本的numpy不支持dtype=np.int）

## 数据集有关说明

**注意：在本代码中存在类别平衡的假设，对于轻微类别不平衡的数据集，代码会自动纠正进行类别平衡。而对于严重类别不平衡的数据，比例大于100:1的数据集，代码中假设了这种情况下天然存在不平衡（如罕见病例预测等），会保留类别不平衡的信息，因此不会进行平衡。如果数据天然存在平衡的情况，但是数据集中表现出严重类别不平衡，需要手动在Utils/data_processor.py中进行相应代码的注释。代码如下所示：**

```python
@Debug
def class_balance(X, y):
    """
    处理训练集类别不平衡的问题
    :param X: 训练集特征
    :param y: 训练集标签
    :return:
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    feature0 = X[y == 0]
    feature1 = X[y == 1]
    label0 = y[y == 0]
    label1 = y[y == 1]
    count0 = len(label0)
    count1 = len(label1)
    cnt = min(count0, count1)

    # # 如果类别不平衡的比例过大，则不进行处理
    # if count0 * 1e2 <count1 or count1 * 1e2 < count0:
    #     return X, y

    # 随机选择count个样本
    indices = np.random.choice(len(label0), cnt, replace=False)
    feature0 = feature0[indices]
    label0 = label0[indices]

    indices = np.random.choice(len(label1), cnt, replace=False)
    feature1 = feature1[indices]
    label1 = label1[indices]

    # 合并数据
    X = np.concatenate((feature0, feature1), axis=0)
    y = np.concatenate((label0, label1), axis=0)
    # 打乱数据
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    return X, y
```

## 文件路径

```
CourseDesign/
├── AdaBoost/                               # AdaBoost算法实现，包含基分类器
│   ├── __init__.py
│   ├── BaseModel.py                        # 包含：模型抽象基类，所有基分类器和AdaBoost类都直接或间接继承自该类;
│   │                                       #      弱学习器抽象类，继承自模型抽象基类，为所有基学习器的父类
│   ├── DecisionStumpClassifier.py          # 决策树桩基分类器
│   ├── LogisticRegressionClassifier.py     # 逻辑回归基分类器
│   └── AdaBoostClassifier.py               # AdaBoost分类器
├── Utils/                                  # 训练或评估等所需的工具函数
│   ├── __init__.py
│   ├── Accuracy.py                         # 用于计算预测正确率
│   ├── data_processor.py                   # 用于读写数据、数据预处理、数据划分、训练等与训练和数据清洗等相关的函数
│   └── debug_wrapper.py                    # 用于调试和定位错误的装饰器
├── experiments/
│   └── base#_fold#.csv                     # 训练结果，基分类器为决策树桩时，#表示基分类器数目，fold表示第几折
├── evaluate.py                             # 作业提供的评估程序
├── main.py                                 # 训练模型的函数入口
├── data.csv                                # 数据集特征集
├── targets.csv                             # 数据集标签集
├── requirements.txt                        # 依赖包列表
└── README.md                               # 说明文档
```
**注意：数据集一定要放在与main.py同一目录下，否则无法正确读取数据集。**

## 命令行运行训练程序
- 使用以下命令，基分类器为决策树桩
    ```shell
    python main.py /path/to/data/data.csv  /path/to/data/target.csv 1
    ```
  或不指定，默认为决策树桩，且从默认路径读取数据集
    ```shell
    python main.py
    ```
- 使用以下命令，基分类器为逻辑回归
    ```shell
    python main.py /path/to/data/data.csv  /path/to/data/target.csv 0
    ```
- 使用以下命令指定文件路径
    ```shell
    python main.py /path/to/data/data.csv  /path/to/data/target.csv  0
    ```
  
## 命令行运行评估程序
```shell
python evaluate.py
```