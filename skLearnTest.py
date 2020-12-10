from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

wine = datasets.load_wine()
x = wine.data
y = wine.target

print(np.shape(x), np.shape(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression().fit(x_train, y_train)
modelSvm = SVC().fit(x_train, y_train)
modelTree = DecisionTreeClassifier().fit(x_train, y_train)
modelMLPClassifier = MLPClassifier(alpha=1e-2, hidden_layer_sizes=(800,), solver='lbfgs', random_state=2).fit(x_train,
                                                                                                              y_train)


def method_name(mdl):
    print("训练数据上的准确率为:%f" % (mdl.score(x_train, y_train)))
    print("测试数据上的准确率为:%f" % (mdl.score(x_test, y_test)))
    print("-------------")


method_name(model)
method_name(modelSvm)
method_name(modelTree)
method_name(modelMLPClassifier)
