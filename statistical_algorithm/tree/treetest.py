

from test import tree
from sklearn.datasets import load_iris, load_linnerud
datas = load_iris()
xs = datas.data
ys = datas.target
# print(xs)
# print(ys)


def test_CARTofClassify():
    model = tree.CARTofClassify(xs, ys, end_fun=lambda x, y: len(y) < 2 or len(x[0]) == 0 or max(y) == min(y))
    for x, y in zip(xs, ys):
        y_p = model.predict(x)
        print(y_p, y)


def test_CARTofRegress():
    model = tree.CARTofRegress(xs, ys, end_fun=lambda x, y: len(y) < 2 or len(x[0]) == 0 or max(y) == min(y))
    for x, y in zip(xs, ys):
        y_p = model.predict(x)
        print(y_p, y)


def test_Tree():
    # model = tree.Tree(xs, ys, select_fun=tree.infoGain, e=-20)
    model = tree.Tree(xs, ys, select_fun=tree.infoGainRate, e=-20, alpha=0.012577)
    model.pruning(model.root)
    for x, y in zip(xs, ys):
        y_p = model.predict(x)
        print(y_p, y)


def test_RandomForest():
    model = tree.RandomForest(xs, ys, feature_num=4)
    for x, y in zip(xs, ys):
        y_p = model.predict(x)
        print(y_p, y)


def test_AdaBoost():
    model = tree.AdaBoost(xs, ys)
    for x, y in zip(xs, ys):
        y_p = model.predict(x)
        print(y_p, y)


def test_BoostingTree():
    model = tree.BoostingTree(xs, ys)
    for x, y in zip(xs, ys):
        y_p = model.predict(x)
        print(y_p, y)


import math
def test_GBDT():
    model = tree.GBDT(xs, ys, loss_fun=lambda x, y:0.5*math.pow(x-y, 2), gradient_fun=lambda x, y: x-y)
    for x, y in zip(xs, ys):
        y_p = model.predict(x)
        print(y_p, y)


# test_CARTofClassify()
# test_CARTofRegress()
# test_Tree()
# test_RandomForest()
# test_AdaBoost()
# test_BoostingTree()
test_GBDT()
