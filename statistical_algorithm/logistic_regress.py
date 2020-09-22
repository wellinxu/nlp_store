"""
逻辑回归实现
"""
import random
import numpy as np


class LogisticRegress(object):
    def __init__(self, xs, ys, lr= 0.01, cycle=500):
        self.xs = xs
        self.ys = ys
        self.lr = lr    # 学习率
        self.cyclt = cycle    # 最大迭代次数
        self.params = [random.random() for i in xs[0]]
        # self.build2()
        self.build()

    def build2(self):
        # 梯度下降
        for i in range(self.cyclt):
            deltas = []
            for x, y in zip(self.xs, self.ys):
                y_hat = self.prob(x)
                delta = (y_hat - y)*x
                deltas.append(delta)
            delta = np.mean(deltas, axis=0)
            self.params = self.params - self.lr * delta

    def build(self):
        # 随机梯度下降
        for i in range(self.cyclt):
            for x, y in zip(self.xs, self.ys):
                y_hat = self.prob(x)
                delta = (y_hat - y)*x
                self.params = self.params - self.lr * delta

    def prob(self, x):
        x_w = -np.dot(x, self.params)
        p = 1/(1+np.exp(x_w))
        return p

    def predict(self, x):
        x_v = np.dot(x, self.params)
        if x_v < 0:
            return 0
        return 1


if __name__ == '__main__':
    w = [1, -2, 3]
    x = np.random.rand(1000, 3)
    x = [[1, v[1], v[2]] for v in x]
    x = np.asarray(x)
    x_w = -np.dot(x, w)
    y = 1/(1 + np.exp(x_w))
    # print(y)
    ys = [1 if i > 0.5 else 0 for i in y]
    lr = LogisticRegress(x, ys)
    print(lr.params)
    p_y = [lr.predict(i) for i in x]
    n = 0
    for a, b in zip(ys, p_y):
        if a != b:
            n += 1
    print(n)

