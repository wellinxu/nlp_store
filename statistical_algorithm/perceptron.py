
import numpy as np

class Perceptron(object):
    def __init__(self, samples, labels, learning_rate):
        self.samples = samples
        self.labels = labels
        self.learning_rate = learning_rate
        self.w = np.zeros((len(samples[0]),))
        self.b = 0
        self.next_i = 0
        self.build()

    def predict(self, sample):
        y = np.dot(self.w, sample) + self.b
        if y >= 0:
            return 1
        return -1

    def build(self):
        num = 0
        while True:
            num += 1
            # 获取下一个误分类样本
            self.next_i = self.get_next_sample()
            if self.next_i == -1 or num > 10 * len(self.labels):
                # 没有误分类点，或者迭代超过一定次数结束（防止是非线性可分数据集）
                return
            sample, label = self.samples[self.next_i], self.labels[self.next_i]
            # 更新参数
            self.w += self.learning_rate * label * sample
            self.b += self.learning_rate * label
            self.next_i += 1

    def build_dual(self):
        num = 0
        a = [0] * len(self.labels)
        while True:
            num += 1
            self.next_i = self.get_next_sample()
            if self.next_i == -1 or num > 10 * len(self.labels):
                return
            sample, label = self.samples[self.next_i], self.labels[self.next_i]
            a[self.next_i] += self.learning_rate
            self.b += self.learning_rate * label
            self.w = label * a * sample
            self.next_i += 1

    def get_next_sample(self):
        """
        获取下一个误分类样本
        :return:
        """
        num = 0
        while True:
            num += 1
            if self.next_i == len(self.labels):
                self.next_i = 0    # 重头再开始找
            sample, label = self.samples[self.next_i], self.labels[self.next_i]
            y_ = np.dot(self.w, sample) + self.b
            if y_ * label <= 0:
                return self.next_i
            if num == len(self.labels):
                # 没有找到误分类样本，返回-1
                self.next_i = -1
                return self.next_i
