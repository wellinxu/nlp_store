
import numpy as np


class SVM(object):
    def __init__(self, samples, labels, c, e, kernel=lambda x, y: np.dot(x, y), cycle_num=500):
        self.samples = samples    # 样本特征
        self.labels = labels    # 样本标签
        self.c = c    # 惩罚系数
        self.e = e    # 精度
        self.a = [0] * len(self.labels)   # 对偶形式的系数
        self.e_lilst = [0] * len(self.labels)    # 存储每次迭代的能量
        self.b = 0
        self.kernel = kernel    # 核函数
        self.cycle_num = cycle_num    # 最多迭代次数
        self.build()

    def predict(self, sample):
        # 预测
        kernel_mul = [self.kernel(x, sample)*label for x, label in zip(self.samples, self.labels)]
        y = np.dot(np.array(self.a), np.array(kernel_mul)) + self.b
        return y

    def build(self):
        # 训练
        num = 0
        while True:
            num += 1
            index_1, index_2 = self.select_a()
            self.solve(index_1, index_2)
            if num > self.cycle_num or self.obey_kkt():
                # 达到kkt条件或者迭代一定次数，结束训练
                return

    def selfct_a(self):
        # 选择两个参数
        index_1 = -1
        max_gap = self.e
        for i in range(len(self.labels)):
            sample, label, a_i = self.samples[i], self.labels[i], self.a[i]
            if 0 < a_i < self.c:
                tem_gap = abs(label*self.predict(sample) - 1)
                if tem_gap > max_gap:
                    max_gap = tem_gap
                    index_1 = i
        if index_1 == -1:
            for i in range(len(self.labels)):
                sample, label, a_i = self.samples[i], self.labels[i], self.a[i]
                if 0 == a_i:
                    tem_gap = 1 - label * self.predict(sample)
                elif a_i == self.c:
                    tem_gap = label * self.predict(sample) - 1
                else:
                    tem_gap = 0
                if tem_gap > max_gap:
                    max_gap = tem_gap
                    index_1 = i
        self.e_list = [self.predict(sample) - label for sample, label in zip(self.samples, self.labels)]
        e_1 = self.e_list[index_1]
        if e_1 > 0:
            index_2 = np.argmin(self.e_list)
        else:
            index_2 = np.argmax(self.e_list)
        return index_1, index_2

    def solve(self, index_1, index_2):
        # 求解二次规划
        y_1, y_2 = self.labels[index_1], self.labels[index_2]
        a_1, a_2 = self.a[index_1], self.a[index_2]
        x_1, x_2 = self.samples[index_1], self.samples[index_2]
        e_1, e_2 = self.e_list[index_1], self.e_list[index_2]
        if y_1 == y_2:
            L = max(0, a_1 + a_2 - self.c)
            H = min(self.c, a_1 + a_2)
        else:
            L = max(0, a_2 - a_1)
            H = min(self.c, self.c + a_2 - a_1)
        n = self.kernel(x_1, x_1) + self.kernel(x_2, x_2) - 2 * self.kernel(x_1, x_2)
        new_a_2 = a_2 + (y_2 * (e_1 - e_2))/n
        if new_a_2 < L:
            new_a_2 = L
        elif new_a_2 > H:
            new_a_2 = H
        new_a_1 = a_1 + y_1 * y_2 * (a_2 - new_a_2)
        new_b_1 = -e_1 - y_1 * self.kernel(x_1, x_1)*(new_a_1-a_2) - y_2*self.kernel(x_2, x_1)*(new_a_2-a_1) + self.b
        new_b_2 = -e_2 - y_1 * self.kernel(x_1, x_2)*(new_a_1-a_2) - y_2*self.kernel(x_2, x_2)*(new_a_2-a_1) + self.b
        self.b = (new_b_1 + new_b_2)/2
        self.a[index_1] = new_a_1
        self.a[index_2] = new_a_2

    def obey_kkt(self):
        # 是否服从kkt条件
        for i in range(len(self.labels)):
            sample, label, a_i = self.samples[i], self.labels[i], self.a[i]
            if 0 == a_i:
                tem_gap = 1 - label * self.predict(sample)
            elif a_i == self.c:
                tem_gap = label * self.predict(sample) - 1
            else:
                tem_gap = abs(label * self.predict(sample) - 1)
            if tem_gap > self.e:
                return False
        sum_y = abs(np.dot(self.a, self.labels))
        if sum_y > self.e:
            return False
        return True