
import numpy as np


class HMM(object):
    def __init__(self, samples, num_hidden, num_obj, labels=None, _lambda=1, num_cycle=500):
        self.samples = samples
        self.labels = labels
        self.A = np.random.randn(num_hidden, num_hidden)
        self.B = np.random.randn(num_hidden, num_obj)
        self.pie = np.random.randn(num_hidden)
        self.num_hidden = num_hidden
        self.num_obj = num_obj
        self._lambda = _lambda
        self.num_cycle = num_cycle    # 无监督学习时迭代的最大次数
        self.tem_alpha = {}
        self.tem_beta = {}
        if labels:
            self.unsupervised_learning()
        else:
            self.supervised_learning()

    def compute_alpha(self, t, o_list):
        # 前向计算
        if t in self.tem_alpha.keys():
            return self.tem_alpha[t]
        alpha = []
        o_i = o_list[t - 1]    # index从0开始，比实际少1
        if t == 1:
            for i, b_i in enumerate(self.B):
                alpha.append(self.pie[i] * b_i[o_i])
        else:
            last_alpha = self.compute_alpha(t - 1, o_list)
            for i , b_i in enumerate(self.B):
                tem_alpha = [last_alpha[j] * self.A[j, i] for j in range(len(last_alpha))]
                tem_alpha = sum(tem_alpha) * b_i[o_i]
                alpha.append(tem_alpha)
        return alpha

    def compute_beta(self, t, o_list):
        # 后向计算
        if t in self.tem_beta.keys():
            return self.tem_beta[t]
        beta = []
        if t == len(o_list):
            for b in self.B:
                beta.append(1)
        else:
            last_beta = self.compute_beta(t + 1, o_list)
            o_i_1 = o_list[t + 1 - 1]     # index从0开始，比实际少1
            for i, b_i in enumerate(self.B):
                tem_beta = [self.A[i, j] * last_beta[j] * self.B[j, o_i_1] for j in range(len(last_beta))]
                tem_beta = sum(tem_beta)
                beta.append(tem_beta)
        return beta

    def alpha_p(self, o_list):
        # 前向算法计算总概率
        alpha = self.compute_alpha(len(o_list), o_list)
        p = sum(alpha)
        self.tem_alpha = {i:v for i, v in enumerate(alpha)}
        return p

    def beta_p(self, o_list):
        # 后向算法计算总概率
        beta = self.compute_beta(1, o_list)
        o_1 = o_list[0]    # index从0开始，比实际少1
        p = [self.pie[i] * self.B[i, o_1] * beta[i] for i in range(len(beta))]
        p = sum(p)
        self.tem_beta = {i:v for i, v in enumerate(beta)}
        return p

    def p_t_i(self, t, i, p, o_list):
        #t步时，隐状态为i的概率
        alpha = self.compute_alpha(t, o_list)
        beta = self.compute_beta(t, o_list)
        return alpha[i] * beta[i] / p

    def p_t_i_j(self, t, i, j, p, o_list):
        #t步隐状态为i，t+1步隐状态为j的概率
        alpha = self.compute_alpha(t, o_list)
        beta = self.compute_beta(t+1, o_list)
        o_t_1 = o_list[t + 1 - 1]
        return alpha[i] * self.A[i, j] * beta[j] * self.B[j, o_t_1] / p

    def supervised_learning(self):
        #监督学习
        label_num_dict = {}
        each_feature_num4label = {}
        first_label = [0 for i in range(self.num_hidden)]
        for sample, label in zip(self.samples, self.labels):
            first_label[label[0]] = first_label[label[0]] + 1
            pre_l = -1
            for s, l in zip(sample, label):
                if l not in label_num_dict.keys():
                    label_num_dict[l] = {}
                    each_feature_num4label[l] = {}
                each_feature_num4label[l][s] = each_feature_num4label[l].get(s, 0) + 1
                if pre_l > -1:
                    label_num_dict[pre_l][l] = label_num_dict[pre_l].get(l, 0) + 1
                    pre_l = l
        self.pie = first_label
        label_dis = []
        for i in range(self.num_hidden):
            tem_label_dis = [0 for i in range(self.num_hidden)]
            i_label_num_dict = label_num_dict.get(i, {})
            i_sum = sum([v for k, v in i_label_num_dict.items()]) + self.num_hidden * self._lambda
            for j in range(self.num_hidden):
                tem_label_dis[j] = (i_label_num_dict.get(j, 0) + self._lambda)/i_sum
            label_dis.append(tem_label_dis)
        self.A = np.array(label_dis)
        label_feature_dis = []
        for i in range(self.num_hidden):
            tem_label_feature_dis = [0 for i in range(self.num_obj)]
            i_label_feature_num = each_feature_num4label.get(i, {})
            i_sum = sum([v for k, v in i_label_feature_num.items()]) + self.num_obj * self._lambda
            for j in range(self.num_obj):
                tem_label_feature_dis[j] = (i_label_feature_num.get(j, 0) + self._lambda)/i_sum
            label_feature_dis.append(tem_label_feature_dis)
        self.B = np.array(label_feature_dis)

    def unsupervised_learning(self):
        # 非监督学习
        for epoch in range(self.num_cycle):
            for sample in self.samples:
                p_o = self.alpha_p(sample)
                self.beta_p(sample)
                for i in range(self.num_hidden):
                    for j in range(self.num_hidden):
                        self.A[i, j] = sum([self.p_t_i_j(t, i, j, p_o, sample) for t in range(len(sample))])/p_o
                    for k in range(self.num_obj):
                        self.B[i, k] = sum([self.p_t_i(t, i, p_o, sample) for t in range(len(sample))])/p_o
                    self.pie[i] = self.p_t_i(0, i, p_o, sample)

    def predict_(self, sample):
        # 近似解法
        hidden = []
        p = self.alpha_p(sample)
        self.beta_p(sample)
        for hi, obj in sample:
            p_list = [self.p_t_i(hi, i, p, sample) for i in range(len(self.num_hidden))]
            hidden.append(np.argmax(p_list))
        return hidden

    def predict(self, sample):
        # 维特比解法
        a = [[]]
        b = [[]]
        for i in range(self.num_hidden):
            a[0][i] = self.pie[i]*self.B[i, sample[0]]
            b[0][i] = 0
        for t in range(1, len(sample)):
            for i in range(self.num_hidden):
                tem_a = [a[t-1][i] * self.A[j, i] for j in range(self.num_hidden)]
                max_ind = np.argmax(tem_a)
                a[t][i] = tem_a[max_ind] * self.B[i, sample[t]]
                b[t][i] = max_ind
        last_a = a[-1]
        hidden = []
        index = np.argmax(last_a)
        hidden.insert(0, index)
        t = len(sample) - 1
        while t >= 0:
            t -= 1
            index = b[t][index]
            hidden.index(0, index)
        return hidden



