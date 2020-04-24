
import numpy as np


class NB(object):
    def __init__(self, samples, labels, _lambda=1):
        self.samples = samples    # 0,1特征
        self.labels = labels
        self._lambda = _lambda    # 拉普拉斯平滑，防止为0的概率出现
        self.each_label_dis = {}
        self.each_feature_dis4label = {}
        self.build()

    def build(self):
        samples_num = len(self.samples)
        samples_dict = {}
        for sample, label in zip(self.samples, self.labels):
            if label not in samples_dict.keys():
                samples_dict[label] = []
            samples_dict[label].append(sample)
        each_label_num = {k:len(v) for k, v in samples_dict.items()}
        # 每个类出现的概率
        self.each_label_dis = {k:(v+self._lambda)/(samples_num+len(each_label_num)*self._lambda) for k, v in each_label_num.items()}
        for tem_label, tem_samples in samples_dict.items():
            # 各个类下，每个特征出现的概率
            feature_sum = np.sum(tem_samples, axis=0)    #[n_sample, n_feature]-->[n_feature]
            tem_feature_dis = [(v+self._lambda)/(each_label_num[tem_label]+len(feature_sum)*self._lambda) for v in feature_sum]
            self.each_feature_dis4label[tem_label] = tem_feature_dis

    def predict(self, sample):
        cls = 0
        max_score = 0
        for label, label_dis in self.each_feature_dis4label.items():
            feature_dis = self.each_feature_dis4label[label]
            score = np.matmul(feature_dis, sample)
            score = np.prod(score) * label_dis
            if score > max_score:
                max_score = score
                cls = label
        return cls



