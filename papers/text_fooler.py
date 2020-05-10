"""
文本对抗生成算法
论文标题：Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment
论文链接：https://arxiv.org/pdf/1907.11932.pdf
论文代码地址：https://github.com/jind11/TextFooler
"""
import numpy as np


class TextFooler(object):
    def __init__(self, model_f, sentence_sim, e: float, N: int, d: float, word2vec_model):
        self.model_f = model_f    # 模型F
        self.sentence_sim = sentence_sim    # 句子相似度函数
        self.e = e     # 句子相似度阈值
        self.N = N    # 相似词最大选择个数
        self.d = d    # 词相似度阈值
        self.word2vec_model = word2vec_model    # 词向量模型，包含词

    def generate(self, x: list):
        """
        生成对抗样本
        :param x: 原样本
        :return: 对抗样本，如果没生成则返回None
        """
        # 初始化
        x_adv = [wi for wi in x]
        y = self.get_label(x)
        # 计算相似度
        importannt_scores = self.get_important_scores(x_adv)
        W_hat = [[wi, si] for wi, si in zip(x_adv, importannt_scores)]
        sorted(W_hat, key=lambda x: x[1], reverse=True)
        W = []
        # 安相似度排序并去重
        W = [wi for wi,_ in W_hat if wi not in W]
        # 停用词过滤
        W = [wi for wi in W if not self.is_stop_word(wi)]
        for wj in W:
            # 根据词向量相似度获取候选词
            C, c_scores = self.get_sim_words(wj)
            pos_j = self.get_pos(wj)
            # 词性过滤
            C = [ck for ck in C if self.get_pos(ck)==pos_j]
            FC = []    # 最终候选词集合
            label_list = []
            score_list = []
            c_score_list = []
            for ck, c_s in zip(C, c_scores):
                x_hat = [wi if wi != wj else ck for wi in x_adv]
                if self.get_sentence_sim(x_hat, x) > self.e:
                    FC.append(ck)
                    yk = self.get_label(x_hat)
                    score_k = self.get_label_scores(x_hat, yk)
                    label_list.append(yk)
                    score_list.append(score_k)
                    c_score_list.append(c_s)
            if FC and y not in label_list:
                k = np.argmax(c_score_list)
                c = FC[k]
                x_adv = [wi if wi != wj else c for wi in x_adv]
                return x_adv
            else:
                k = np.argmin(score_list)
                c = FC[k]
                x_adv = [wi if wi != wj else c for wi in x_adv]
        return None

    def get_important_scores(self, x: list):
        """
        获取每个词的重要性
        :param x:
        :return: [float]
        """
        important_scores = []
        y = self.get_label(x)
        for i, wi in enumerate(x):
            x_wi = [v for j, v in enumerate(x) if j != i]
            yi = self.get_label(x_wi)
            if yi == y:
                I_wi = self.get_label_scores(x, y) - self.get_label_scores(x_wi, y)
            else:
                I_wi = self.get_label_scores(x, y) - self.get_label_scores(x_wi, y)+\
                self.get_label_scores(x, yi) - self.get_label_scores(x_wi, yi)
            important_scores.append(I_wi)
        return important_scores

    def get_label(self, x: list):
        """获取标签,根据具体情况写"""
        # return self.model_f(x)
        return 0

    def get_label_scores(self, x: list, y: int):
        """获取x样本是y标签的置信度得分，根据具体情况写"""
        # return self.model_f(x, y)
        return 0.5

    def is_stop_word(self, wi: str):
        """判断是否是停用词，根据具体情况写"""
        return False

    def get_sim_words(self, wi: str):
        """获取相似词及其相似度，根据具体情况写"""
        # return self.word2vec_model(wi, self.N, self.d)
        return [], []

    def get_pos(self, wi: str):
        """获取词性，根据具体情况写"""
        return ""

    def get_sentence_sim(self, x1: list, x2: list):
        """获取句子相似度"""
        # return self.sentence_sim(x1, x2)
        return 0.5