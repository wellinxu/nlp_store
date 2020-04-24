
import random

import math
import numpy as np


def entropy(labels, weights=1):
    """
    信息熵
    :param labels:
    :param weights:
    :return:
    """
    class_num = {}
    if weights == 1:
        weights = [1 for l in labels]
    length = sum(weights)
    for label, weight in zip(labels, weights):
        # 计算每个类的比重
        class_num[label] = class_num.get(label, 0) + weight
    # 计算每个类的概率
    class_num = {k: v/length for k, v in class_num.items()}
    # 计算每个类的概率对数值，这里用底为2的对数
    class_num_log = {k: math.log2(v) for k, v in class_num.items()}
    entropy = 0
    for k in class_num.keys():
        entropy -= class_num[k] * class_num_log[k]
    return entropy


def infoGain(datas, labels, feature_i, weights=1):
    """
    信息增益
    :param datas:数据集
    :param labels:类标签
    :param feature_i:计算第几个特征
    :param weights:权重
    :return:
    """
    if weights == 1:
        weights = [1 for l in labels]
    # 计算整体类标签的熵
    h_d = entropy(labels, weights)
    # 根据第i个特征划分数据集
    feature_dict = {}
    weight_dict = {}
    for data, label, weight in zip(datas, labels, weights):
        if data[feature_i] not in feature_dict:
            feature_dict[data[feature_i]] = []
            weight_dict[data[feature_i]] = []
        feature_dict[data[feature_i]].append(label)
        weight_dict[data[feature_i]].append(weight)
    g_d = 0    # 计算条件熵
    entropy_dict = {k: entropy(v, weight_dict[k]) for k, v in feature_dict.items()}
    length = sum(weights)
    for k in feature_dict.keys():
        g_d += sum(weight_dict[k])/length * entropy_dict[k]
    return h_d-g_d


def infoGainRate(datas, labels, feature_i, weights=1):
    """
    信息增益比
    :param datas:
    :param labels:
    :param feature_i:
    :param weights:
    :return:
    """
    if weights == 1:
        weights = [1 for l in labels]
    # 信息增益
    g_d = infoGain(datas, labels, feature_i, weights)
    features = [f[feature_i] for f in datas]
    # 第i个特征的信息熵
    h_a = entropy(features, weights)
    g_d_r = g_d / h_a
    return g_d_r


def gini(labels, weights=1):
    """
    基尼指数
    :param labels:
    :param weights:
    :return:
    """
    class_num = {}
    if weights == 1:
        weights = [1 for l in labels]
    for label, weight in zip(labels, weights):
        class_num[label] = class_num.get(label, 0) + weight
    length = sum(weights)
    g_n = 1
    for k, v in class_num.items():
        g_n -= math.pow(v/length, 2)
    return g_n


def giniGain(datas, labels, feature_i, select_fun, weights=1):
    """
    条件基尼指数
    :param datas:
    :param labels:
    :param feature_i:
    :param select_fun:第i个特征是否为某种情况，一般为是否等于某个值
    :param weights:
    :return:
    """
    if isinstance(weights, int) and weights == 1:
        weights = [1 for l in labels]
    datas_1 = []
    weights_1 = []
    datas_2 = []
    weights_2 = []
    for data, label, weight in zip(datas, labels, weights):
        if select_fun(data[feature_i]):
            datas_1.append(label)
            weights_1.append(weight)
        else:
            datas_2.append(label)
            weights_2.append(weight)
    g1 = gini(datas_1, weights_1)
    g2 = gini(datas_2, weights_2)
    g = sum(weights_1)/sum(weights)*g1 + sum(weights_2)/sum(weights)*g2
    return g


class Node(object):
    def __init__(self, datas=None, labels=None, label=None, feature=None, end_node=False):
        self.samples = datas
        self.labels = labels
        self.label = label
        self.feature = feature
        self.value = None
        self.childs = {}
        self.parent = None
        self.end_node = end_node

    def copy(self):
        node = Node(self.samples, self.labels, self.label, self.feature, self.end_node)
        for tem_node in self.childs.values():
            tem_node_copy = tem_node.copy()
            tem_node_copy.parent = node
        return node


class Tree(object):
    def __init__(self, datas, labels, select_fun=infoGain, e=-1, alpha=0, select_fea_fun=lambda x: x, weights=1):
        self.select_fun = select_fun    # 特征选择准则，默认使用信息增益
        self.e = e    # 阈值e
        self.alpha = alpha    # 剪枝用
        self.select_fea_fun = select_fea_fun    # 随机森林选择属性用的
        self.weights = weights    # boost用
        if self.weights == 1:
            self.weights = [1 for i in labels]
        self.root = self.build(datas, labels, [i for i in range(len(datas[0]))], self.weights)

    def predict(self, data):
        node = self.root
        while True:
            if node.end_node:
                return node.label
            feature_value = data[node.feature]
            if feature_value in node.childs.keys():
                node = node.childs[feature_value]
            else:
                return node.label

    def loss_like(self, labels_list):
        """
        计算损失函数
        :param labels_list:
        :return:
        """
        num = 0
        c_list = []
        for labels in labels_list:
            tem_c = entropy(labels)
            num += len(labels)
            c_list.append(tem_c)
        c = 0
        for labels, tem_c in zip(labels_list, c_list):
            c += len(labels) * tem_c
        c += self.alpha * len(labels_list)
        return c

    def get_labels_list(self, node):
        """
        获取node各个子结点中的样本情况
        :param node:
        :return:
        """
        result = []
        if node.end_node:
            result.append(node.labels)
            return result
        for tem_node in node.childs.values():
            result.extend(self.get_labels_list(tem_node))
        return result

    def pruning(self, node):
        if node.end_node:
            return
        for tem_node in node.childs.values():
            self.pruning(tem_node)
        # 计算子结点损失函数
        c_child = self.loss_like(self.get_labels_list(node))
        # 计算父结点损失函数
        c = self.loss_like([node.labels])
        if c < c_child:
            node.end_node = True
            node.childs = {}

    def build(self, datas, labels, feature_list, weights):
        datas = np.array(datas)
        # 根据类别划分数据集
        label_dict = self.get_dict(datas, labels, labels, weights)
        # 获取最多数的类
        tem_label = self.arg_max_label_class(label_dict)
        if len(label_dict) == 1:    # 只有一个类时，返回树
            return Node(datas, labels, labels[0], end_node=True)
        if not feature_list:    # 特征集为空时，返回树
            return Node(datas, labels, tem_label, end_node=True)
        # 选择特征集，随机森林中使用，决策树默认选择所有可用特征，
        # 即selected_feature_list==feature_list
        selected_feature_list = self.select_fea_fun(feature_list)
        # 计算各特征的信息增益（比）或其他指定准则
        info_list = [self.select_fun(datas, labels, i, weights) for i in selected_feature_list]
        max_info = np.max(info_list)
        if max_info < self.e:   # 当信息增益（比）小于阈值值，返回树
            return Node(datas, labels, tem_label, end_node=True)
        max_info_feature = selected_feature_list[np.argmax(info_list)]
        # 根据选择的特征重新划分数据集
        datas_dict = self.get_dict(datas, labels, datas[:, max_info_feature], weights)
        node = Node(datas, labels, tem_label, feature=max_info_feature)
        # 移除以选择的特征
        feature_list.remove(max_info_feature)
        # 递归地构建子树
        for k, v in datas_dict.items():
            node.childs[k] = self.build(v[0], v[1], feature_list.copy(), v[2])
            node.childs[k].parent = node
        return node

    def arg_max_label_class(self, label_dict):
        tem_label = None
        tem_num = 0
        for k, v in label_dict.items():
            if sum(v[2]) > tem_num:
                tem_label = k
                tem_num = sum(v[2])
        return tem_label

    def get_dict(self, datas, labels, keys, *other):
        """
        根据keys划分datas,labels,other等集合
        :param datas: 数据集
        :param labels: 类
        :param keys: 划分依据
        :param other: 其他集合
        :return:
        """
        label_dict = {}
        for i in range(len(keys)):
            key = keys[i]
            if key not in label_dict:
                label_dict[key] = [[], []]
                for tem_list in other:
                    label_dict[key].append([])
            label_dict[key][0].append(datas[i])
            label_dict[key][1].append(labels[i])
            for j in range(len(other)):
                label_dict[key][j + 2].append(other[j][i])
        return label_dict


class CARTofRegress(Tree):
    def __init__(self, datas, labels, end_fun, loss_fun=None):
        self.end_fun = end_fun
        self.loss_fun = loss_fun
        self.root = self.build(datas, labels)

    def predict(self, data):
        node = self.root
        while True:
            if node.end_node:
                return node.label
            value = data[node.feature]
            if value <= node.value:
                node = node.childs["left"]
            else:
                node = node.childs["right"]

    def arg_min_label(self, labels):
        """
        根据loss函数计算最佳输出，默认为均方loss
        :param labels:
        :return:
        """
        if self.loss_fun is None:
            # 均方loss时，直接返回均值
            return np.average(labels)
        # 其他loss，用一维搜索计算最佳输出
        return self.linear_search(labels)

    def linear_search(self, labels):
        """
        黄金分割搜索法，假设区间在min与max之间
        :param labels:
        :return:
        """
        a = min(labels)
        b = max(labels)
        e = 0.001
        while b - a > e:
            a1 = a * 0.618 + b * 0.382
            b1 = a * 0.382 + b * 0.618
            loss_a1 = sum([self.loss_fun(l, a1) for l in labels])
            loss_b1 = sum([self.loss_fun(l, b1) for l in labels])
            if loss_a1 < loss_b1:
                b = b1
            else:
                a = a1
        return (a + b)/2

    def build(self, datas, labels):
        # 计算此时最佳的输出值
        tem_label = self.arg_min_label(labels)
        if self.end_fun(datas, labels):    # 符合结束条件，则返回树
            node = Node(datas=datas, labels=labels, label=tem_label, end_node=True)
            return node
        # 计算最佳特征与切分点
        feature_i, feature_value = self.split_data(datas, labels)
        node = Node(label=tem_label, feature=feature_i)
        node.value = feature_value
        # 划分区域R1
        data1, labels1 = self.get_child_data(datas, labels, lambda x: x[feature_i] <= feature_value)
        left_node = self.build(data1, labels1)    # 递归构造树
        node.childs["left"] = left_node
        left_node.parent = node
        # 划分区域R2
        data2, labels2 = self.get_child_data(datas, labels, lambda x: x[feature_i] > feature_value)
        right_node = self.build(data2, labels2)
        node.childs["right"] = right_node
        right_node.parent = node
        return node

    def get_child_data(self, datas, labels, fun):
        """
        根据函数划分数据集
        :param datas:
        :param labels:
        :param fun: 划分函数
        :return:
        """
        r = [i for i in range(len(datas)) if fun(datas[i, :])]
        tem_data = datas[r]
        tem_label = labels[r]
        return tem_data, tem_label

    def split_data(self, datas, labels):
        """
        根据loss计算最佳切分特征与切分点，默认是均方loss
        :param datas:
        :param labels:
        :return:
        """
        fun = self.loss_fun
        if self.loss_fun is None:
            fun = lambda x, y: math.pow(x - y, 2)
        tem_feature_i = -1
        tem_feature_value = None
        tem_score = None
        for i in range(len(datas[0])):
            tem_value_set = set(datas[:, i])
            for tem_value in tem_value_set:
                data1, labels1 = self.get_child_data(datas, labels, lambda x: x[i] <= tem_value)
                data2, labels2 = self.get_child_data(datas, labels, lambda x: x[i] > tem_value)
                avg_label1 = np.average(labels1)
                avg_label2 = np.average(labels2)
                sum1 = [fun(v, avg_label1) for v in labels1]
                sum2 = [fun(v, avg_label2) for v in labels2]
                sum = np.sum(sum1) + np.sum(sum2)
                if tem_score is None or sum < tem_score:
                    tem_feature_i = i
                    tem_feature_value = tem_value
                    tem_score = sum
        return tem_feature_i, tem_feature_value


class CARTofClassify(Tree):
    def __init__(self, datas, labels, end_fun, x_val=None, y_val=None, weights=1):
        self.end_fun = end_fun
        if weights == 1:
            self.weights = [1 for l in labels]
        self.x_val = x_val    # 剪枝用
        self.y_val = y_val
        self.root = self.build(datas, labels, [i for i in range(len(datas[0]))], self.weights)

    def loss_like(self, labels_list):
        num = 0
        c_list = []
        for labels in labels_list:
            tem_c = entropy(labels)
            num += len(labels)
            c_list.append(tem_c)
        c = 0
        for labels, tem_c in zip(labels_list, c_list):
            c += len(labels) * tem_c
        return c

    def pruning(self, node):
        min_score_list = []
        min_node_list = []    # 子树序列
        node_i = self.root.copy()
        while self.need_pruning(node_i):
            # 获取最小的a以及对应的结点
            min_score, min_nodes = self.min_node(node_i)
            for min_node in min_nodes:
                min_node.end_node = True
                min_node.childs = {}
            min_score_list.append(min_score)
            min_node_list.append(node_i)
            node_i = node_i.copy()
        # 交叉选择最优子树
        self.root = self.select_node(min_node_list)

    def select_node(self, node_list):
        """
        根据验证集，在子树序列中选择最优子树
        :param node_list: 子树序列
        :return:
        """
        min_error = len(self.y_val)
        node = None
        tree = CARTofClassify()
        for tem_node in node_list:
            tree.root = tem_node
            y_p = [tree.predict(x) for x in self.x_val]
            num = 0
            for y1, y2 in zip(y_p, self.y_val):
                if y1 == y2:
                    num += 1
            if num < min_error:
                min_error = num
                node = tem_node
        return node

    def need_pruning(self, node):
        """
        判断是否需要剪枝
        判断是否有跟结点与两个叶子结点构成的树
        :param node:
        :return:
        """
        if node.end_node:
            return False
        for tem_node in node.childs.values():
            if not tem_node.end_node:
                return True
        return False

    def min_node(self, node):
        """
        递归地结算node内部所有子结点，获取最小的a及其对应结点
        :param node:
        :return:
        """
        if node.end_node:
            return None, [node]
        min_score = None
        min_node = []
        for tem_node in node.childs.values():
            tem_score, tem_node = self.min_node(tem_node)
            if tem_score is not None:
                if min_score is None or tem_score <= min_score:
                    min_score = tem_score
                    if tem_score == min_score:
                        min_node.extend(tem_node)
                    else:
                        min_node = tem_node
        label_list = self.get_labels_list(node)
        c_child = self.loss_like(label_list)
        c = self.loss_like([node.labels])
        g = (c - c_child)/ (len(label_list) - 1)
        if g < 0:
            g = 0
        if min_score is None or g <= min_score:
            min_score = g
            if g == min_score:
                min_node.append(node)
            else:
                min_node = [node]
        return min_score, min_node

    def predict(self, data):
        node = self.root
        while True:
            if node.end_node:
                return node.label
            value = data[node.feature]
            if value == node.value:
                node = node.childs["equal"]
            else:
                node = node.childs["other"]

    def build(self, datas, labels, feature_list, weights):
        # 根据类标签划分数据集
        label_dict = self.get_dict(datas, labels, labels, weights)
        # 获取最多个数的类标签
        tem_label = self.arg_max_label_class(label_dict)
        if self.end_fun(datas, labels):    # 满足结束条件，则返回树
            node = Node(datas=datas, labels=labels, label=tem_label, end_node=True)
            return node
        # 选择最佳切分特征与切分点
        feature_i, feature_value = self.split_data(datas, labels,feature_list, weights)
        node = Node(label=tem_label, feature=feature_i)
        node.value = feature_value
        # 获取数据集R1
        data1, labels1, weights1 = self.get_child_data(datas, labels, weights, lambda x: x[feature_i] == feature_value)
        feature_list1 = [i for i in feature_list if i != feature_i]
        # 递归地生成树
        left_node = self.build(data1, labels1, feature_list1, weights1)
        node.childs["equal"] = left_node
        left_node.parent = node
        # 获取数据集R2
        data2, labels2, weights2 = self.get_child_data(datas, labels, weights, lambda x: x[feature_i] != feature_value)
        right_node = self.build(data2, labels2, feature_list, weights2)
        node.childs["other"] = right_node
        right_node.parent = node
        return node

    def get_child_data(self, datas, labels, weights, fun):
        """
        根据选择函数，选择数据集
        :param datas:
        :param labels:
        :param weights:
        :param fun: 选择函数
        :return:
        """
        r = [i for i in range(len(datas)) if fun(datas[i, :])]
        tem_data = datas[r]
        tem_label = labels[r]
        tem_weights = np.array(weights)[r]
        return tem_data, tem_label, tem_weights

    def split_data(self, datas, labels, feature_list, weights):
        """
        选择最佳切分特征与切分点
        :param datas:
        :param labels:
        :param feature_list:
        :param weights:
        :return:
        """
        tem_feature_i = -1
        tem_feature_value = None
        tem_score = None
        for i in feature_list:
            tem_value_set = set(datas[:, i])
            for tem_value in tem_value_set:
                g = giniGain(datas, labels, i, lambda x: x==tem_value, weights)
                if tem_score is None or g < tem_score:
                    tem_feature_i = i
                    tem_feature_value = tem_value
                    tem_score = g
        return tem_feature_i, tem_feature_value


class RandomForest(object):
    def __init__(self, datas, labels, tree_num=10, feature_num=None):
        self.datas = datas
        self.labels = labels
        self.tree_num = tree_num    # 决策树个数
        self.trees = self.build()

    def predict(self, sample):
        class_dict = {}
        for tree in self.trees:
            c = tree.predict(sample)
            class_dict[c] = class_dict.get(c, 0) + 1
        num = 0
        c = None
        for k, v in class_dict.items():
            if v > num:
                num = v
                c = k
        return c

    def build(self):
        trees = []
        for i in range(self.tree_num):
            datas, labels = self.select_random_samples()
            tree = Tree(datas, labels, infoGain, select_fea_fun=self.select_fea_fun)
            trees.append(tree)
        return trees

    def select_random_samples(self):
        """
        自助采样
        :return:
        """
        datas = []
        labels = []
        index_list = [i for i in range(len(self.labels))]
        while len(datas) < len(self.labels):
            tem_index = random.choice(index_list)
            datas.append(self.datas[tem_index])
            labels.append(self.labels[tem_index])
        return np.array(datas), np.array(labels)

    def select_fea_fun(self, feature_list):
        """
        随机特征选择时的个数
        :param feature_list:
        :return:
        """
        k = int(math.log2(len(feature_list)))
        if k == 0:
            k = 1
        return random.choices(feature_list, k=k)


class AdaBoost(object):
    def __init__(self, datas, labels, weights=1, classify_model=Tree, m=10):
        self.samples = datas
        self.labels = labels
        if weights == 1:    # 权重
            self.weights = [1 for l in labels]
        else:
            self.weights = weights
        self.claaify_model = classify_model    # 基学习器，如决策树
        self.m = m    # 学习器个数
        self.models = []
        self.rates = []
        self.build()

    def predict(self, sample):
        class_dict = {}
        for model, rate in zip(self.models, self.rates):
            c = model.predict(sample)
            class_dict[c] = class_dict.get(c, 0) + rate
        c = None
        score = 0
        for tem_c, s in class_dict.items():
            if s > score:
                score = s
                c = tem_c
        return c

    def build(self):
        weights = self.weights
        for i in range(self.m):
            model_i = self.claaify_model(self.samples, self.labels, weights=weights)
            e, y_pred = self.error_rate(model_i)    # 计算错误率
            rate = 0.5 * math.log((1-e)/e)    # 计算模型权重
            weights = self.update_weights(weights, rate, y_pred)    # 更新数据权重
            self.models.append(model_i)
            self.rates.append(rate)

    def update_weights(self, weights, rate, y_pred):
        """
        更新数据权重
        :param weights:
        :param rate:
        :param y_pred:
        :return:
        """
        sum = 0
        a = math.exp(-rate)    # 正确的样本权重变小
        b = math.exp(rate)    # 错误的样本权重变大
        new_weights = []
        for w, y, label in zip(weights, y_pred, self.labels):
            if y == label:
                w *= a
            else:
                w *= b
            sum += w
            new_weights.append(w)
        new_weights = [w/sum for w in new_weights]
        return new_weights


    def error_rate(self, model):
        num = 0
        y_pred = [model.predict(sample) for sample in self.samples]
        for y, label in zip(y_pred, self.labels):
            if y != label:
                num +=1
        return num/len(self.labels), y_pred


class BoostingTree():
    """
    以cart回归树为基模型的提升树
    """
    def __init__(self, datas, labels, m=10):
        self.samples = datas
        self.labels = np.array(labels)
        self.m = m
        self.models = []
        self.build()

    def predict(self, sample):
        y = 0
        for model in self.models:
            y += model.predict(sample)
        return y

    def build(self):
        y_pred = np.zeros((len(self.labels,)))
        for i in range(self.m):
            r_m = self.labels - y_pred    # 残差
            model_i = CARTofRegress(self.samples, r_m, end_fun=lambda x, y: len(y) < 2 or len(x[0]) == 0 or max(y) == min(y))
            self.models.append(model_i)
            new_y = [model_i.predict(sample) for sample in self.samples]
            y_pred = y_pred + np.array(new_y)


class GBDT(object):
    def __init__(self, datas, labels, loss_fun, gradient_fun, m=10):
        self.samples = datas
        self.labels = labels
        self.loss_fun = loss_fun    # 损失函数
        self.gradient_fun = gradient_fun    # 损失函数的一阶导函数
        self.m = m
        self.models = []
        self.build()

    def predict(self, sample):
        y = 0
        for model in self.models:
            y += model.predict(sample)
        return y

    def build(self):
        model_0 = CARTofRegress(self.samples, self.labels, loss_fun=self.loss_fun, end_fun=lambda x, y: True)
        self.models.append(model_0)
        y_pred = [model_0.predict(sample) for sample in self.samples]
        for i in range(self.m):
            # 计算梯度
            r_m = np.array([self.gradient_fun(l, y) for l, y in zip(self.labels, y_pred)])
            model_i = CARTofRegress(self.samples, r_m, loss_fun=self.loss_fun, end_fun=lambda x, y: len(y) < 2 or len(x[0]) == 0 or max(y) == min(y))
            self.models.append(model_i)
            new_y = [model_i.predict(sample) for sample in self.samples]
            y_pred = y_pred + np.array(new_y)

    # def arg_min_loss(self):
    #     c = 0
    #     score = None
    #     for tem_c in self.labels:
    #         tem_score = sum([self.loss_fun(l, tem_c) for l in self.labels])
    #         if score is None or tem_score < score:
    #             score = tem_score
    #             c = tem_c
    #     return c


class Stacking(object):
    def __init__(self, datas, labels, models, meta_model):
        self.samples = datas
        self.labels = labels
        self.models = models    # 各个基学习器
        self.meta_model = meta_model    # 元学习器
        self.build()

    def predict(self, sample):
        x = []
        for model in self.models:
            x.append(model.predict(sample))
        return self.meta_model.predict(x)

    def build(self):
        datas = []
        for model in self.models:
            model.build(self.samples, self.labels)
        for sample in self.samples:
            x = [model.predict(sample) for model in self.models]
            datas.append(x)
        self.meta_model.build(datas, self.labels)