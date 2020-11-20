"""
word2vec的复现,未完成
"""
import tensorflow as tf
import numpy as np
import random


class BaseModel(tf.keras.models.Model):
    def __init__(self, voc_size, emb_dim, is_cbow=True, is_huffman=True, is_negative=False, huffman_node_list=None):
        super(BaseModel, self).__init__()
        self.is_cbow = is_cbow
        self.is_huffman = is_huffman
        self.is_negative = is_negative
        self.embedding = tf.keras.layers.Embedding(voc_size, emb_dim)
        self.huffman_node_list = huffman_node_list
        if self.is_huffman:
            self.huffman_params = tf.keras.layers.Embedding(2*voc_size, emb_dim)
            # todo bug
            self.huffman_choice = tf.keras.layers.Embedding(2, 1, weights=[-1, 1])
        if self.is_negative:
            self.negative_params = tf.keras.layers.Embedding(voc_size, emb_dim)

    def call(self, inputs, training=None, mask=None):
        x, huffman_label, huffman_index, y, negative_index = inputs
        x = self.embedding(x)
        if self.is_cbow:
            x = tf.reduce_sum(x, axis=-2)
        loss = 0
        if self.is_huffman:
            huffman_param = self.huffman_params(huffman_index)    # [path_len, emb_dim]
            huffman_x = tf.einsum("ab,b->a", huffman_param, x)    # [path_len]
            # todo bug
            huffman_label = self.huffman_choice(huffman_label)    # [path_len]
            loss += tf.reduce_prod(tf.sigmoid(tf.einsum("a,a->a", huffman_x, huffman_label)))
        if self.is_negative:
            y_param = self.negative_params(y)
            y_weight = self.embedding(y)
            negative_param = self.negative_params(negative_index)
            negative_weight = self.embedding(negative_index)
            y_dot = tf.einsum("ab,ab->a", y_param, y_weight)
            negative_dot = tf.einsum("ab,ab->a", negative_param, negative_weight)
            y_sum = tf.reduce_sum(y_dot)
            negative_sum = tf.reduce_sum(negative_dot)
            loss += y_sum/(y_sum+negative_sum)
        return loss


class Word2vec(object):
    def __init__(self, docs, emb_dim, windows=5, is_cbow=True, is_huffman=True, is_negative=False, epochs=10):
        self.docs = docs    # [[我 是 一段 文本],[这是 第二段 文本]]
        self.windows = windows
        self.emb_dim = emb_dim
        self.is_cbow = is_cbow
        self.is_huffman = is_huffman
        self.is_negative = is_negative
        self.huffman_label = []
        self.huffman_index = []
        self.huffman_node_list = []
        self.negative_index = []
        self.negative_num = windows * 1
        self.epochs = epochs
        self.build_word_dict()
        self.create_train_data()
        self.train()

    def train(self):
        optimizer = tf.optimizers.SGD(0.01)
        self.model = BaseModel(self.voc_size, self.emb_dim, self.is_cbow, self.is_huffman, self.is_negative, self.huffman_node_list)
        for epoch in range(self.epochs):
            for x, huffman_label, huffman_index, y, negative_index in zip(self.xs, self.huffman_label, self.huffman_index,self.ys, self.negative_index):
                inputs = np.array(x), np.array(huffman_label), np.array(huffman_index), np.array(y), np.array(negative_index)
                with tf.GradientTape() as tape:
                    loss = self.model(inputs)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def create_train_data(self):
        self.xs = []
        self.ys = []
        y_index = self.windows // 2
        for doc in self.docs:
            if len(doc) < self.windows:
                continue
            for i in range(0, len(doc) - self.windows + 1):
                words = doc[i:i+self.windows]
                words = [self.word_map[w] for w in words]
                y = [words.pop(y_index)]
                if self.is_cbow:
                    self.xs.append(words)
                    self.ys.append(y)
                else:
                    self.xs.append(y)
                    self.ys.append(words)
        if self.is_huffman:
            huffman_tree = HuffmanTree(self.words)
            self.huffman_node_list = huffman_tree.nodes_list
            for labels in self.ys:
                tlabels = []
                for label in labels:
                    tlabels.append(huffman_tree.word_path_map[label])
                self.huffman_label.append(tlabels)
                index = [0]
                for l in tlabels[:-1]:
                    ind = self.huffman_node_list[index[-1]] + l
                    index.append(l)
                self.huffman_index.append(index)
        if self.is_negative:
            word_end_p = [self.words[0][0]]
            for i in range(1, self.voc_size):
                word_end_p.append(word_end_p[-1]+self.words[i][0])
            for y in self.ys:
                indexs = []
                while len(indexs) < self.negative_num:
                    index = self._binary_search(random.random(), word_end_p, 0, self.voc_size-1)
                    if index not in indexs and index not in y:
                        indexs.append(index)
                self.negative_index.append(indexs)

    def _binary_search(self, n, nums, start, end):
        if start == end: return end
        mid = (start+end) >> 1
        if nums[mid] >= n:
            return self._binary_search(n, nums, start, mid)
        return self._binary_search(n, nums, mid+1, end)

    def build_word_dict(self):
        self.words = []    # (频率，id)
        self.word_map = {}    # id
        word_num_map = {}    # 频率
        for doc in self.docs:
            for word in doc:
                if word in self.word_map:
                    word_num_map[word] += 1
                else:
                    self.word_map[word] = len(self.word_map)
                    self.words.append(word)
                    word_num_map[word] = 1
                # word_map[word] = word_map.get(word, len(word_map))
                # word_num_map[word] = word_num_map.get(word, 0) + 1
        word_num_map = {k:np.power(v, 0.75) for k, v in word_num_map.items()}
        num = sum(word_num_map.values())
        word_num_map = {k: v/num for k, v in word_num_map.items()}
        self.words = [(word_num_map[w], self.word_map[w]) for w in self.words]
        self.words.sort()    # (频率，id)
        self.voc_size = len(self.words)


class Node(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.path = []
        self.index = 0
        self.left = None
        self.right = None

    def combine(self, node):
        new_node = Node(None, self.value + node.value)
        if self.key is not None:
            new_node.right = self
            new_node.left = node
        else:
            new_node.right = node
            new_node.left = self
        return new_node


class HuffmanTree(object):
    # 用一个数组表示huffman树的所有非叶子节点
    # 用map表记录根到每个叶子节点的路径
    def __init__(self, words):
        self.words = words    # value=频次, key=词，由小到大排序
        self.word_path_map = {}    # 词在huffman树中的路径映射
        self.nodes_list = []     # 压缩为数组的huffman树
        self.build_huffman_tree()

    def build_huffman_tree(self):
        nodes = [Node(key, value) for value, key in self.words]
        while len(nodes) > 1:
            a_node = nodes.pop(0)
            b_node = nodes.pop(0)
            new_node = a_node.combine(b_node)
            i = 0
            for node in nodes:
                if new_node.value <= node.value:
                    break
                i += 1
            nodes.insert(i, new_node)

        # 将树压缩为数组
        stack = [nodes[0]]
        while stack:
            new_stack = []
            for node in stack:
                node.index = len(self.nodes_list)
                if node.key is not None:
                    self.word_path_map[node.key] = node.path
                if node.right:
                    node.right.path = node.path + [1]
                    new_stack.append(node.right)
                if node.left:
                    self.nodes_list.append(node.left)
                    node.left.path = node.path + [0]
                    new_stack.append(node.left)
                else:
                    self.nodes_list.append(node)
            stack = new_stack
        self.nodes_list = [node.index for node in self.nodes_list]


if __name__ == '__main__':
    docs = [
        ["这是", "第一", "段", "文本", "内容"],
        ["这是", "第二", "段", "文本", "内容"],
        ["这是", "第三", "段", "文本", "内容"],
        ["这是", "第四", "段", "文本", "内容"],
        ["这是", "第五", "段", "文本", "内容"],
    ]
    word2vec = Word2vec(docs, 10, is_negative=True)