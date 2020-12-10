"""
word2vec的复现，前向传播与反向传播都是使用的tensorflow2
本文实现了：
    1、CBOW与SG两种结构
    2、huffman树与负采样两种加速方式
    3、将huffman树压缩为数组

训练：SGD,考虑到每个样本、每个标签在huffman树上的编码不同，所以每次只训练了一个样本（可以通过mask等操作，进行batch训练）

使用方式：
# 训练, 训练时docs为必要参数，docs是嵌套list:[[w1,w2,...,wn]]
word2vec = Word2vec(docs, emb_dim=50, is_cbow=True, is_huffman=True, is_negative=True, save_path="word2vec.txt")

# 使用模型
word2vec = Word2vec() # 初始化
word2vec.load_txt("word2vec.txt") # 加载模型
print(word2vec.similarity(["湖人", "首发"])) # 计算相似词
"""
import tensorflow as tf
import numpy as np
import random
import time


class BaseWord2vecModel(tf.keras.models.Model):
    # 当前的实现没有batch维度，是一个样本一个样本进行训练
    def __init__(self, voc_size, emb_dim, is_huffman=True, is_negative=False):
        super(BaseWord2vecModel, self).__init__()
        self.voc_size = voc_size
        self.is_huffman = is_huffman
        self.is_negative = is_negative
        self.embedding = tf.keras.layers.Embedding(voc_size, emb_dim, embeddings_regularizer="l2")
        if not self.is_huffman and not is_negative:
            # 不使用huffman树也不使用负采样，所有词的输出参数
            self.output_weight = self.add_weight(shape=(voc_size, emb_dim),
                                                 initializer=tf.zeros_initializer, trainable=True)
            self.softmax = tf.keras.layers.Softmax()
        if self.is_huffman:
            # 所有节点的参数，huffman树压缩为数组的时候，保留了所有叶子节点，所以数组长度为2*voc_size
            # 也可以选择只保留非叶子节点，这样长度可减半
            self.huffman_params = tf.keras.layers.Embedding(2*voc_size, emb_dim,
                                                            embeddings_initializer=tf.keras.initializers.zeros)
            self.huffman_choice = tf.keras.layers.Embedding(2, 1, weights=(np.array([[-1], [1]]),))
        if self.is_negative:
            # 负采样时，每个词的输出参数
            self.negative_params = tf.keras.layers.Embedding(voc_size, emb_dim,
                                                             embeddings_initializer=tf.keras.initializers.zeros)

    def call(self, inputs, training=None, mask=None):
        # x:［context_len］
        # huffman_label: [label_size, code_len]
        # huffman_index: [label_size, code_len]
        # y : [label_size]
        # negative_index: [negatuve_num]
        x, huffman_label, huffman_index, y, negative_index = inputs
        x = self.embedding(x)    # ［context_len, emb_dim］
        x = tf.reduce_sum(x, axis=-2)    # [emb_dim]

        loss = 0
        if not self.is_huffman and not self.is_negative:
            # 不使用huffman树也不使用负采样，则使用原始softmax
            output = tf.einsum("ve,e->v", self.output_weight, x)    # [voc_size]
            output = self.softmax(output)
            y_index = tf.one_hot(y, self.voc_size)    # [label_size, voc_size]
            y_index = tf.reduce_sum(y_index, axis=0)    # [voc_size]
            l = tf.einsum("a,a->a", output, y_index)    # [voc_size]
            loss -= tf.reduce_sum(l)

        # huffman树loss计算
        if self.is_huffman:
            # # 各个label的code_len不一致，所以不能一起算，除非补齐再添加一个mask输入
            # # 获取huffman树编码上的各个节点参数
            # huffman_param = self.huffman_params(huffman_label)  # [label_size, code_len, emb_dim]
            # # 各节点参数与x点积
            # huffman_x = tf.einsum("lab,b->la", huffman_param, x)  # [label_size,code_len]
            # # 获取每个节点是左节点还是右节点
            # tem_label = tf.squeeze(self.huffman_choice(huffman_label), axis=-1)  # [label_size, code_len]
            # # 左节点：sigmoid(-WX),右节点sigmoid(WX)
            # l = tf.sigmoid(tf.einsum("la,la->la", huffman_x, tem_label))  # [label_size, code_len]
            # loss -= tf.reduce_sum(tf.math.log(l))
            for tem_label, tem_index in zip(huffman_label, huffman_index):
                # 获取huffman树编码上的各个节点参数
                huffman_param = self.huffman_params(tem_index)    # [code_len, emb_dim]
                # 各节点参数与x点积
                huffman_x = tf.einsum("ab,b->a", huffman_param, x)    # [code_len]
                # 获取每个节点是左节点还是右节点
                tem_label = tf.squeeze(self.huffman_choice(tem_label), axis=-1)    # [code_len]
                # 左节点：sigmoid(-WX),右节点sigmoid(WX)
                l = tf.sigmoid(tf.einsum("a,a->a", huffman_x, tem_label))    # [code_len]
                l = tf.math.log(l)
                loss -= tf.reduce_sum(l)

        # 负采样loss计算
        if self.is_negative:
            y_param = self.negative_params(y)    # [label_size, emb_dim]
            negative_param = self.negative_params(negative_index)    # [negative_num, emb_dim]
            y_dot = tf.einsum("ab,b->a", y_param, x)    # [label_size]
            y_p = tf.math.log(tf.sigmoid(y_dot))    # [label_size]
            negative_dot = tf.einsum("ab,b->a", negative_param, x)    # [negative_num]
            negative_p = tf.math.log(tf.sigmoid(-negative_dot))    # [negative_num]
            l = tf.reduce_sum(y_p) + tf.reduce_sum(negative_p)
            loss -= l
        # 错误思想，不是将softmax的分母减少
        # if self.is_negative:
        #     y_param = self.negative_params(y)    # [label_size, emb_dim]
        #     negative_param = self.negative_params(negative_index)    # [negative_num, emb_dim]
        #     y_dot = tf.einsum("ab,b->a", y_param, x)    # [label_size]
        #     y_exp = tf.exp(y_dot)    # [label_size]
        #     negative_dot = tf.einsum("ab,b->a", negative_param, x)    # [negative_num]
        #     negative_exp = tf.exp(negative_dot)    # [negative_num]
        #     y_sum = tf.reduce_sum(y_exp)    # 分子
        #     negative_sum = tf.reduce_sum(negative_exp)    # 负样本的分母
        #     loss -= tf.math.log(y_sum/(y_sum+negative_sum))
        return loss


class Word2vec(object):
    def __init__(self, docs=None, emb_dim=100, windows=5, negative_num=10, is_cbow=True, is_huffman=True, is_negative=False, epochs=5, save_path=None, min=3):
        self.docs = docs    # [[我 是 一段 文本],[这是 第二段 文本]]
        self.windows = windows    # 窗口长度
        self.emb_dim = emb_dim    # 词向量维度
        self.is_cbow = is_cbow   # 是否使用CBOW模式，False则使用SG模式
        self.is_huffman = is_huffman    # 是否使用huffman树
        self.is_negative = is_negative    # 是否使用负采样
        self.huffman_label = []    # huffman数据的标签，判断每次选择左子树还是右子树
        self.huffman_index = []   # huffman数据的编码，用来获取编码上节点的权重
        self.negative_index = []    # 负采样的词索引
        self.negative_num = negative_num    # 负采样数量
        self.epochs = epochs    # 训练轮次
        self.save_path = save_path    # 模型保存路径
        self.min = min    # 最低出现次数
        if docs:  # 训练模型
            self.build_word_dict()    # 构建词典，获取词频
            self.create_train_data()    # 创建训练数据
            self.train()    # 进行训练
            if self.save_path:
                self.save_txt(self.save_path)    # 保存词向量
        elif self.save_path:  # 直接加载词向量
            self.load_txt(self.save_path)

    def train(self):
        sample_num = len(self.xs)    # 样本数量
        optimizer = tf.optimizers.SGD(0.01)    # 优化器
        # 基础word2vec模型
        self.model = BaseWord2vecModel(self.voc_size, self.emb_dim, self.is_huffman, self.is_negative)
        # 模型训练
        for epoch in range(self.epochs):
            print("start epoch %d" % epoch)
            i = 0
            for inputs in zip(self.xs, self.huffman_label, self.huffman_index,self.ys, self.negative_index):
                with tf.GradientTape() as tape:
                    loss = self.model(inputs)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                if i % 1000 == 0:
                    print("-%s->%f" % ("-" * (i * 100 // sample_num), i * 100 / sample_num) + "%")
                i += 1
        # 获取词向量
        self.word_embeddings = self.model.embedding.embeddings.numpy()
        norm = np.expand_dims(np.linalg.norm(self.word_embeddings, axis=1), axis=1)
        self.word_embeddings /= norm    # 归一化

    def load_txt(self, path):
        self.word_map = {}
        self.word_embeddings = []
        voc_size , emb_dim = 0, 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                lines = line.split(" ")
                if voc_size == 0:
                    voc_size, emb_dim = int(lines[0]), int(lines[1])
                else:
                    w = lines[0]
                    emb = lines[1:]
                    emb = [float(v) for v in emb]
                    self.word_map[w] = len(self.word_map)
                    self.word_embeddings.append(np.array(emb))
            self.word_embeddings = np.array(self.word_embeddings)
            self.id_word_map = {v:k for k, v in self.word_map.items()}

    def save_txt(self, path):
        datas = []
        for i in range(self.voc_size):
            w = self.id_word_map[i]
            emb = list(self.word_embeddings[i])
            emb = [str(v) for v in emb]
            line = w + " " + " ".join(emb)
            datas.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(self.voc_size) + " " + str(self.emb_dim) + "\n")
            for line in datas:
                f.write(line + "\n")

    def similarity(self, words, topn=5):
        indexs = [self.word_map[w] for w in words]
        word_emb = self.word_embeddings[indexs]      # [word_len, embedding]
        similarity = np.einsum("we,se->ws", word_emb, self.word_embeddings)      # [word_len, voc_size]
        topn_index = np.argsort(similarity)
        result = []
        for tindexs, sim in zip(topn_index, similarity):
            temresult = []
            for ind in tindexs[-1:-topn-1:-1]:
                temresult.append((self.id_word_map[ind], sim[ind]))
            result.append(temresult)
        return result

    def create_train_data(self):
        # 构建CBOW或者SG训练数据
        self.xs = []
        self.ys = []
        y_index = self.windows // 2
        for doc in self.docs:
            idoc = [self.word_map[w] for w in doc if w in self.word_map.keys()]
            if len(idoc) < self.windows: continue
            for i in range(0, len(idoc) - self.windows + 1):
                words = idoc[i:i+self.windows]
                # words = [self.word_map[w] for w in words]
                y = [words.pop(y_index)]
                words, y = np.array(words), np.array(y)
                if self.is_cbow:    # 构建CBOW数据
                    self.xs.append(words)
                    self.ys.append(y)
                else:    # 构建SG数据
                    self.xs.append(y)
                    self.ys.append(words)

        # 构建huffman数据
        if self.is_huffman:
            huffman_tree = HuffmanTree(self.words)    # 根据词与词频，构建huffman树
            for labels in self.ys:
                tlabels = []
                for label in labels:
                    tlabels.append(np.array(huffman_tree.word_code_map[label]))
                index = []
                for tlabel in tlabels:
                    tem_index = [0]
                    for l in tlabel[:-1]:
                        ind = huffman_tree.nodes_list[tem_index[-1]] + l
                        tem_index.append(ind)
                    index.append(np.array(tem_index))
                self.huffman_label.append(tlabels)  # 获取标签词在huffman树中的编码
                self.huffman_index.append(index)    # 获取标签词在huffman树中的编码上对应的所有非叶子节点
        else:    # 不适用huffman的时候，添加空数组
            self.huffman_label = [0 for i in self.ys]
            self.huffman_index = [0 for i in self.ys]

        # 构建负采样数据
        if self.is_negative:
            # 如果"我 爱 你 啊"出现的概率分别是0.4,0.2,0.3,,0.1，
            # 那么word_end_p就为[0.4,0.6,0.9, 1.0],即[0.4,0.4+0.2,0.4+0.2+0.3,0.4+0.2+0.3+0.1]
            word_end_p = [self.words[0][0]]    # 每个词出现的概率段
            for i in range(1, self.voc_size):
                word_end_p.append(word_end_p[-1]+self.words[i][0])
            # 为每一条训练数据抽取负样本
            for y in self.ys:
                indexs = []
                while len(indexs) < self.negative_num * len(y):
                    index = self._binary_search(random.random(), word_end_p, 0, self.voc_size-1)
                    # 随机抽取一个词，不能再标签中也不能已经被抽到
                    if index not in indexs and index not in y:
                        indexs.append(index)
                self.negative_index.append(np.array(indexs))
        else:    # 不使用负采样的时候，添加空数组
            self.negative_index = [0 for i in self.ys]

    def _binary_search(self, n, nums, start, end):
        # 二分查找，查找n在nums[start:end]数组的那个位置
        if start == end: return end
        mid = (start+end) >> 1
        if nums[mid] >= n:
            return self._binary_search(n, nums, start, mid)
        return self._binary_search(n, nums, mid+1, end)

    def build_word_dict(self):
        # 构建词典，获取词频
        word_num_map = {}    # 频率
        for doc in self.docs:
            if len(doc) < self.windows: continue
            for word in doc:
                word_num_map[word] = word_num_map.get(word, 0) + 1

        # 词频设为原本的0.75次方，根据词频进行负采样的时候，可以降低高词频的概率，提高低词频的概率（高词频的概率仍然大于低词频的概率）
        word_num_map = {k:np.power(v, 0.75) for k, v in word_num_map.items() if v >= self.min}
        num = sum(word_num_map.values())
        word_num_map = {k: v/num for k, v in word_num_map.items()}

        self.word_map = {}  # {词:id}
        for k in word_num_map.keys():
            self.word_map[k] = len(self.word_map)
        self.words = [(v, self.word_map[w]) for w, v in word_num_map.items()]  # (频率，id)
        self.words.sort()    # 根据频率排序
        self.voc_size = len(self.words)    # 词表大小
        self.id_word_map = {v: k for k, v in self.word_map.items()}  # {id:词}
        if self.is_negative and self.voc_size < self.windows * (self.negative_num+1):
            print("错误！词表太小，不能进行负样本抽样，可以使用将is_negative设置为False")
            raise


class Node(object):
    def __init__(self, key, value):
        self.key = key    # 本文代码里huffman中，非叶子节点都为None
        self.value = value    # 权重
        # 编码，即重跟节点走到本节点的方向，0表示左子树，1表示右子树
        # 010表示跟节点->左子树->右子树->左子树（本节点）
        self.code = []    # 记录当前节点在整个huffman树中的编码
        self.index = 0    # 第几个节点，压缩为数组用，即为该节点在数组型的huffman树种的索引位置
        self.left = None
        self.right = None

    def combine(self, node):
        # 两棵树（两个节点）合并成一个新树，叶子节点放在新树的右子树上
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
    # 用word_code_map表记录根到每个叶子节点的编码
    def __init__(self, words):
        start = time.time()
        words.sort()  # 根据频率排序
        self.words = words    # [(value:频次, key:词)]，由小到大排序
        self.word_code_map = {}    # 词在huffman树中的编码映射
        self.nodes_list = []     # 压缩为数组的huffman树
        self.build_huffman_tree()
        print("build huffman tree end,time:", time.time()-start)

    def build_huffman_tree(self):
        # 构建huffman树
        # 每个元素都构成单节点的树，并按照权重重大到小排列
        # 合并权重最小的两个子树，并以权重和作为新树的权重
        # 将新树按照权重大小插入到序列中
        # 重复上述两步，直到只剩一棵树
        nodes = [Node(key, value) for value, key in self.words]
        while len(nodes) > 1:
            a_node = nodes.pop(0)
            b_node = nodes.pop(0)
            new_node = a_node.combine(b_node)
            left, right = 0, len(nodes)
            l = right - 1
            i = right >> 1
            while True:
                if nodes and nodes[i].value >= new_node.value:
                    if i == 0 or nodes[i-1].value < new_node.value:
                        nodes.insert(i, new_node)
                        break
                    else:
                        right = i
                        i = (left+right) >> 1
                else:
                    if i == 0 or i == l:
                        nodes.insert(i, new_node)
                        break
                    left = i
                    i = (left+right) >> 1

        # 将树压缩为数组
        # 数组中每一个元素都是树种的一个节点，数组中的值表示该节点的左节点的位置，如果值跟索引一样大，表示此节点是叶子节点没有子节点
        # 跟节点 root.index = 0
        # 叶子节点 nodes_list[node.index] = node.index
        # 非叶子节点 nodes_list[node.index] = node.left.index
        # 非叶子节点 nodes_list[node.index] + 1 = node.right.index
        stack = [nodes[0]]
        while stack:
            new_stack = []
            for node in stack:
                node.index = len(self.nodes_list)
                if node.key is not None:   # 叶子结点
                    self.word_code_map[node.key] = node.code  # 保存编码
                    self.nodes_list.append(node)  # 在数组中添加叶子结点本身
                if node.right:
                    node.right.code = node.code + [1]
                    new_stack.append(node.right)
                if node.left:
                    # 在数组相应位置添加该结点的左结点
                    self.nodes_list.append(node.left)
                    node.left.code = node.code + [0]
                    new_stack.append(node.left)
            stack = new_stack
        self.nodes_list = [node.index for node in self.nodes_list]


if __name__ == '__main__':
    # docs = [
    #     ["我是", "第一", "段", "文本", "内容"],
    #     ["这是", "第二", "段", "文本", "内容"],
    #     ["这是", "第三", "段", "文本", "内容"],
    #     ["这是", "第四", "段", "文本", "信息"],
    #     ["这是", "第五", "段", "文本", "信息"],
    # ]
    docs = []
    with open("cnews.val.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            lines = line.split("\t")
            lines = [v for v in lines if v and v.strip()]
            docs.append(lines)
    print("start train word2vec's model")

    # 训练
    word2vec = Word2vec(docs, 50, is_cbow=False, is_huffman=True, is_negative=True, save_path="word2vec.txt")

    # # 测试向量
    # word2vec = Word2vec()
    # word2vec.load_txt("word2vec.txt")
    print(word2vec.similarity(["湖人", "首发"]))
    while True:
        w = input("输入文本：")
        try:
            print(word2vec.similarity([w]))
        except:
            pass