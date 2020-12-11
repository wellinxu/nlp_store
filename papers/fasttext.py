"""
fasttext论文复现， 未完成
fasttext的模型结构跟word2vec基本一致，所以基础模型会沿用word2vec中的模型

"""
import tensorflow as tf
import numpy as np
from papers.word2vec import HuffmanTree, BaseWord2vecModel,negetive_sample


class BaseFasttextModel(BaseWord2vecModel):
    def __init__(self, voc_size, emb_dim, ngram_num, output_dim=0, is_huffman=True, is_negative=False):
        super(BaseFasttextModel, self).__init__(voc_size, emb_dim, is_huffman, is_negative, ngram_num)
        # 做文本分类任务的时候，要修改输出的维度大小
        if not is_huffman and not is_negative:
            # 不使用huffman树进行文本分类的时候，输出维度是标签个数，而不是voc_size
            self.output_weight = self.add_weight(shape=(output_dim, emb_dim),
                                                 initializer=tf.zeros_initializer, trainable=True)
        if is_huffman:
            # 使用huffman树进行文本分类的时候，输出维度2*output_dim，而不是2*voc_size
            self.huffman_params = tf.keras.layers.Embedding(2*output_dim, emb_dim,
                                                            embeddings_initializer=tf.keras.initializers.zeros)
        if is_negative:
            # 负采样时，每个词的输出参数
            self.index = np.array([i for i in range(output_dim)])
            self.negative_params = tf.keras.layers.Embedding(output_dim, emb_dim,
                                                             embeddings_initializer=tf.keras.initializers.zeros)

    def predict_one(self, x, huffman_tree: HuffmanTree):
        x = self.embedding(x)  # ［context_len, emb_dim］
        x = tf.reduce_sum(x, axis=-2)  # [emb_dim]

        if not self.is_huffman and not self.is_negative:
            output = tf.einsum("ve,e->v", self.output_weight, x)  # [voc_size]
            output = self.softmax(output)
            y_index = tf.argmax(output).numpy()
            return int(y_index), output

        if self.is_negative:
            output = tf.einsum("ve,e->v", self.negative_params(self.index), x)  # [voc_size]
            output = tf.sigmoid(output)
            y_index = tf.argmax(output).numpy()
            return int(y_index), output

        ps = {}
        ps[0] = 1.0
        maxp, resultw = 0, 0
        for w, code in huffman_tree.word_code_map.items():
            index, curp = 0, 1.0
            for c in code:
                left_index = huffman_tree.nodes_list[index]
                if left_index not in ps.keys():
                    # p = tf.sigmoid(tf.einsum("a,a->", self.huffman_params(np.array([index])), x))
                    param = tf.squeeze(self.huffman_params(np.array([index])))
                    p = tf.sigmoid(tf.einsum("a,a->", param, x))
                    p = p.numpy()
                    ps[left_index] = 1 - p
                    ps[left_index + 1] = p
                index = left_index + c
                curp *= ps[index]
                if curp < maxp: break
            if curp > maxp:
                maxp = curp
                resultw = w
        # print(resultw, ps)
        return resultw, maxp


class Fasttext(object):
    def __init__(self, docs=None, labels=None, emb_dim=10, ngram=(2,), ngram_num=100000,
                 windows=5, negative_num=10, epochs=3, save_path=None, min=3, is_huffman=None, is_negative=None):
        self.docs = docs
        self.labels = labels
        self.emb_dim = emb_dim
        self.ngram = ngram
        self.ngram_num = ngram_num
        self.windows = windows
        self.negative_num = negative_num
        self.epochs = epochs
        self.save_path = save_path
        self.min = min
        self.huffman_tree = None
        self.is_huffman = is_huffman
        self.is_negative = is_negative
        self.is_classify = self.labels is not None
        self.is_embedding = self.labels is None
        if self.is_huffman is None: self.is_huffman = self.is_classify
        if self.is_negative is None: self.is_negative = self.is_embedding
        self.output_dim = 0
        if docs:  # 训练模型
            self.build_word_dict()    # 构建词典，获取词频
            self.create_train_data()    # 创建训练数据
            self.train()    # 进行训练

    def train(self):
        sample_num = len(self.xs)    # 样本数量
        optimizer = tf.optimizers.SGD(0.001)    # 优化器
        # 基础word2vec模型
        self.model = BaseFasttextModel(self.voc_size, self.emb_dim, self.ngram_num, self.output_dim, self.is_huffman, self.is_negative)
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
        # norm = np.expand_dims(np.linalg.norm(self.word_embeddings, axis=1), axis=1)
        # self.word_embeddings /= norm    # 归一化

    def predict(self, doc):
        idoc = [self.word_map[w] for w in doc if w in self.word_map.keys()]
        ngram_features = self._get_ngram(doc, False)
        ngram_features = [self.ngram2id_map[v] for v in ngram_features if v in self.ngram2id_map.keys()]
        idoc.extend(ngram_features)
        w, p = self.model.predict_one(np.array(idoc), self.huffman_tree)
        w = self.id_label_map[w]
        return w, p

    def evaluate(self, docs, labels):
        right = 0
        for doc, label in zip(docs, labels):
            tlabel, p = self.predict(doc)
            # print(tlabel, label)
            if tlabel == label:
                right += 1
        print("准确率：", right/len(labels))

    def create_train_data(self):
        # 构建CBOW或者SG训练数据
        self.xs = []
        self.ys = []
        if self.is_classify:
            for i, doc in enumerate(self.docs):
                idoc = [self.word_map[w] for w in doc if w in self.word_map.keys()]
                label = [self.label_map[self.labels[i]]]
                ngram_features = self.w2ngram_map[i]
                idoc.extend(ngram_features)
                self.xs.append(np.array(idoc))
                self.ys.append(np.array(label))
        if self.is_embedding:
            y_index = self.windows // 2
            for doc in self.docs:
                idoc = [self.word_map[w] for w in doc if w in self.word_map.keys()]
                if len(idoc) < self.windows: continue
                for i in range(0, len(idoc) - self.windows + 1):
                    words = idoc[i:i+self.windows]
                    x = [words.pop(y_index)]
                    x.extend(self.w2ngram_map.get(self.id_word_map[x[0]], []))
                    self.xs.append(np.array(x))
                    self.ys.append(np.array(words))

        # 构建huffman数据
        if self.is_huffman:
            self.huffman_tree = HuffmanTree(self.sort_output)    # 根据词与词频，构建huffman树
            self.huffman_label, self.huffman_index = self.huffman_tree.create_label(self.ys)
        else:    # 不适用huffman的时候，添加空数组
            self.huffman_label = [0 for i in self.ys]
            self.huffman_index = [0 for i in self.ys]

        # 构建负采样数据
        if self.is_negative:
            self.negative_index = negetive_sample(self.sort_output, self.ys, self.negative_num)
        else:    # 不使用负采样的时候，添加空数组
            self.negative_index = [0 for i in self.ys]

    def build_word_dict(self):
        # 构建词典，获取词频
        word_num_map = {}  # 频率
        for doc in self.docs:
            if len(doc) < self.windows: continue
            for word in doc:
                word_num_map[word] = word_num_map.get(word, 0) + 1
        word_num_map = {k: v for k, v in word_num_map.items() if v >= self.min}

        self.word_map = {}  # {词:id}
        for k in word_num_map.keys():
            self.word_map[k] = len(self.word_map)
        self.voc_size = len(self.word_map)  # 词表大小
        self.id_word_map = {v: k for k, v in self.word_map.items()}  # {id:词}

        # 输出相关
        if self.is_embedding:
            # 词频设为原本的0.75次方，根据词频进行负采样的时候，可以降低高词频的概率，提高低词频的概率（高词频的概率仍然大于低词频的概率）
            word_num_map = {k: np.power(v, 0.75) for k, v in word_num_map.items()}
            num = sum(word_num_map.values())
            word_num_map = {k: v / num for k, v in word_num_map.items()}
            self.sort_output = [(v, self.word_map[w]) for w, v in word_num_map.items()]  # (频率，id)
        if self.is_classify:
            self._get_labels()

        # ngram相关
        self.w2ngram_map = {}
        self.ngram2id_map = {}
        if self.is_classify:
            for i, doc in enumerate(self.docs):
                r = self._get_ngram(doc)
                self.w2ngram_map[i] = r
        if self.is_embedding:
            for w in self.word_map.keys():
                r = self._get_ngram("<" + w + ">")
                self.w2ngram_map[w] = r
        self.reduce_ngram_num_by_hash()
        for w in self.w2ngram_map.keys():
            r = self.w2ngram_map[w]
            r = [self.ngram2id_map[v] for v in r]
            self.w2ngram_map[w] = r

    def _get_labels(self):
        self.sort_output = []
        self.label_map = {}
        label_num_map = {}
        for label in self.labels:
            label_num_map[label] = label_num_map.get(label, 0) + 1
        for label in label_num_map.keys():
            self.label_map[label] = len(self.label_map)
        s = sum(label_num_map.values())
        self.sort_output = [(v/s, self.label_map[k]) for k, v in label_num_map.items()]
        self.sort_output.sort()
        self.output_dim = len(self.label_map)
        self.id_label_map = {v:k for k,v in self.label_map.items()}

    def reduce_ngram_num_by_hash(self):
        if len(self.ngram2id_map) > self.ngram_num:
            idmap = {}
            for w in self.ngram2id_map.keys():
                h = abs(hash(w))
                h = h % self.ngram_num
                if h not in idmap.keys():
                    idmap[h] = len(idmap) + self.voc_size
                self.ngram2id_map[w] = h

    def _get_ngram(self, alist, is_train=True):
        result, l = set(), len(alist)
        for n in self.ngram:
            for i in range(l - n):
                w = "".join(alist[i:i + n])
                result.add(w)
                if is_train and w not in self.ngram2id_map.keys():
                    self.ngram2id_map[w] = len(self.ngram2id_map) + self.voc_size
        return result


if __name__ == '__main__':
    # fname = r"D:\BaiduNetdiskDownload\cnews\cnews.val.word.txt"
    def get_data(fname):
        docs = []
        labels = []
        datas = []
        with open(fname, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                lines = line.split(" ")
                lines = [v for v in lines if v and v.strip()]
                datas.append(lines)
        import random
        random.shuffle(datas)
        for lines in datas:
            label = lines.pop(0)
            docs.append(lines)
            labels.append(label)
        return docs, labels
    docs, labels = get_data(r"D:\BaiduNetdiskDownload\cnews\cnews.val.word.txt")
    # test_docs, test_labels = get_data(r"D:\BaiduNetdiskDownload\cnews\cnews.test.word.txt")

    # embedding计算
    fasttext = Fasttext(docs[:10])

    # 分类任务
    # fasttext = Fasttext(docs, labels)
    # fasttext.evaluate(test_docs, test_labels)