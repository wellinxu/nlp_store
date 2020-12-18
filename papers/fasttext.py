"""
> 论文标题：Bag of Tricks for Efficient Text Classification
> 论文链接：https://arxiv.org/pdf/1607.01759.pdf

> 论文标题：Enriching Word Vectors with Subword Information
> 论文链接：https://arxiv.org/pdf/1607.04606.pdf

fasttext论文复现， 涉及到上诉两个论文中的模型，文本分类模型与文本表示模型
本文基于tensorflow2实现了：
    1、文本分类模型，cbow+词ngram特征(hash处理)+huffman树
    2、文本表示模型，skip-gram+字符ngram特征（hash处理）+负采样
    3、基于fasttext的词向量计算，尤其是oov的词向量

使用方式：
# 训练, 训练时docs为必要参数，docs是嵌套list:[[w1,w2,...,wn]]
fasttext = Fasttext(docs, emb_dim=50, save_path="fasttext_model")

# 使用模型
fasttext = Fasttext(save_path="fasttext_model") # 初始化
print(fasttext.similarity(["湖人", "首发"])) # 计算相似词

简单根据THUCNews文本分类验证数据集cnews.val.txt的5000条文本进行训练，得到的词向量部分展示结果如下：
输入文本：湖人
[[('湖人', 1.0), ('热火', 0.8925801117031581), ('魔术', 0.872596987597849), ('湖人队', 0.859044780955096), ('火箭队', 0.8552345646456492)]]
输入文本：科比
[[('科比', 1.0), ('罗斯', 0.9040430138875364), ('基德', 0.8889259455892049), ('钱德勒', 0.8829596832629936), ('加索尔', 0.8817821001627595)]]
输入文本：基金
[[('基金', 0.9999999999999997), ('葛', 0.8308545518163584), ('型基金', 0.7967489553455214), ('货币基金', 0.7958789897457055), ('股票', 0.7434350941273405)]]
输入文本：中锋
[[('中锋', 1.0), ('控卫', 0.8781126810222232), ('前锋', 0.8637543640767649), ('得分手', 0.8572004271332887), ('球星', 0.8480197635084927)]]
输入文本：女星
[[('女星', 1.0), ('好莱坞', 0.8078040918742064), ('情侣', 0.7989858395590296), ('美女', 0.7696643578221976), ('最爱', 0.7556225114381871)]]
输入文本：副本
[[('副本', 1.0000000000000002), ('掉落', 0.7979368015657543), ('击杀', 0.7837939625708474), ('战斗', 0.7705947430252393), ('界面', 0.7683014744799672)]]
输入文本：北京
[[('北京', 0.9999999999999998), ('上海', 0.7439026500156447), ('北京市', 0.7205646919151837), ('西安', 0.6659278446008475), ('北京站', 0.6640663416217902)]]
输入文本：上海
[[('上海', 1.0000000000000002), ('深圳', 0.8728466259327158), ('广州', 0.7986657310922648), ('北京站', 0.7757978887922621), ('重庆', 0.7453369654672216)]]
输入文本：政策
[[('政策', 0.9999999999999999), ('各项政策', 0.8479695639437519), ('财政政策', 0.8471574290631535), ('宏观经济', 0.8424638445975352), ('产业政策', 0.8214828391757821)]]
输入文本：上海大学    # 出现频率低的词
[[('上海大学', 1.0), ('上海交通大学', 0.7737606667278534), ('广东', 0.7574011491255976), ('中国国民党', 0.7509025304235472), ('北京理工大学', 0.7366751111982804)]]
输入文本：南京大学   #oov的词
[[('北京大学', 0.93184192289716), ('同济大学', 0.8386322963681418), ('浙江大学', 0.8382096923929014), ('中国人民大学', 0.8297620771590131), ('华南理工大学', 0.8283274278173995)]]

"""
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from papers.word2vec import HuffmanTree, BaseWord2vecModel,negetive_sample
import os
import random


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

    def call(self, inputs, training=None, mask=None):
        if not self.is_negative:
            # 单个样本训练
            return super(BaseFasttextModel, self).call()

        # 批量训练
        # x:［batch, context_len］
        # huffman_label: [batch, label_size, code_len]
        # huffman_index: [batch, label_size, code_len]
        # y : [batch, label_size]
        # negative_index: [batch, negatuve_num]
        x, huffman_label, huffman_index, y, negative_index = inputs
        x = self.embedding(x)    # ［batch, context_len, emb_dim］
        x = tf.reduce_sum(x, axis=-2)    # [batch, emb_dim]

        # 负采样loss计算
        y_param = self.negative_params(y)  # [batch, label_size, emb_dim]
        negative_param = self.negative_params(negative_index)  # [batch, negative_num, emb_dim]
        y_dot = tf.einsum("ble,be->bl", y_param, x)  # [batch, label_size]
        y_p = tf.math.log(tf.sigmoid(y_dot))  # [batch, label_size]
        negative_dot = tf.einsum("bne,be->bn", negative_param, x)  # [batch, negative_num]
        negative_p = tf.math.log(tf.sigmoid(-negative_dot))  # [batch, negative_num]
        l = -tf.reduce_sum(y_p, axis=-1) - tf.reduce_sum(negative_p, axis=-1)  # [batch]
        loss = tf.reduce_mean(l)
        return loss

    def predict_one(self, x, huffman_tree: HuffmanTree):
        x = self.embedding(x)  # ［context_len, emb_dim］
        x = tf.reduce_sum(x, axis=-2)  # [emb_dim]

        # 使用softmax做分类
        if not self.is_huffman and not self.is_negative:
            output = tf.einsum("ve,e->v", self.output_weight, x)  # [voc_size]
            output = self.softmax(output)
            y_index = tf.argmax(output).numpy()
            return int(y_index), output

        # 使用负采样做分类
        if self.is_negative:
            output = tf.einsum("ve,e->v", self.negative_params(self.index), x)  # [voc_size]
            output = tf.sigmoid(output)
            y_index = tf.argmax(output).numpy()
            return int(y_index), output

        # 使用huffman树做分类
        ps = {}
        ps[0] = 1.0
        maxp, resultw = 0, 0
        for w, code in huffman_tree.word_code_map.items():
            index, curp = 0, 1.0
            for c in code:
                left_index = huffman_tree.nodes_list[index]
                if left_index not in ps.keys():
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
        return resultw, maxp


class Fasttext(object):
    def __init__(self, docs=None, labels=None, emb_dim=10, ngram=(3, 4), ngram_num=100000,
                 windows=5, negative_num=10, epochs=3, save_path=None, min=3, is_huffman=None, is_negative=None):
        self.docs = docs    # [[我 是 一段 文本],[这是 第二段 文本]]
        self.labels = labels    # 分类模型的[类别标签]
        self.emb_dim = emb_dim    # 词向量维度
        self.ngram = ngram    # ngram的n大小容器
        self.ngram_num = ngram_num    # ngram特征最大数量，超过这个特征则会使用hash
        self.windows = windows    # 词向量模型的窗口大小
        self.negative_num = negative_num    # 负采样数量
        self.epochs = epochs    # 训练轮次
        self.save_path = save_path    # 保存路径，父路径地址
        self.min = min    # 最低出现频次
        self.huffman_tree = None
        self.is_huffman = is_huffman    # 是否使用huffman树
        self.is_negative = is_negative    # 是否使用负采样
        self.is_classify = self.labels is not None    # 是否是文本分类模型
        self.is_embedding = self.labels is None    # 是否是文本表示模型
        if self.is_huffman is None: self.is_huffman = self.is_classify
        if self.is_negative is None: self.is_negative = self.is_embedding
        self.output_dim = 0    # 分类的输出维度
        if docs:  # 训练模型
            self.build_word_dict()    # 构建词典，获取词频
            self.create_train_data()    # 创建训练数据
            self.train()    # 进行训练
            if self.save_path:
                self.save()
        elif self.save_path:
            self.load()

    def load(self):
        # 加载参数
        config = {}
        with open(self.save_path + "/config.txt", "r", encoding="utf-8") as f:
            config = eval(f.readline())
        self.emb_dim = config["emb_dim"]
        self.ngram = config["ngram"]
        self.ngram_num = config["ngram_num"]
        self.windows = config["windows"]
        self.negative_num = config["negative_num"]
        self.epochs = config["epochs"]
        self.min = config["min"]
        self.is_huffman = config["is_huffman"]
        self.is_negative = config["is_negative"]
        self.is_classify = config["is_classify"]
        self.is_embedding = config["is_embedding"]
        self.output_dim = config["output_dim"]
        self.voc_size = config["voc_size"]
        self.ngram_num = config["ngram_num"]
        # 加载词表
        with open(self.save_path + "/vocab.txt", "r", encoding="utf-8") as f:
            i = 0
            self.word_map, self.ngram2id_map = {}, {}
            for line in f:
                line = line.strip()
                if i < self.voc_size: # 加载词
                    self.word_map[line] = i
                else: # 加载ngram
                    lines = line.split("\t")
                    self.ngram2id_map[lines[0]] = int(lines[1])
                i += 1
        if self.is_embedding:
            # 加载词向量
            self.id_word_map = {v: k for k, v in self.word_map.items()}
            self.ngram_embeddings = {}    # 词向量
            self.word_embeddings = []    # ngram向量
            with open(self.save_path + "/fasttext.txt", "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    lines = line.split(" ")
                    if len(self.word_embeddings) < self.voc_size:
                        emb = [float(v) for v in lines]
                        self.word_embeddings.append(np.array(emb))
                    else:
                        w = int(lines[0])
                        emb = [float(v) for v in lines[1:]]
                        self.ngram_embeddings[w] = np.array(emb)
                for w, ind in self.word_map.items():
                    ngrams = self._get_ngram("<" + w + ">", False)
                    indexs = [self.ngram2id_map[n] for n in ngrams if n in self.ngram2id_map.keys()]
                    tem_emb = [self.ngram_embeddings[i] for i in indexs]
                    tem_emb.append(self.word_embeddings[ind])
                    emb = np.mean(tem_emb, axis=0)
                    norm = np.linalg.norm(emb)
                    emb /= norm
                    self.word_embeddings[ind] = emb
                self.word_embeddings = np.array(self.word_embeddings)
        if self.is_classify:
            # 加载模型
            self.model = BaseFasttextModel(self.voc_size, self.emb_dim, self.ngram_num, self.output_dim,
                                           self.is_huffman,
                                           self.is_negative)
            self.model.load_weights(self.save_path + "/fasttext")
            # 加载标签huffman树
            self.label_map = {}
            self.huffman_tree = HuffmanTree([])
            with open(self.save_path + "/label.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if "\t" in line:
                        line = line.strip()
                        lines = line.split("\t")
                        self.label_map[lines[0]] = len(self.label_map)
                        self.huffman_tree.word_code_map[self.label_map[lines[0]]] = eval(lines[1])
                    else:
                        line = line.strip()
                        self.huffman_tree.nodes_list.append(int(line))
            self.id_label_map = {v: k for k, v in self.label_map.items()}

    def save(self):
        config = {}
        config["emb_dim"] = self.emb_dim
        config["ngram"] = self.ngram
        config["ngram_num"] = self.ngram_num
        config["windows"] = self.windows
        config["negative_num"] = self.negative_num
        config["epochs"] = self.epochs
        config["min"] = self.min
        config["is_huffman"] = self.is_huffman
        config["is_negative"] = self.is_negative
        config["is_classify"] = self.is_classify
        config["is_embedding"] = self.is_embedding
        config["output_dim"] = self.output_dim
        config["voc_size"] = self.voc_size
        config["ngram_num"] = self.ngram_num
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # 保存参数
        with open(self.save_path + "/config.txt", "w", encoding="utf-8") as f:
            f.write(str(config))
        # 保存词表
        with open(self.save_path + "/vocab.txt", "w", encoding="utf-8") as f:
            # 保存词
            for i in range(len(self.word_map)):
                f.write(self.id_word_map[i] + "\n")
            # 保存ngram
            for k, v in self.ngram2id_map.items():
                f.write(k + "\t" + str(v) + "\n")
        if self.is_embedding:
            # 保存词向量
            datas = []
            for emb in self.word_embeddings:
                emb = [str(v) for v in emb]
                datas.append(" ".join(emb))    # 词向量
            for k, v in self.ngram_embeddings.items():
                emb = [str(k)] + [str(i) for i in v]
                datas.append(" ".join(emb))    # ngram向量
            with open(self.save_path + "/fasttext.txt", "w", encoding="utf-8") as f:
                for line in datas:
                    f.write(line + "\n")
        if self.is_classify:
            # 保存分类模型
            self.model.save_weights(self.save_path + "/fasttext")
            # 保存类别huffman树
            with open(self.save_path + "/label.txt", "w", encoding="utf-8") as f:
                for i in range(len(self.label_map)):
                    line = self.id_label_map[i] + "\t"
                    if self.is_huffman:
                        line += str(self.huffman_tree.word_code_map[i])
                    f.write(line + "\n")
                if self.is_huffman:
                    for i in self.huffman_tree.nodes_list:
                        f.write(str(i) + "\n")

    def get_train_data(self, batch=16):
        # 按批次获取训练数据
        data_map = {}
        for inputs in zip(self.xs, self.huffman_label, self.huffman_index, self.ys, self.negative_index):
            size = len(inputs[0])
            if size not in data_map.keys():
                data_map[size] = [[] for i in inputs]
            for i,v in enumerate(inputs):
                data_map[size][i].append(v)
            if len(data_map[size][0]) == batch:
                output = (np.array(v) for v in data_map[size])
                yield output
                del data_map[size]
        for k, v in data_map.items():
            output = (np.array(i) for i in v)
            yield output

    def train(self):
        # lr = 0.01 if self.is_embedding else 0.001
        # optimizer = tf.optimizers.SGD(lr)    # 优化器
        optimizer = tf.optimizers.Adam()
        # 基础fasttext模型
        self.model = BaseFasttextModel(self.voc_size, self.emb_dim, self.ngram_num, self.output_dim, self.is_huffman,
                                       self.is_negative)
        if self.is_classify:
            sample_num = len(self.xs) # 样本数量
            # 模型训练
            for epoch in range(self.epochs):
                print("start epoch %d" % epoch)
                i = 0
                for inputs in zip(self.xs, self.huffman_label, self.huffman_index,self.ys, self.negative_index):
                    with tf.GradientTape() as tape:
                        loss = self.model(inputs)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    if i % 10 == 0:
                        print(
                            "-%s->%f%%, loss:%f" % ("-" * (i * 100 // sample_num), i * 100 / sample_num, loss.numpy()),
                            end="\r")
                    i += 1
        if self.is_embedding:
            batch = 256
            sample_num = len(self.xs)//batch + 1 # 样本数量
            # 模型训练
            for epoch in range(self.epochs):
                print("start epoch %d" % epoch)
                i = 0
                for inputs in self.get_train_data(batch):
                    with tf.GradientTape() as tape:
                        loss = self.model(inputs)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    if i % 10 == 0:
                        print("-%s->%f%%, loss:%f" % ("-" * (i * 100 // sample_num), i * 100 / sample_num, loss.numpy()), end="\r")
                    i += 1
                    if i >= sample_num:
                        sample_num += 1
                print("")

            # 获取词向量
            self.embeddings = self.model.embedding.embeddings.numpy()
            self.word_embeddings = []
            self.ngram_embeddings = {v: self.embeddings[v] for v in self.ngram2id_map.values()}
            for k, v in self.word_map.items():
                ngrams = self.w2ngram_map[k]
                ngrams.append(v)
                nemb = [self.embeddings[n] for n in ngrams]
                emb = np.mean(nemb, axis=0)
                self.word_embeddings.append(emb)
            self.word_embeddings = np.array(self.word_embeddings)
            norm = np.expand_dims(np.linalg.norm(self.word_embeddings, axis=1), axis=1)
            self.word_embeddings /= norm    # 归一化

    def get_word_emb(self, words):
        # 获取词向量，当词不存在时用该词的ngram之和表示
        word_emb = []  # [word_len, embedding]
        for w in words:
            if w in self.word_map.keys():
                word_emb.append(self.word_embeddings[self.word_map[w]])
            else:
                ngrams = self._get_ngram("<" + w + ">", False)
                indexs = [self.ngram2id_map[n] for n in ngrams if n in self.ngram2id_map.keys()]
                tem_emb = [self.ngram_embeddings[i] for i in indexs]
                emb = np.mean(tem_emb, axis=0)
                norm = np.linalg.norm(emb)
                emb /= norm
                word_emb.append(emb)
        return word_emb

    def similarity(self, words, topn=10):
        # 计算相似词
        word_emb = self.get_word_emb(words)
        similarity = np.einsum("we,se->ws", word_emb, self.word_embeddings)      # [word_len, voc_size]
        topn_index = np.argsort(similarity)
        result = []
        for tindexs, sim in zip(topn_index, similarity):
            temresult = []
            for ind in tindexs[-1:-topn-1:-1]:
                temresult.append((self.id_word_map[ind], sim[ind]))
            result.append(temresult)
        return result

    def predict(self, doc):
        # 预测分类
        idoc = [self.word_map[w] for w in doc if w in self.word_map.keys()]
        ngram_features = self._get_ngram(doc, False)
        ngram_features = [self.ngram2id_map[v] for v in ngram_features if v in self.ngram2id_map.keys()]
        idoc.extend(ngram_features)
        w, p = self.model.predict_one(np.array(idoc), self.huffman_tree)
        w = self.id_label_map[w]
        return w, p

    def evaluate(self, docs, labels):
        # 批量评估分类模型
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
        self.word_num_map = {k: v for k, v in word_num_map.items() if v >= self.min}

        self.word_map = {}  # {词:id}
        for k in self.word_num_map.keys():
            self.word_map[k] = len(self.word_map)
        self.voc_size = len(self.word_map)  # 词表大小
        self.id_word_map = {v: k for k, v in self.word_map.items()}  # {id:词}

        # 输出相关
        if self.is_embedding:
            # 词频设为原本的0.75次方，根据词频进行负采样的时候，可以降低高词频的概率，提高低词频的概率（高词频的概率仍然大于低词频的概率）
            word_num_map = {k: np.power(v, 0.75) for k, v in self.word_num_map.items()}
            num = sum(word_num_map.values())
            word_num_map = {k: v / num for k, v in word_num_map.items()}
            self.sort_output = [(v, self.word_map[w]) for w, v in word_num_map.items()]  # (频率，id)
            self.output_dim = len(self.sort_output)
        if self.is_classify:
            self._get_labels()

        # ngram相关
        self.w2ngram_map = {}
        self.ngram2id_map = {}
        self.ngram_num_map = {}
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
            r = [self.ngram2id_map[v] for v in r if v in self.ngram2id_map.keys()]
            self.w2ngram_map[w] = r

    def _get_labels(self):
        # 获取类别标签数据
        self.sort_output = []    # [(频率，标签id)]，按频率从小到大
        self.label_map = {}   # 标签：id
        label_num_map = {}
        for label in self.labels:
            label_num_map[label] = label_num_map.get(label, 0) + 1
        for label in label_num_map.keys():
            self.label_map[label] = len(self.label_map)
        s = sum(label_num_map.values())
        self.sort_output = [(v/s, self.label_map[k]) for k, v in label_num_map.items()]
        self.sort_output.sort()
        self.output_dim = len(self.label_map)    # 输出维度，标签数量
        self.id_label_map = {v:k for k,v in self.label_map.items()}    # id:标签

    def reduce_ngram_num_by_hash(self):
        # 如果ngram特征数量大于制定数量，则让具有同样hash值的ngram特征指向同一个表示向量
        for w, v in self.ngram_num_map.items():
            # TODO 第一个判断可以取消了
            if v >= self.min and w not in self.ngram2id_map.keys():
                self.ngram2id_map[w] = len(self.ngram2id_map) + self.voc_size
        print("len:",len(self.ngram2id_map))
        if len(self.ngram2id_map) > self.ngram_num:
            idmap = {}
            for w in self.ngram2id_map.keys():
                h = abs(hash(w))
                h = h % self.ngram_num    # 用hash值的最后几位作为新hash值
                if h not in idmap.keys():
                    idmap[h] = len(idmap) + self.voc_size
                self.ngram2id_map[w] = h

    def _get_ngram(self, alist, is_train=True):
        # 获取alist中包含的ngram特征
        result, l = set(), len(alist)
        for n in self.ngram:
            for i in range(l - n + 1):
                w = "".join(alist[i:i + n])
                result.add(w)
                if is_train:
                    # 如果是训练阶段，则将ngram特征添加到相应map中
                    if self.is_embedding:
                        self.ngram_num_map[w] = self.ngram_num_map.get(w, 0) + self.word_num_map[alist[1:-1]]
                    else:
                        self.ngram_num_map[w] = self.ngram_num_map.get(w, 0) + 1
        return result


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    def get_data(fname, need_random=True):
        docs = []
        labels = []
        datas = []
        with open(fname, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                lines = line.split(" ")
                lines = [v for v in lines if v and v.strip()]
                datas.append(lines)
        if need_random:
            random.shuffle(datas)
        for lines in datas:
            label = lines.pop(0)
            docs.append(lines)
            labels.append(label)
        return docs, labels

    # embedding计算
    docs, labels = get_data(r"D:\BaiduNetdiskDownload\cnews\cnews.val.word.txt", False)
    fasttext = Fasttext(docs, emb_dim=50, save_path="fasttext_model")
    # fasttext = Fasttext(save_path="fasttext_model")
    print(fasttext.similarity(["湖人"]))
    while True:
        w = input("输入文本：")
        try:
            print(fasttext.similarity([w]))
        except:
            pass

    # 分类任务
    # docs, labels = get_data(r"D:\BaiduNetdiskDownload\cnews\cnews.val.word.txt")
    # test_docs, test_labels = get_data(r"D:\BaiduNetdiskDownload\cnews\cnews.test.word.txt")
    # fasttext = Fasttext(docs, labels, save_path="fasttext_model")
    # fasttext.evaluate(test_docs, test_labels)
