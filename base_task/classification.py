"""
文本分类任务
"""
from papers.textcnn import TextCNN
import tensorflow as tf
import numpy as np
import random


def get_THUCNews(filename):
    datas = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            label, text = line.split("\t", maxsplit=1)
            datas.append(text)
            labels.append(label)
    print(filename," 数据读取完成")
    return datas, labels


def get_voc(filename):
    voc_map = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            voc_map[w] = len(voc_map)
    id2w_map = {v:k for k, v in voc_map.items()}
    print(filename, " 词表加载完成")
    return voc_map, id2w_map


def token2id(texts, w2id_map):
    texts_id = []
    for text in texts:
        text_id = [w2id_map[t] for t in text if t in w2id_map.keys()]
        texts_id.append(text_id)
    return texts_id


def label2id(labels, label_map=None):
    if not label_map:
        label_map = {}
        for label in labels:
            if label not in label_map.keys():
                label_map[label] = len(label_map)
    labels_id = [label_map[label] for label in labels]
    return labels_id, label_map


def test_THUCNews():
    train_data, train_label = get_THUCNews("D:\软件数据\百度网盘\THUCNews\cnews.train.txt")
    datas = [(a, b) for a, b in zip(train_data, train_label)]
    random.shuffle(datas)
    train_data = [a for a,_ in datas]
    train_label = [b for _,b in datas]

    test_data, test_label = get_THUCNews("D:\软件数据\百度网盘\THUCNews\cnews.test.txt")
    voc_map, id2w_map = get_voc("D:\软件数据\百度网盘\THUCNews\cnews.vocab.txt")
    train_data = token2id(train_data, voc_map)
    test_data = token2id(test_data, voc_map)
    train_label, label_map = label2id(train_label)
    test_label,_ = label2id(test_label, label_map)

    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value=voc_map["<PAD>"], padding="post",
                                                               truncating="post", maxlen=128)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, value=voc_map["<PAD>"], padding="post",
                                                              truncating="post", maxlen=128)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_dataset = train_dataset.shuffle(buffer_size=512).batch(64)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_dataset = test_dataset.batch(512)

    print("数据转换结束")

    # weights = np.random.rand(len(voc_map), 100)
    # model = TextCNN(output_dim=len(label_map), voc_size=len(voc_map), kernels=[2,3,4], filters=100, emb_dim=100, two_channel=False, embeddings=weights)
    model = TextCNN(output_dim=len(label_map), voc_size=len(voc_map), kernels=[2, 3, 4], filters=100, emb_dim=100,two_channel=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # 训练模型
    print("开始训练模型")
    history = model.fit(train_dataset, epochs=10)

    # 评估测试集
    print("开始评估模型")
    model.evaluate(test_dataset, verbose=2)


if __name__ == '__main__':
    test_THUCNews()