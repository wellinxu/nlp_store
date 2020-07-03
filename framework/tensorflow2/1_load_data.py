"""
tensorflow2.0中数据加载
"""
import tensorflow as tf
import numpy as np


# 加载numpy数据
def load_numpy_data():
    # 随机生成numpy数据
    x = np.random.randint(0, 10, size=(1000, 12))
    y = np.random.randint(0, 2, size=(1000, 1))

    # 用tf.data.Dataset加载numpy数据
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return train_dataset


# 加载生成器数据,数据太大，加载不到内存或显存时，可以使用
def load_gen_data():
    # 数据生成器
    def gen():
        for i in range(125):
            yield np.random.rand(8, 12),np.random.randint(0, 2, size=(8, 1))

    # 用tf.data.Dataset加载生成器数据
    train_dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32), output_shapes=((8, 12), (8, 1)))
    return train_dataset


def trian():
    # train_dataset = load_numpy_data()
    train_dataset = load_gen_data()
    # 打乱并批次化数据集
    train_dataset = train_dataset.shuffle(1000).batch(batch_size=32)

    # 训练模型
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation="softmax")
        ])
    model.compile(optimizer="SGD", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

    model.fit(train_dataset, epochs=5)


# tf_record相关
def create_bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))


def create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def create_int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def serialize_example(x_feature, y_feature):
    """生成example并序列化"""
    feature = {
        'x': x_feature,
        'y': y_feature,
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord():
    with tf.io.TFRecordWriter("test.tf_record") as writer:
        for i in range(100):
            x, y = np.random.rand(12), np.random.randint(0, 2, size=1)
            example = serialize_example(create_float_feature(x), create_int64_feature(y))
            writer.write(example)


def read_tfrecord():
    filenames = ["test.tf_record"]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for i in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(i.numpy())

        print(example.features.feature)


# dateset相关处理
dateset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# map()
dateset = dateset.map(lambda x: x-1)
print(list(dateset.as_numpy_iterator()))    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# shuffle()
dateset = dateset.shuffle(10)
print(list(dateset.as_numpy_iterator()))    # [5, 9, 0, 4, 3, 6, 1, 8, 2, 7]


# filter()
dateset = dateset.filter(lambda x: x < 5)
print(list(dateset.as_numpy_iterator()))    # [0, 1, 2, 3, 4]


# batch()
dateset = dateset.batch(3)
# [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]
print(list(dateset.as_numpy_iterator()))


# padded_batch
dateset = tf.data.Dataset.from_tensor_slices([[i] for i in range(10)])
dateset = dateset.padded_batch(3, padded_shapes=(2,), padding_values=0)
# # [array([[0, 0], [1, 0], [2, 0]]), array([[3, 0], [4, 0], [5, 0]]), array([[6, 0], [7, 0], [8, 0]]), array([[9, 0]])]
print(list(dateset.as_numpy_iterator()))

#repeat()
dateset = dateset.repeat(2)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(list(dateset.as_numpy_iterator()))


# d = tf.data.Dataset()
# d.take(1)
# d.apply()

