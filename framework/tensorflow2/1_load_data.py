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


# 加载生成器数据
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


def write_rfrecord():
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
        print(example)
read_tfrecord()



# d = tf.data.Dataset()
# d.from_tensor_slices()
# d.take(1)
# d.apply()
# d.batch() # 使用
# d.padded_batch()
# d.shuffle() # 使用
# d.map()

