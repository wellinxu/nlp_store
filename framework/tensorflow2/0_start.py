"""
tf2.0 快速入门
"""
import tensorflow as tf


# 图片分类
def picture_cls():
    # 下载MNIST数据
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("x_train.shape:", x_train.shape)    #[60000, 28, 28]
    print("y_train.shape:", y_train.shape)    #[60000],共10个类别

    def create_model_by_sequential():
        # 使用sequential构建模型
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),    # [batch_size, 28 * 28]
            tf.keras.layers.Dense(128, activation="relu"),    # [batch_size, 128]
            tf.keras.layers.Dropout(0.2),    # [batch_size, 128]
            tf.keras.layers.Dense(10, activation="softmax")    # [batch_size, 10]
        ])
        return model

    def create_model_by_subclass():
        # 使用模型子类化构建模型
        class MyModel(tf.keras.models.Model):
            def __init__(self):
                super(MyModel, self).__init__()
                self.flatten = tf.keras.layers.Flatten()
                self.d1 = tf.keras.layers.Dense(128, activation="relu")
                self.dropout = tf.keras.layers.Dropout(0.2)
                self.d2 = tf.keras.layers.Dense(10, activation="softmax")

            def call(self, inputs, training=None, mask=None):
                # inputs: [batch_size, 28, 28]
                x = self.flatten(inputs)    # [batch_size, 28 * 28]
                x = self.d1(x)    # [batch_size, 128]
                x = self.dropout(x)    # [batch_size, 128]
                x = self.d2(x)    # [batch_size, 10]]
                return x
        return MyModel()

    # 构建模型
    # model = create_model_by_sequential()    # 使用sequential构建模型
    model = create_model_by_subclass()    # 使用模型子类化构建模型

    # 配置模型训练参数
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # 训练模型
    model.fit(x_train, y_train, epochs=5)
    # 评估测试集
    model.evaluate(x_test,  y_test, verbose=2)


# 文本分类
# 下载IMDB数据
vocab_size = 10000    # 保留词的个数
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
print("train len:", len(train_data))    # [25000]
print("test len:", len(test_data))    # [25000]

# 一个将单词映射到整数索引的词典
word_index = imdb.get_word_index()   # 索引从1开始
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# 统一文本序列长度
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", truncating="post", maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", truncating="post", maxlen=256)


def create_model_by_sequential():
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Embedding(vocab_size, 16))    # [batch_size, seq_len, 16]
    # model.add(tf.keras.layers.GlobalAveragePooling1D())    # [batch_size, 16]
    # model.add(tf.keras.layers.Dense(16, activation='relu'))    # [batch_size, 16]
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))    # [batch_size, 1]
    # 上下这两种方式是完全等价的
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 16),    # [batch_size, seq_len, 16]
        tf.keras.layers.GlobalAveragePooling1D(),    # [batch_size, 16]
        tf.keras.layers.Dense(16, activation='relu'),    # [batch_size, 16]
        tf.keras.layers.Dense(1, activation='sigmoid')    # [batch_size, 1]
    ])
    model.summary()  # 打印网络结构概览
    return model


def create_model_by_subclass():
    # 使用模型子类化构建模型
    class MyModel(tf.keras.models.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.embedding = tf.keras.layers.Embedding(vocab_size, 16)
            self.g_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
            self.d1 = tf.keras.layers.Dense(16, activation="relu")
            self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")

        def call(self, inputs, training=None, mask=None):
            # inputs: [batch_size, seq_len]
            x = self.embedding(inputs)    # [batch_size, seq_len, 16]
            x = self.g_avg_pool(x)    # [batch_size, 16]
            x = self.d1(x)    # [batch_size, 16]
            x = self.d2(x)    # [batch_size, 1]]
            return x
    return MyModel()


model = create_model_by_sequential()    # 使用sequential构建模型
# model = create_model_by_subclass()    # 使用模型子类化构建模型


# 配置模型训练参数
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
# 训练模型
history = model.fit(train_data, train_labels, epochs=40, batch_size=512)
# 评估测试集
model.evaluate(test_data,  test_labels, verbose=2)


# 保存与加载模型
def save_load_by_checkpoint(model):
    # 保存权重
    model.save_weights("checkpoint/my_checkpoint")
    # 加载权重
    new_model = create_model_by_subclass()
    # 预测之前需要先编译
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    new_model.load_weights("checkpoint/my_checkpoint")
    # 评估测试集
    new_model.evaluate(test_data,  test_labels, verbose=2)


def save_load_by_hdf5(model):
    """只能用于Functional model or a Sequential model，目前不能用于subclassed model，2020-06"""
    # 保存模型
    model.save("h5/my_model.h5")
    # 加载模型
    # 重新创建完全相同的模型，包括其权重和优化程序
    new_model = tf.keras.models.load_model('h5/my_model.h5')
    # 显示网络结构
    new_model.summary()
    # 评估测试集
    new_model.evaluate(test_data, test_labels, verbose=2)


def save_load_by_savedmodel(model):
    # 保存模型
    tf.saved_model.save(model, "saved_model/1")
    # 加载模型
    new_model = tf.saved_model.load("saved_model/1")
    # 预测结果
    result = new_model(test_data)
    print("result shape:", result.shape)

# save_load_by_checkpoint(model)
# save_load_by_hdf5(model)
# save_load_by_savedmodel(model)
