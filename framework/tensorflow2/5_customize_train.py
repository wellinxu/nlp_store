"""
tf2 自定义训练
"""
import tensorflow as tf

# 下载IMDB数据
vocab_size = 10000    # 保留词的个数
imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

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

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=512).batch(512)

val_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
val_dataset = val_dataset.batch(512)


class MyModel(tf.keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 16)
        self.g_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.d1 = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer="l2")
        self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        # inputs: [batch_size, seq_len]
        x = self.embedding(inputs)  # [batch_size, seq_len, 16]
        x = self.g_avg_pool(x)  # [batch_size, 16]
        x = self.d1(x)  # [batch_size, 16]
        x = self.d2(x)  # [batch_size, 1]]
        return x


model = MyModel()

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.BinaryAccuracy()


# 简单训练与评估过程
def simple():
    # 训练过程
    for epoch in range(20):
        print("\nStart of epoch %d" % (epoch,))
        for step, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x)    # 前向计算过程
                loss_value = loss(y, logits)    # 计算loss
            # 计算梯度
            grads = tape.gradient(loss_value, model.trainable_variables)
            # 根据梯度更新权重
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 评估过程
    for step, (x, y) in enumerate(val_dataset):
        logits = model(x)
        metrics(y, logits)
    print(metrics.result())    # 0.8721782


# 训练过程中监控metric的变化
def metric():
    # 训练过程
    for epoch in range(20):
        metrics.reset_states()    # 度量重置
        for step, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x)    # 前向计算过程
                loss_value = loss(y, logits)    # 计算loss
            metrics.update_state(y, logits)    # 更新度量
            if step % 10 == 0:    # 显示度量
                print(epoch, step, metrics.result())
            # 计算梯度
            grads = tape.gradient(loss_value, model.trainable_variables)
            # 根据梯度更新权重
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(epoch, metrics.result())

    # 评估过程
    metrics.reset_states()
    for step, (x, y) in enumerate(val_dataset):
        logits = model(x)
        metrics(y, logits)
    print(metrics.result())    # 0.8741081


# 添加前向过程中产生的loss
def losses():
    # 训练过程
    for epoch in range(20):
        print("\nStart of epoch %d" % (epoch,))
        for step, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x)    # 前向计算过程
                loss_value = loss(y, logits)    # 计算loss
                loss_value += sum(model.losses)    # 添加正则化loss或add_loss得到的其他loss
            # 计算梯度
            grads = tape.gradient(loss_value, model.trainable_variables)
            # 根据梯度更新权重
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 评估过程
    for step, (x, y) in enumerate(val_dataset):
        logits = model(x)
        metrics(y, logits)
    print(metrics.result())    # 0.86235625


# @tf.function 静态图运行提速
def function():
    import time

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x)  # 前向计算过程
            loss_value = loss(y, logits)  # 计算loss
            loss_value += sum(model.losses)  # 添加正则化loss或add_loss得到的其他loss
        # 计算梯度
        grads = tape.gradient(loss_value, model.trainable_variables)
        # 根据梯度更新权重
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


    start = time.time()
    # 训练过程
    for epoch in range(20):
        print("\nStart of epoch %d" % (epoch,))
        for step, (x, y) in enumerate(train_dataset):
            train_step(x, y)
    print(time.time()-start)    # 13.017641544342041

    # start = time.time()
    # # 训练过程
    # for epoch in range(20):
    #     print("\nStart of epoch %d" % (epoch,))
    #     for step, (x, y) in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             logits = model(x)    # 前向计算过程
    #             loss_value = loss(y, logits)    # 计算loss
    #             loss_value += sum(model.losses)    # 添加正则化loss或add_loss得到的其他loss
    #         # 计算梯度
    #         grads = tape.gradient(loss_value, model.trainable_variables)
    #         # 根据梯度更新权重
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # print(time.time()-start)    # 22.17099928855896


class MyModel2(tf.keras.models.Model):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 16)
        self.g_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.d1 = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer="l2")
        self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        # inputs: [batch_size, seq_len]
        x = self.embedding(inputs)  # [batch_size, seq_len, 16]
        x = self.g_avg_pool(x)  # [batch_size, 16]
        x = self.d1(x)  # [batch_size, 16]
        x = self.d2(x)  # [batch_size, 1]]
        return x

    def train_step(self, data):
        x, y = data    # data的结构取决于模型跟传给fit的数据结构
        with tf.GradientTape() as tape:
            y_pred = self(x)    # 前向计算
            # 计算loss
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # 计算梯度
        grads = tape.gradient(loss, self.trainable_variables)
        # 更新参数
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 更新度量
        self.compiled_metrics.update_state(y, y_pred)
        # 返回度量结果，度量中包括loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data  # data的结构取决于模型跟传给fit的数据结构
        y_pred = self(x)  # 前向计算
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # 更新度量
        self.compiled_metrics.update_state(y, y_pred)
        # 返回度量结果，度量中包括loss
        return {m.name: m.result() for m in self.metrics}


model = MyModel2()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
# 训练模型
history = model.fit(train_data, train_labels, epochs=20, batch_size=512)
# 评估测试集
model.evaluate(test_data,  test_labels, verbose=2)


# data中可以包含样本权重等
def sample_weight():
    def train_step(self, data):
        # data的结构取决于模型跟传给fit的数据结构
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x)  # 前向计算
            # 计算loss
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
        # 计算梯度
        grads = tape.gradient(loss, self.trainable_variables)
        # 更新参数
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 更新度量
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        # 返回度量结果，度量中包括loss
        return {m.name: m.result() for m in self.metrics}


# 手动计算loss跟metrics
def loss_and_metrics():
    loss_tracker = tf.keras.metrics.Mean()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics_fn = tf.keras.metrics.BinaryAccuracy()


    class MyModel2(MyModel):
        def train_step(self, data):
            x, y = data    # data的结构取决于模型跟传给fit的数据结构
            with tf.GradientTape() as tape:
                y_pred = self(x)    # 前向计算
                # 计算loss
                loss = loss_fn(y, y_pred)
                loss += sum(self.losses)    # 添加前向过程的loss
            # 计算梯度
            grads = tape.gradient(loss, self.trainable_variables)
            # 更新参数
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # 更新度量
            loss_tracker.update_state(loss)
            metrics_fn.update_state(y, y_pred)
            # 返回度量结果，度量中包括loss
            return {m.name: m.result() for m in self.metrics}

        @property
        def metrics(self):
            return [loss_tracker, metrics_fn]

    model = MyModel2()
    model.compile(optimizer=tf.keras.optimizers.Adam())
    # 训练模型
    model.fit(train_data, train_labels, epochs=20, batch_size=512)

