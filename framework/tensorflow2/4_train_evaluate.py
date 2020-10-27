"""
tf2的训练与评估
"""
import tensorflow as tf
import numpy as np

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


# 模型构建
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 16),    # [batch_size, seq_len, 16]
        tf.keras.layers.GlobalAveragePooling1D(),    # [batch_size, 16]
        tf.keras.layers.Dense(16, activation='relu'),    # [batch_size, 16]
        tf.keras.layers.Dense(1, activation='sigmoid')    # [batch_size, 1]
    ])


# 配置模型训练参数
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
# 训练模型
history = model.fit(train_data, train_labels, epochs=40, batch_size=512)
# 评估测试集
model.evaluate(test_data,  test_labels, verbose=2)


# callbacks = [
#     # 提前终止训练
#     tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=2, verbose=1),
#     # 保存中间模型
#     tf.keras.callbacks.ModelCheckpoint(filepath="mymodel_{epoch}", save_best_only=True, monitor="val_loss", verbose=1),
#     # 可视化化损失与指标
#     tf.keras.callbacks.TensorBoard(log_dir="/full_path_to_your_logs", histogram_freq=0, embeddings_freq=0, update_freq="epoch")
# ]
# model.fit(train_data, train_labels, epochs=40, batch_size=512, callbacks=callbacks)


def multi_input_output_model():
    image_input = tf.keras.Input(shape=(32, 32, 3), name="img_input")
    timeseries_input = tf.keras.Input(shape=(None, 10), name="ts_input")

    x1 = tf.keras.layers.Conv2D(3, 3)(image_input)
    x1 = tf.keras.layers.GlobalMaxPooling2D()(x1)

    x2 = tf.keras.layers.Conv1D(3, 3)(timeseries_input)
    x2 = tf.keras.layers.GlobalMaxPooling1D()(x2)

    x = tf.keras.layers.concatenate([x1, x2])

    score_output = tf.keras.layers.Dense(1, name="score_output")(x)
    class_output = tf.keras.layers.Dense(5, activation="softmax", name="class_output")(x)

    model = tf.keras.Model(
        inputs=[image_input, timeseries_input], outputs=[score_output, class_output]
    )

    # 显示模型结构
    tf.keras.utils.plot_model(model, show_layer_names=False)

    # list的形式
    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(1e-3),
    #     loss=[tf.keras.losses.MeanSquaredError(), tf.keras.losses.CategoricalCrossentropy()],
    #     metrics=[
    #         [
    #             tf.keras.metrics.MeanAbsolutePercentageError(),
    #             tf.keras.metrics.MeanAbsoluteError(),
    #         ],
    #         [tf.keras.metrics.CategoricalAccuracy()],
    #     ],
    # )

    # dict的形式
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss={
            "score_output": tf.keras.losses.MeanSquaredError(),
            "class_output": tf.keras.losses.CategoricalCrossentropy(),
        },
        metrics={
            "score_output": [
                tf.keras.metrics.MeanAbsolutePercentageError(),
                tf.keras.metrics.MeanAbsoluteError(),
            ],
            "class_output": [tf.keras.metrics.CategoricalAccuracy()],
        },
        loss_weights={"score_output": 2.0, "class_output": 1.0},
    )

    # 某个输出不为训练，只为预测
    # # list的形式
    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(1e-3),
    #     loss=[None, tf.keras.losses.CategoricalCrossentropy()],
    # )
    #
    # # 或dict的形式
    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(1e-3),
    #     loss={"class_output": tf.keras.losses.CategoricalCrossentropy()},
    # )


    # 随机生成NumPy数据
    img_data = np.random.random_sample(size=(100, 32, 32, 3))
    ts_data = np.random.random_sample(size=(100, 20, 10))
    score_targets = np.random.random_sample(size=(100, 1))
    class_targets = np.random.random_sample(size=(100, 5))

    # # list形式
    # model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=1)
    #
    # # 或者dict形式
    # model.fit(
    #     {"img_input": img_data, "ts_input": ts_data},
    #     {"score_output": score_targets, "class_output": class_targets},
    #     batch_size=32,
    #     epochs=1,
    # )

    # dataset格式
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"img_input": img_data, "ts_input": ts_data},
            {"score_output": score_targets, "class_output": class_targets},
        )
    )
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    model.fit(train_dataset, epochs=1)


# multi_input_output_model()