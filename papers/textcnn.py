"""
复现textcnn
"""
import tensorflow as tf


class TextCNN(tf.keras.Model):
    def __init__(self, output_dim, voc_size, kernels, filters, emb_dim, embeddings=None, two_channel=False, train_emb=True):
        super(TextCNN, self).__init__()
        self.output_dim = output_dim    # 输出维度
        self.voc_size = voc_size    # 词表带大小
        self.kernels = kernels    # 卷积核宽度list
        self.filters = filters    # 卷积核数量
        self.emb_dim = emb_dim    # 向量维度
        self.embeddings = [embeddings] if embeddings is not None else None    # 预训练向量
        self.train_emb = train_emb if self.embeddings else True    # 是否更新词向量
        self.two_channel = two_channel    # 是否使用双通道
        
        # 词向量层
        self.emb = tf.keras.layers.Embedding(self.voc_size, self.emb_dim, weights=self.embeddings, trainable=self.train_emb)
        if self.two_channel:
            if not self.embeddings:
                raise    # 双通道时需要提供预训练的词向量
            # 双通道时，第二词向量层
            self.emb2 = tf.keras.layers.Embedding(self.voc_size, self.emb_dim, weights=self.embeddings, trainable=False)
            
        self.cnn_layers = []    # 卷积层list
        for kernel in self.kernels:
            # 根据卷积核宽度生成卷积层
            self.cnn_layers.append(
                tf.keras.layers.Conv2D(self.filters, (kernel, self.emb_dim), activation="relu")
                # 因为涉及到多通道的原因，所以用二维卷积，否则使用一维卷积更简单
                # tf.keras.layers.Conv1D(self.filters, kernel, activation="relu")
            )
        self.pool = tf.keras.layers.GlobalMaxPool1D()    # 最大池化层
        self.dropout = tf.keras.layers.Dropout(0.1)    # dropout层
        # 输出层，带L2正则
        self.output_layer = tf.keras.layers.Dense(self.output_dim, activation="softmax", kernel_regularizer="l2")

    def get_embedding(self, inputs):
        """
        获取词向量层的结果
        :param inputs: [batch, seq_len]
        :return: [batch, seq_len, emb_dim, channel]
        """
        emb = self.emb(inputs)    # [batch, seq_len, emb_dim]
        emb = tf.expand_dims(emb, 3)
        if self.two_channel:
            emb2 = self.emb2(inputs)
            emb2 = tf.expand_dims(emb2, 3)
            emb = tf.concat([emb, emb2], 3)
        return emb    # [batch, seq_len, emb_dim, channel]

    def call(self, inputs, training=None, mask=None):
        # inputs: [batch, seq_len]
        emb = self.get_embedding(inputs)    # [batch, seq_len, emb_dim, channel]
        cnn_feature = []
        for cnn in self.cnn_layers:
            f = cnn(emb)    # [batch, seq_len-, 1, filter]
            f = tf.squeeze(f, axis=2)    # [batch, seq_len-, filter]
            f = self.pool(f)    # [batch, filter]
            cnn_feature.append(f)
        x = tf.concat(cnn_feature, 1)    # [batch, filter*kernel_num]
        x = self.dropout(x)
        y = self.output_layer(x)    # [batch, output_dim]
        return y


    # def train_step(self, data):
    #     x, y = data    # data的结构取决于模型跟传给fit的数据结构
    #     with tf.GradientTape() as tape:
    #         y_pred = self.call(x)    # 前向计算
    #         # 计算loss
    #         loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #     # 计算梯度
    #     grads = tape.gradient(loss, self.trainable_variables)
    #     # 更新参数
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #     # 更新度量
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # 返回度量结果，度量中包括loss
    #     return {m.name: m.result() for m in self.metrics}
    #
    # def test_step(self, data):
    #     x, y = data  # data的结构取决于模型跟传给fit的数据结构
    #     y_pred = self(x)  # 前向计算
    #     self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #     # 更新度量
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # 返回度量结果，度量中包括loss
    #     return {m.name: m.result() for m in self.metrics}

    
