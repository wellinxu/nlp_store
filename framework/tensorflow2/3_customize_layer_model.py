"""
tf2 自定义层
"""
import tensorflow as tf


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        # add_weight与下面注释部分是等价的，只是更方便
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer=tf.random_normal_initializer,
                                 trainable=True)
        self.b = self.add_weight(shape=(units, ),
                                 initializer=tf.zeros_initializer,
                                 trainable=True)
        # w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
        #                                           dtype=tf.float32),
        #                      trainable=True)
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(initial_value=b_init(shape=(units,),
        #                                           dtype=tf.float32),
        #                      trainable=True)

    def call(self, inputs, training=None, mask=None):
        # if training:
        #     inputs = tf.nn.dropout(inputs, rate=0.9)
        return tf.matmul(inputs, self.w) + self.b


# 动态创建权重，在知道输入形状之后再创建权重
class Linear2(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear2, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=tf.random_normal_initializer,
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units, ),
                                 initializer=tf.zeros_initializer,
                                 trainable=True)

    def call(self, inputs, training=None, mask=None):
        self.add_loss(0.5 * tf.reduce_sum(tf.square(self.w)))    # L2正则
        # self.add_metric()
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


# 层之间的递归组合
class MLPBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 =  Linear2(2)
        self.linear_2 =  Linear2(2)
        self.linear_3 =  Linear2(1)

    def call(self, inputs, **kwargs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


# 自定义模型，模型构建可以使用sequential、subclass、functional，具体可以参考0_start.py文件

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


def create_model_by_functional():
    # 使用函数式api构建模型
    inputs = tf.keras.Input(shape=(256,))
    emb = tf.keras.layers.Embedding(10000, 16)(inputs)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(emb)
    d1 = tf.keras.layers.Dense(16, activation="relu")(avg_pool)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(d1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# 测试
x = tf.ones((2, 2))
# linear_layer = Linear(4, 2)
# linear_layer = Linear2(4)
linear_layer = MLPBlock()
y = linear_layer(x)
print(y)

layer = Linear2(64)
config = layer.get_config()
print(config)
new_layer = Linear2.from_config(config)