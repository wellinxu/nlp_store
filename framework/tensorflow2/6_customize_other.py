"""
tf2中自定义callback、loss、metrics
"""
import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


# 当loss值低于某个值的时候将学习率减半
class CustomLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, loss_value=0.5):
        super(CustomLearningRate, self).__init__()
        self.loss_value = loss_value
        self.already_set = False

    def on_epoch_end(self, epoch, logs=None):
        if self.already_set: return
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if logs.get("loss") < self.loss_value:
            self.already_set = True
            # 获取当前的学习率
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            # 将学习率减半
            lr /= 2
            # 设置新的学习率
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, lr))


# 使用LambdaCallback创建简单callback
def print_loss(epoch, logs):
    loss = logs.get("loss")
    print("\nEpoch %05d: Loss is %6.4f." % (epoch, loss))

print_loss_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=print_loss)


# 函数式均方误差loss
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


# 计算均方误差，但对预测值偏离0.5较多的进行惩罚
class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        # 额外参数，惩罚比例
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        # 均方误差
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        # 偏移0.5的惩罚
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


# 计算并正确分类的样本数
class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        # 创建并初始化状态
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 更新状态
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        # 计算并返回结果
        return self.true_positives

    def reset_states(self):
        # 重新初始化状态
        self.true_positives.assign(0.0)