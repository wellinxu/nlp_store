# coding=utf-8
"""
源码阅读，结合
查看，效果更好
"""
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
  """
  创建优化器训练操作optimizer training op.
  :param loss: loss值
  :param init_lr: int，初始学习率
  :param num_train_steps: int，训练总步数
  :param num_warmup_steps: int，预热步数
  :param use_tpu: bool，是否使用TPU
  :return: 训练中优化参数时需要进行的操作
  """
  global_step = tf.train.get_or_create_global_step()    # 获取当前步数

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)    # 学习率初始化值

  # 实现学习率的线性衰减
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # 实现线性warmup.
  # 如果global_step < num_warmup_steps
  # 那么学习率为`global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float    # 线性比例
    warmup_learning_rate = init_lr * warmup_percent_done    # 线性比例的学习率

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)    # 是否在预热阶段
    # 如果在预热阶段，学习率就为warmup_learning_rate，否则为learning_rate
    # 一般是先上升后下降，因为learning_rate本身会线性衰减
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # 推荐使用这个优化器来进行微调，因为预训练是用这个训练的
  # (要注意Adam中的变量m/v不是从init_checkpoint中加载来的)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:   # TPU时使用的优化器
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()    # 可训练的变量
  grads = tf.gradients(loss, tvars)    # 计算各变量对loss的梯度

  # 梯度裁剪，预训练就是这样的
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  # 获取参数优化当中需要的操作算子
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # 通常情况下global_step的更新应该在`apply_gradients`方法中进行.
  # 但是`AdamWeightDecayOptimizer`中并没有完成这一步 doesn't do this. But if you use
  # 如果使用其他优化器，可能需要移除这一步
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """包含修正L2正则化（权重衰减）的Adam优化器"""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """
    创建一个Adam权重衰减优化器
    :param learning_rate: float，学习率
    :param weight_decay_rate: float，权重衰减比率
    :param beta_1: float， 默认0.9，梯度一阶矩估计用的参数
    :param beta_2: float， 默认0.999，梯度二阶矩估计用的参数
    :param epsilon: float，防止程序计算除0
    :param exclude_from_weight_decay: list，不需要L2正则化（权重衰减）的参数名称
    :param name: str，优化器名称
    """
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """实现Adam梯度更新算法"""
    assignments = []
    # 循环各个变量与梯度
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)    # 获变量的名字

      # 获取梯度的一阶矩估计
      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      # 获取梯度的二阶矩估计
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # 梯度的一阶矩与二阶矩估计更新
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)   # 梯度下降的方向

      # 对于Adam算法来说，如果直接将权重的平方加到loss函数上，这不是正确的
      # L2正则化（权重衰减）方法，因为这样会影响Adam中m与v的计算。
      #
      # 我们需要一种不影响m/v参数的权重衰减方法，这里讲loss的优化与L2正则
      # 分开计算，先根据Adam算法计算针对loss需要更新的量，然后加上L2正则
      # 需要改变的量.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param     # L2正则化（权重衰减）

      update_with_lr = self.learning_rate * update    # 梯度下降的方向与量

      next_param = param - update_with_lr    # 完成梯度更新

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])    # 保存更新操作，参数更新，梯度的一阶和二阶矩估计更新
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """判断名为param_name的变量是否要使用L2权重衰减（L2正则化）"""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      # 判断变量名是否在不需要做权重衰减的名单里
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """从张量名字中获取变量名字"""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)   # 获取正则表达式中第一个括号中的内容
    return param_name
