"""
源码阅读，结合https://note.youdao.com/ynoteshare1/index.html?id=b81b64e5f174ac03cc2bc37b5a72484d&type=note
查看，效果更好
"""
# coding=utf-8
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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class BertConfig(object):
  """BERT模型的配置参数"""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """

    Args:
      vocab_size: int，bert词表大小.
      hidden_size: int，encoder层与poller层输出维度,以及词向量维度.
      num_hidden_layers: int，Transformer中隐藏层的大小.
      num_attention_heads: int，Transformer中attention的头数.
      intermediate_size: int，每层中间前向传播时的维度
      hidden_act:sring， 非线性激活函数名称. 每层中间前向传播时使用，原注释说pooler层也使用，但代码里poller层固定使用的tanh激活函数
      hidden_dropout_prob: float， embeddings, encoder和pooler层的丢弃概率.
      attention_probs_dropout_prob:float， attention得分概率丢弃的比例
      max_position_embeddings: int，序列允许的最大长度
      type_vocab_size: int， token类型数量
      initializer_range: float,初始化参数的方差范围
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  示例:

  ```python
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """BERT模型的构造

    Args:
      config: `BertConfig` instance.
      is_training: bool. 是不是train模式，控制是否使用dropout
      input_ids: int32张量，形状为[batch_size, seq_length].
      input_mask: (可选) int32张量，形状为[batch_size, seq_length]
      token_type_ids: (可选) int32张量，形状为[batch_size, seq_length],类型ids，bert里面0表示第一句，1表示第二句
      use_one_hot_embeddings: (可选) bool. 获取词向量时，使用one-hot方式还是tf.gather()方式
      scope: (可选) 变量范围，默认"bert".

    Raises:
      ValueError: 配置无效或者某个输入张量维度无效
    """
    config = copy.deepcopy(config)
    if not is_training:    # 非训练模式时，不进行dropout
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    # 获取输入形状，batch_size大小与seq_length大小
    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    # input_mask或token_type_ids为空时，生成默认结果
    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # 获取词向量
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # 加上位置向量与token类型向量，然后进行层标准化与dropout
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # 将维度位[batch_size, seq_length]的2维mask张量转化成维度为[batch_size, seq_length, seq_length]的3维mask张量
        # 在计算attention得分时使用
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # 构建叠层transformer.
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]    #  [batch_size, seq_length, hidden_size]
      # pooler层将序列编码张量的形状[batch_size, seq_length, hidden_size]转换为[batch_size, hidden_size]
      # 这对句子级别或者句子对级别的分类很有作用，因为那时候需要一个固定维度的向量来表示句子
      with tf.variable_scope("pooler"):
        # 简单使用第一个token的隐藏状态来计算
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        # pooled_output:[batch_size, hidden_size]
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    # 获取pooler层结果
    return self.pooled_output

  def get_sequence_output(self):
    """获取最后一层encoder结果.

    Returns:
      float张量，形状为[batch_size, seq_length, hidden_size]
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    # 获取所有encoder层结果
    return self.all_encoder_layers

  def get_embedding_output(self):
    """获取embedding层结果

    Returns:
      float张量，形状为[batch_size, seq_length, hidden_size] 词向量、位置向量、token类型向量求和后，
      再进行层标准化与dropout的结果。是encoder层的输入
    """
    return self.embedding_output

  def get_embedding_table(self):
    # 获取词向量词表
    return self.embedding_table


def gelu(x):
  """
  高斯误差线性单元
  原论文: https://arxiv.org/abs/1606.08415
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """根据激活函数名称获取激活函数"""
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """本文件中没有被使用Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """执行dropout.
  Args:
    input_tensor: float张量
    dropout_prob: float. 丢弃的概率
  Returns:
    执行过dropout后的张量
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """对张量的最后一维进行层标准化"""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """进行层标准化再进行dropout"""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """根据给定范围创建截尾正态分布初始器"""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """根据id张量查询词向量

  Args:
    input_ids: id张量，整型，维度为[batch_size, seq_length]
    vocab_size: int型，词汇量大小.
    embedding_size: int型，词向量维.
    initializer_range: float型，词向量初始化范围
    word_embedding_name: 字符串，词向量表格名称
    use_one_hot_embeddings: 布尔型，True时，使用one-hot方式获取词向量，否则使用`tf.gather()`.

  Returns:
    float型张量，维度为[batch_size, seq_length, embedding_size].
  """
  # 假设输入维度为[batch_size, seq_length,num_inputs].
  # 如果输入维度是[batch_size, seq_length], 则改为[batch_size, seq_length, 1].
  # 如果只是bert模型使用，这一步其实没有必要的
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    # 使用one-hot的方法获取词向量，one-hot乘词表,词表较小时，这种方式更快
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    # 使用tf.gather()获取词向量
    output = tf.gather(embedding_table, flat_input_ids)

  input_shape = get_shape_list(input_ids)    # [batch_size, seq_length, 1].

  #[batch_size, seq_length, embedding_size].
  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """对词向量进行后处理

  Args:
    input_tensor: float张量，形状为[batch_size, seq_length,embedding_size]，词向量
    use_token_type: bool. 是否添加token的类型向量
    token_type_ids: (可选) int32张量，形状为[batch_size, seq_length]，use_token_type为True时必要有
    token_type_vocab_size: int. token类型的数量
    token_type_embedding_name: string. token的类型向量表的名字
    use_position_embeddings: bool. 是否添加位置向量
    position_embedding_name: string. 位置向量表的名字
    initializer_range: float. 初始化的范围参数
    max_position_embeddings: int. 位置向量的最大长度，只能比输入序列更长
    dropout_prob: float. 最后输出的丢弃概率

  Returns:
    跟输入维度一致的float张量

  Raises:
    ValueError: 张量形状或者输入值无效
  """
  # 获取输入张量维度，batch_size,seq_length,width(词向量的维度)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor    # 初始化输出张量

  if use_token_type:
    # 加上token类型向量
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    # token类型向量表变量
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # 因为类型词表总是很小，所以直接使用one-hot的方式获取向量，因为这种方式在小词表时总是更快
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings    # 直接将token类型向量加到输出上

  if use_position_embeddings:
    # 加上位置向量
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      # 位置向量表变量
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # full_position_embeddings已经建立了0到max_position_embeddings-1位置上的向量，
      # 为了获取0到seq_length-1位置上的向量，只要使用slice操作即可
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # 只有最后两维是相关的(`seq_length` and `width`), 所以只要广播开始的维度，通常是batch_size的维度
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])    # position_broadcast_shape=[1, seq_length, width]
      position_embeddings = tf.reshape(position_embeddings,position_broadcast_shape)    # [1, seq_length, width]
      output += position_embeddings    # 直接将位置向量加到输出上

  # 先进行层标准化再dropout
  output = layer_norm_and_dropout(output, dropout_prob)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """将维度位[batch_size, seq_length]的2维mask张量转化成维度为[batch_size, seq_length, seq_length]的3维mask张量

  Args:
    from_tensor: 形状为[batch_size, from_seq_length, ...]的张量
    to_mask: int32张量，形状为[batch_size, to_seq_length].

  Returns:
    float张量，形状为[batch_size, from_seq_length, to_seq_length].
  """
  # 获取from_shape维度，输入的秩职能是2或3，得到batch_size,from_seq_length
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  # 获取to_mask维度，输入的秩职能是2或3，得到to_seq_length
  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  # 将to_mask转为float32型，并reshape为[batch_size, 1, to_seq_length]
  to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # 创建形状为[batch_size, from_seq_length, 1]的全1张量
  broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  mask = broadcast_ones * to_mask    # [batch_size, from_seq_length, to_seq_length]

  return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """根据from_tensor与to_tensor执行多头attention

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  事实上，多头attention是通过转置与改变维度来处理的，而不是真的切分张量，因为这样会快得多

  Args:
    from_tensor: float张量，维度为[batch_size, from_seq_length, from_width].
    to_tensor: float张量，维度为[batch_size, to_seq_length, to_width].
    attention_mask: (可选) int32张量，维度为[batch_size,from_seq_length, to_seq_length]. 值是1或0.
    num_attention_heads: int. attention头的数量.
    size_per_head: int. 每个attention头的维度.
    query_act: (可选) query映射时的激活函数.
    key_act: (可选) key 映射时的激活函数.
    value_act: (可选) alue映射时的激活函数.
    attention_probs_dropout_prob: (可选) float. attention概率的丢弃比例.
    initializer_range: float. 初始化参数范围
    do_return_2d_tensor: bool. True时，返回的形状为[batch_size
      * from_seq_length, num_attention_heads * size_per_head]，否则返回的形状为[batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (可选) int. 如果输入是两维的，需要提供
    from_seq_length: (可选) int. 如果输入是两维的，需要提供
    to_seq_length: (可选) int. 如果输入是两维的，需要提供

  Returns:
    float张量，维度为[batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (如果`do_return_2d_tensor`是
      true, 则形状为[batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: 参数或者张量形状无效
  """
  # reshape张量后，转换第二维与第三维
  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  # 获取batch_size， from_seq_length， to_seq_length
  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # 各个字母维度含义:
  #   B = batch size
  #   F = `from_tensor`的序列长度
  #   T = `to_tensor`的序列长度
  #   N = attention头数
  #   H = 每个attention头数的隐藏层

  from_tensor_2d = reshape_to_matrix(from_tensor)    # [B*F, embedding_size]
  to_tensor_2d = reshape_to_matrix(to_tensor)    # [B*T, embedding_size]

  # `query_layer` = [B*F, N*H],将from_tensor映射为query
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H],将to_tensor映射为key
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H],将to_tensor映射为value
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # 将query与key进行点积，得到attention得分.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  # 根据维度进行缩放，将方差校正为1
  # query_layer与key_layer都服从0均值1方差的标准正太分布，点积后服从0均值size_per_head方差的标准正太分布
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # attention_mask为1是需要被注意到，为0是不要被注意到，创建一个张量，需要被主要到时等于0，不需要时等于-10000
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # 在softmax之前，将adder加到attention得分中，可以高效地处理不需要关注的位置
    attention_scores += adder

  # 计算attention概率
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # attention_probs与value点积， `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)    # attention结果

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]，返回2d模式
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """根据"Attention is All You Need"的多头，多层Transformer,几乎跟原实现一样

  原论文:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float张量，形状为[batch_size, seq_length, hidden_size].
    attention_mask: (可选) int32张量，形状为[batch_size, seq_length,seq_length], 1表示可以被关注，0表示不可以被关注
    hidden_size: int.Transformer的隐藏层大小.
    num_hidden_layers: int. Transformer的层数.
    num_attention_heads: int. Transformer中attention的头数.
    intermediate_size: int. 每层中间前向传播时的维度
    intermediate_act_fn: 非线性激活函数. 每层中间前向传播时使用
    hidden_dropout_prob: float. 每层结果丢弃的概率
    attention_probs_dropout_prob: float. attention得分概率丢弃的比例
    initializer_range: float. 初始化的范围
    do_return_all_layers: 返回所有层还是只返回最后一层

  Returns:
    float张量，形状为[batch_size, seq_length, hidden_size], Transformer最后一层结果.
    如果do_return_all_layers是True，则返回list包含每一层结果

  Raises:
    ValueError: 张量维度或参数无效
  """
  # 隐藏层维度必须是attention头数的倍数
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)    # 每一个attention头的维度
  # 获取输入张量的维度，batch_size, seq_length, input_width
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer需要进行残差相加，所以input_width与hidden_size需要一致
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # 将3d张量转为2d张量，因为TPU上reshape方法会消耗资源，事先转换好减少后续reshape操作
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []    # 所有层的结果
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output    # 每层输入

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          # 计算多头self-attention
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # 如果多头attention是切分后分开计算的，直接拼接在一起
          attention_output = tf.concat(attention_heads, axis=-1)

        # 进行线性映射后加上残差`layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # 进行中间的全连接处理
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # 在降维映射后加上残差attention_output
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)    # 每层输出
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)    # [batch_size, seq_length, hidden_size]
      final_outputs.append(final_output)
    return final_outputs    # 所有层的结果
  else:
    # 转为3d张量，[batch_size, seq_length, hidden_size]
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
  """获取张量的维度形状

  Args:
    tensor: 待获取维度形状的张量
    expected_rank: (可选参数) 整数，期望的秩
    name: 可选参数，张量的名字

  Returns:
    张量维度形状组成的list
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """将张量转为2为矩阵，最后一维保持不变"""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """将矩阵转为原维度的张量"""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """校验张量的秩是不是期望的秩，如果不是则报错。

  Args:
    tensor: 待校验的tf张量
    expected_rank: 期望的张量，整数或者整数list
    name: 张量的名字

  Raises:
    ValueError: 如果期望的维度跟实际维度不一致
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
