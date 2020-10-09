"""
源码阅读，结合
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
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re

import modeling
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# 输入文件路径
flags.DEFINE_string("input_file", None, "")

# 特征保存路径
flags.DEFINE_string("output_file", None, "")

# 选取特征层数，用“,”分割
flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

# bert配置文件路径
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

# 最大长度限制，默认128
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

# 预训练模型路径
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

# 词表路径
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# 是否进行小写
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

# batch size大小
flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

# 是否使用TPU
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# TPU相关参数
flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

# TPU相关参数
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# 是否使用one_hot_embedddings
flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """单个数据特征"""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
  """
  返回Estimator使用的input_fn函数
  :param features: 特征list
  :param seq_length: 最大长度
  :return: input_fn
  """

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    batch_size = params["batch_size"]

    num_examples = len(features)

    # 这是演示使用，如果要处理大量数据，需要用TFRecordReader来加载数据
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """
  返回Estimator的模型函数
  :param bert_config: bert配置
  :param init_checkpoint: bert模型文件
  :param layer_indexes: 返回的层数
  :param use_tpu: 是否使用TPU,具体见modeling.BertModel中相关参数
  :param use_one_hot_embeddings:是否使用one_hot_embedding,具体见modeling.BertModel中相关参数
  :return: model_fn
  """

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """Estimator的模型函数"""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    # 根据配置获取模型
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    # 获取当前变量与初始化checkpoint变量中的交集
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    # 从模型文件中初始化参数
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    all_layers = model.get_all_encoder_layers()    # bert模型所有的encode层

    predictions = {
        "unique_id": unique_ids,
    }

    # 返回指定的层数作为特征
    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """
  将样本转为特征
  :param examples:样本，list
  :param seq_length: 最大长度，int
  :param tokenizer: FullTokenizer
  :return: [InputFeatures]
  """

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    # 将句子对截断到最大限制长度
    if tokens_b:
      # 为 [CLS], [SEP], [SEP] 留3个位置
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # 为 [CLS], [SEP], [SEP] 留2个位置
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # BERT中句子的约定是:
    # (a)对于句子对:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b)对于单句:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # type_ids表示是第一句还是第二句，这个其实不是必要的，因为有[SEP]分割句子，但这可以让模型更容易学到句子的概念
    #
    # 对于分类任务，第一个字符（[CLS]）可以看做句向量，前提是整个模型进过了微调
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)    # token转id

    # 真实的token用1表示，padding的token用0表示
    input_mask = [1] * len(input_ids)

    # 用0padding到最大长度
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:    # 显示部分样本
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """
  将句子对截断到最大限制长度，每次截取较长的句子
  :param tokens_a: a句
  :param tokens_b: b句
  :param max_length: 最大长度
  :return:
  """
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file):
  """
  读取输入文件中的样本，默认以“|||”切分a句与b句
  :param input_file: [InputExample]
  :return:
  """
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:    # 只有一句
        text_a = line
      else:    # 以“|||”切分a句与b句
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
  return examples


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)    # 设置打印日志级别

  layer_indexes = [int(x) for x in FLAGS.layers.split(",")]     # 返回哪几层特征

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)    # 获取bert配置

  # 初始化token切分器
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2    # tpu相关参数
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  examples = read_examples(FLAGS.input_file)    # 读取输入文件中的样本

  # 将样本转为特征
  features = convert_examples_to_features(
      examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

  # Estimator的模型函数
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      layer_indexes=layer_indexes,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # 如果TPU不可用，则会退化为CPU或GPU
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size)

  # input_fn函数
  input_fn = input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length)

  with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
                                               "w")) as writer:
    # 生成特征并写入文件
    for result in estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result["unique_id"])
      feature = unique_id_to_feature[unique_id]
      output_json = collections.OrderedDict()
      output_json["linex_index"] = unique_id
      all_features = []
      for (i, token) in enumerate(feature.tokens):
        all_layers = []
        for (j, layer_index) in enumerate(layer_indexes):
          layer_output = result["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
              round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]
          all_layers.append(layers)
        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers
        all_features.append(features)
      output_json["features"] = all_features
      writer.write(json.dumps(output_json) + "\n")


if __name__ == "__main__":
  # 必要参数
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()
