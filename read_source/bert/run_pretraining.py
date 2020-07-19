"""
源码阅读，结合https://note.youdao.com/ynoteshare1/index.html?id=501cb88f2e31709c0eff701e5f9050ac&type=note
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## 必要参数Required parameters
# bert配置文件，描述了模型的结构
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

# 输入文件
flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

# 输出地址
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## 其他参数Other parameters
# 初始化时的checkpoint，通常是预训练好的bert模型
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

# 最大序列长度
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

# 每个序列最大预测数量
flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

# 是否进行训练
flags.DEFINE_bool("do_train", False, "Whether to run training.")

# 是否进行验证
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

# 训练btach size
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

# 验证batch size
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

# 学习率
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

# 训练步数
flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

# 预热步数
flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

# 多少步保存一次checkpoint
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

# TPU使用参数
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

# 验证的最大步数
flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

# 是否使用tpu
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# TPU使用参数
tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

# TPU使用参数
tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# TPU使用参数
tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# TPU使用参数
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# TPU使用参数
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """
  返回给TPUEstimator使用的模型函数-model_fn
  :param bert_config: bert配置，描述bert结构
  :param init_checkpoint: 初始化时的checkpoint
  :param learning_rate: 学习率
  :param num_train_steps: 训练步数
  :param num_warmup_steps: 预热步数
  :param use_tpu: 是否使用tpu
  :param use_one_hot_embeddings: 是否用one hot方式获取embedding，跟use_tpu一致，因为tpu上才使用这种方式
  :return: 模型函数，model_fn
  """

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """
    待返回的模型函数，model_fn
    :param features: 输入dict
    :param labels: 标签
    :param mode: 模式，训练还是 eval
    :param params:
    :return: 输出结果
    """

    # 记录特征名称与形状
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    # 获取各个输入
    input_ids = features["input_ids"]    # 输入的id张量, [batch_size, seq_len]
    input_mask = features["input_mask"]    # 输入的mask张量, [batch_size, seq_len]
    segment_ids = features["segment_ids"]    # 第一句、第二句, [batch_size, seq_len]
    masked_lm_positions = features["masked_lm_positions"]     # 语言模型中被遮蔽的位置, [batch_size, masked_len]
    masked_lm_ids = features["masked_lm_ids"]    # 遮蔽语言模型的标签, [batch_size, masked_len]
    masked_lm_weights = features["masked_lm_weights"]    # 遮蔽语言模型中被遮蔽的标签的权重, [batch_size, masked_len]
    next_sentence_labels = features["next_sentence_labels"]   # 下一句预测的标签, [batch_size]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)    # 是否训练模型

    # 获取bert模型，具体参考modeling.py文件
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # 下游任务，遮蔽语言模型的相关处理
    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    # 下游任务，下一句预测的相关处理
    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss    # 计算两个任务的整体loss计算

    tvars = tf.trainable_variables()    # 模型中可训练参数

    initialized_variable_names = {}    #被初始化的变量名字
    scaffold_fn = None
    # 用已有模型初始化参数
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:
        # tpu相关变量初始化
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        # 非tpu变量初始化
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 记录各变量名称与形状
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:    # 训练模式
      # 创建优化器
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      # 训练返回结果
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:    # 验证模式

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """
        度量函数，计算模型的loss与acc
        :param masked_lm_example_loss:  遮蔽语言模型的样本loss, [batch_size, masked_len]
        :param masked_lm_log_probs:  遮蔽语言模型的对数概率值, [batch_size*masked_len, voc_size]
        :param masked_lm_ids:  遮蔽语言模型的标签id, [batch_size, masked_len]
        :param masked_lm_weights: 遮蔽语言模型的标签权重, [batch_size, masked_len]
        :param next_sentence_example_loss:  下一句预测的样本loss, [batch_size]
        :param next_sentence_log_probs:  下一句预测的对数概率值, [batch_size, 2]
        :param next_sentence_labels:  下一句预测的标签, [batch_size]
        :return:
        """
        # 除最后一个维度，其他维度铺平, [batch_size*masked_len, voc_size]
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        # 获取概率最大的位置得到预测值, [batch_size*masked_len]
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])    # 铺平loss, [batch_size*masked_len]
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])    # 铺平, [batch_size*masked_len]
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])    # 铺平, [batch_size*masked_len]
        # 根据真实值与预测值计算acc
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        # 根据各个样本loss与权重计算整体loss
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        # 除最后一个维度，其他维度铺平, [batch_size, 2]
        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        # 获取概率最大的位置得到预测值, [batch_size]
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])    # 平铺, [batch_size]
        # 根据真实值与预测值计算acc
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        # 计算下一句预测的平均loss
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      # 验证时的度量函数与参数
      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      # 验证模式的输出结果
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec   # 返回输出结果

  return model_fn    # 返回模型函数


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """
  下游任务，遮蔽语言模型网络结构，获取遮蔽语言模型的loss与对数概率值
  :param bert_config: bert配置，描述bert网络结构
  :param input_tensor: 输入向量   [batch_size, seq_len, embedding_dim]
  :param output_weights: 输出权重，bert的词向量权重    [voc_size, embedding_dim]
  :param positions: 遮蔽的位置    [batch_size, masked_len]
  :param label_ids: 遮蔽的标签    [batch_size, masked_len]
  :param label_weights:  遮蔽的权重    [batch_size, masked_len]
  :return: 整体loss，每个样本的loss [batch_size*masked_len]，预测的对数概率值 [batch_size*masked_len, voc_size]
  """
  # 获取输入张量中对应位置的值, [batch_size * masked_len, embedding_dim]
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # 在输出层之前，还额外进行了一次非线性转换映射，预训练完之后，这个映射不再使用
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)   # 层标准化 [batch_size * masked_len, embedding_dim]

    # 输出权重跟输入的embedding很相似，只是输出多了个bias
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)    # [batch_size * masked_len, voc_size]
    logits = tf.nn.bias_add(logits, output_bias)    # [batch_size * masked_len, voc_size]
    log_probs = tf.nn.log_softmax(logits, axis=-1)    # [batch_size * masked_len, voc_size]

    label_ids = tf.reshape(label_ids, [-1])    # [batch_size * masked_len]
    label_weights = tf.reshape(label_weights, [-1])    # [batch_size * masked_len]

    # [batch_size * masked_len, voc_size]
    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # 如果预测的数量小于最大数量，会补齐至最大数量，补齐的weight值为0，其他为1
    # 每一个样本的loss，[batch_size * masked_len]
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    # 根据权重计算总loss
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator    #计算平均loss

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """
  下游任务，下一句预测网络结构，获取下一句预测的loss与对数概率值
  :param bert_config: bert配置，描述bert的模型结构
  :param input_tensor: 输入张量 [batch_size, embedding_dim]
  :param labels: 下一句预测标签，[batch_size]
  :return: 整体loss，每个样本的loss [batch_size]，预测的对数概率值 [batch_size, 2]
  """
  # 简单的二分类，0表示是下一句，1表示随机的依据，预训练之后不会再使用此层
  with tf.variable_scope("cls/seq_relationship"):
    # [2, embedding_dim]
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)    # [batch_size, 2]
    logits = tf.nn.bias_add(logits, output_bias)    # [batch_size, 2]
    log_probs = tf.nn.log_softmax(logits, axis=-1)    # [batch_size, 2]
    labels = tf.reshape(labels, [-1])    # [batch_size]
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)    # [batch_size, 2]
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)    # 计算每个样本loss,[batch_size]
    loss = tf.reduce_mean(per_example_loss)    # 计算平均loss
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """
  获取指定位置上的向量
  :param sequence_tensor: [batch_size, seq_length, embedding_dim]
  :param positions:  索引位置 [batch_size, masked_len]
  :return: [batch_size * masked_len, embedding_dim]
  """
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]    # embedding_dim

  # 铺平需要补充的位移
  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])    # [batch_size]
  # 铺平后的位置
  flat_positions = tf.reshape(positions + flat_offsets, [-1])    # [batch_size * masked_len]
  # [batch_size * masked_len, embedding_dim]
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)    # [batch_size * masked_len, embedding_dim]
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """
  创建给TPUEstimator使用的输入函数，input_fn
  :param input_files: list, 输入文件名称列表
  :param max_seq_length: int, 最大序列长度
  :param max_predictions_per_seq:  int,每个序列最大预测个数
  :param is_training: bool, 是否训练
  :param num_cpu_threads: int, cpu线程数,默认4
  :return: 输入函数，input_fn
  """

  def input_fn(params):
    """
    实际输入函数
    :param params: 参数字典
    :return:
    """
    batch_size = params["batch_size"]

    # tf record中特征的名称、维度及类型
    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # 对于训练模式，我们需要做大量的并行阅读与打乱样本
    # 对于验证模式，我们不需要打乱样本，并行阅读也不重要
    if is_training:    # 训练模式
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # 并行读取的文件数
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      d = d.repeat()

    # 使用tpu训练的时候必须丢弃掉最后不够组成一个batch的样本，因为tpu需要固定的维度
    # 验证的时候，我们都假设使用cpu或gpu，则不需要丢弃最后的少量样本
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d    # 输入函数的输出结果

  return input_fn


def _decode_record(record, name_to_features):
  """
  将tf record数据转为 example数据
  """
  example = tf.parse_single_example(record, name_to_features)

  # tpu只支持tf.int32,所以将所有的tf.int64转为tf.int32
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)    # 设置日志等级

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  # 解析配置文件，获取bert模型配置
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)   # 创建输出文件夹

  input_files = []   # 获取输入文件
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  # tpu相关
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # tpu相关
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  # 模型函数，输入到输出中间的结构定义
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # 如果tpu不可用，则会退化成cpu或者gpu版本
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  # 进行训练
  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    # 训练的输入函数，产生训练输入样本
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    # 根据输入函数与模型函数进行训练模型
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  # 进行评估
  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # 评估的输入函数，产生评估输入样本
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)
    # 根据输入函数与模型函数进行模型评估
    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    # 评估结果写入文件
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
  # 必要参数
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()    # 运行main(_)函数
