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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# 输入的行文件路径（或路径list）
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

# 输出的TF example文件路径（或路径list）
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

# 词表文件路径，BERT基于此进行训练
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# 字符是否要小写化
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

# 是否进行全词遮蔽，WWM
flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

# 允许的序列最大长度，默认128，BERT_BASE是512
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

# 被遮蔽token的最大数目，默认20
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")
# 生成随机数的种子
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

# 输入数据被重复使用的次数（每次会进行不同的mask）
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

# 被遮蔽的概率
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

# 生成比max_seq_length更小的句子的概率，为了减少预训练与微调时句子长度不一致的问题
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
  """单个训练实例（句子对）"""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens    # 替换遮蔽词token之后的所有token
    self.segment_ids = segment_ids    # 属于第一句还是第二句
    self.is_random_next = is_random_next    # 是否是随机下一句，“下一句预测”任务的标签
    self.masked_lm_positions = masked_lm_positions    # 被遮蔽的位置
    self.masked_lm_labels = masked_lm_labels    # 被遮蔽词的真实token

  def __str__(self):    # 重写str()方法
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """
  根据训练样本，写入TF样本文件
  :param instances: 训练样本list， [TrainingInstance]
  :param tokenizer: token切分类，FullTokenizer
  :param max_seq_length: 允许最大序列长度，int
  :param max_predictions_per_seq: 允许最大遮蔽数目，int
  :param output_files: TF exmaple输出文件路径， list
  :return:
  """
  writers = []
  for output_file in output_files:    # 文件会很大，所以会写入多个文件
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)    # token转为id
    input_mask = [1] * len(input_ids)    # 有实际输入的位置为1，没有实际输入的后续用0pad
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    # 将输入序列长度padding到“max_seq_length”，用0padding
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    # 将遮蔽序列长度padding到“max_predictions_per_seq”，用0padding
    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0    # 下一句预测的标签

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)    # 输入的id, [max_seq_length]
    features["input_mask"] = create_int_feature(input_mask)    # 输入的mask, [max_seq_length]
    features["segment_ids"] = create_int_feature(segment_ids)    # 第一句、第二句, [max_seq_length]
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)    # 语言模型中被遮蔽的位置, [max_predictions_per_seq]
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)    # 遮蔽语言模型的标签, [max_predictions_per_seq]
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)    # 遮蔽语言模型中被遮蔽的标签的权重, [max_predictions_per_seq]
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])   # 下一句预测的标签, [1]

    # 转为TF example
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())    # 写入文件
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:    # 展示前20个样本
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  # int类型数据转换
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  # float类型数据转换
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """
  从行文本数据中创建训练样本
  :param input_files:  输入行文件路径（或路径集合），[str]
  :param tokenizer: token切分类，FullTokenizer
  :param max_seq_length: 允许的最大序列长度，int
  :param dupe_factor: 数据重复使用次数，默认10次，int
  :param short_seq_prob: 生成短句子的概率，默认10%，float
  :param masked_lm_prob: 被遮蔽的概率，默认15%， float
  :param max_predictions_per_seq: 允许最多被遮蔽的token数目，int
  :param rng: 随机数生成器
  :return: 训练样本list，[TrainingInstance]
  """
  all_documents = [[]]

  # 输入文本格式：
  #（1）一句一行。需要是正真的一句，不能是一整个段落，也不能是文本的某个截断。
  # 因为在“下一句预测”任务中需要用到句子边界。
  # （2）文档之间用空行分割。这样“下一句预测”任务就不会跨越两个文档。

  # 读取文本数据并切分成token
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # 移除空文档并打乱顺序
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  # 多次重复数据，每次都会对数据进行不同的mask
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          # 根据单个文档生成训练样本
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)    # 打乱样本顺序
  return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """
  根据单个文档生成训练样本
  :param all_documents: 所有的文档list，用来随机抽取下一句使用的
  :param document_index: 当前文档的索引，int
  :param max_seq_length: 允许的最大序列长度，int
  :param short_seq_prob: 生成短句子的概率，默认10%，float
  :param masked_lm_prob: 被遮蔽的概率，默认15%， float
  :param max_predictions_per_seq: 允许最多被遮蔽的token数目，int
  :param vocab_words: 词表， list
  :param rng: 随机数生成器， Random
  :return: 单个训练样本，TrainingInstance
  """
  document = all_documents[document_index]

  #最大长度要给 [CLS]（开头）, [SEP]（两句之间）, [SEP]（结尾）留三个位置
  max_num_tokens = max_seq_length - 3

  # 大多数情况下，我们都会填满整个句子，直到“max_seq_length”长度，因为短文本太浪费算力了。
  # 但是，我们有时（short_seq_prob=10%的情况）需要用更短的句子来减少预训练与微调阶段之间的差距。
  # “target_seq_length”只是一个粗略的长度目标，而“max_seq_length”是固定的限制。
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:  # short_seq_prob=10%的情况下，句子长度会随机取一个小于最大长度的值
    target_seq_length = rng.randint(2, max_num_tokens)

  # 我们不会仅仅将文档中的tokens拼接到一个长句中，然后任意地选择一个切分点，因为这样会让“下一句预测”任
  # 务太简单。实际上，我们会根据用户输入，根据实际的句子将输入分成“A”句跟“B”句。
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        a_end = 1   # current_chunk中句子放到“A”句（第一句）中的数量
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # 下一句是否随机选择
        is_random_next = False
        # current_chunk中只有一句，或者50%的概率，下一句进行随机选择
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)    # “B”句的最大长度限制

          # 对大型语料来说，很少需要一次以上的迭代，但是为了小心起见，我们尽量保证
          # 随机的文档不是当前正在处理的文档。
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)    # 随机选择一句作为"B"句的开始
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # 随机选择下一句的时候，current_chunk中有一些句子是没有用到过的，
          # 为了不浪费语料，将这些句子“放回去”
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        else:    # 真的下一句
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)    # 将句子对截断到最大限制长度

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []    # 原始句子对token
        segment_ids = []    # 判断属于第一句还是第二句
        tokens.append("[CLS]")    # 添加开头token
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")     # 添加第一句结尾token
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")    # 添加结尾token
        segment_ids.append(1)

        # 根据遮蔽语言模型，生成预测目标
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        # 单个训练实例（句子对）
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


# 带名字的元组
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """
  根据遮蔽语言模型，生成预测目标
  :param tokens: 原始句子对token， list
  :param masked_lm_prob: 被遮蔽的概率，默认15%， float
  :param max_predictions_per_seq: 允许最多被遮蔽的token数目，int
  :param vocab_words: 词表，list
  :param rng: 随机数生成器，Random
  :return: 被遮蔽后的句子对output_tokens,list
           被遮蔽的位置masked_lm_positions, list
           被遮蔽的真实tokenmasked_lm_labels, list
  """

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # WWM全词遮蔽表示我们会遮蔽跟原词相关的所有词片段。当词被分层词片段，第一个token
    # 没有任何标记，后面的每一个token都会加上“##”前缀。所以一旦看见带##的token，我
    # 们会将它添加到前面一个词索引的集合里。

    # 要注意到，全词遮蔽并没有改变训练代码，我们依然需要独立地预测每一个词片段，
    # 需要计算整个词表的softmax。
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)    # 打乱候选索引集

  output_tokens = list(tokens)    # 被遮蔽后的句子对token

  # 被遮蔽数量
  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # 如果添加当前候选集后，遮蔽的数量超过了最大限制，则跳过此候选集
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False    # 判断token是否出现错
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue    # 如果token出现过，则跳过，为了尽量遮蔽不同的token
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80%会将token换成[MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10%会保持原词
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10%会替换成一个随机的词
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token    # 将遮蔽位置的词替换掉

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []    # 被遮蔽的位置
  masked_lm_labels = []    # 被遮蔽词的真实token
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """
  将句子对截断到最大限制长度
  :param tokens_a: “A”句token， list
  :param tokens_b: “B”句token，list
  :param max_num_tokens: 允许最大token数目， int
  :param rng: 随机数生成器，Random
  :return: 没有返回，原地修改
  """
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    # 截断长度较长的一句
    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # 为了避免偏差，50%的情况下我们丢弃开头的token，50%的情况下丢弃结尾的token
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # token切分类
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)    # 随机数生成器
  # 从行文本数据中创建训练样本
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  # 根据训练样本，写入TF样本文件
  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  # 必要字段
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
