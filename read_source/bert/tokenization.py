"""
源码阅读，结合https://note.youdao.com/ynoteshare1/index.html?id=05e273cc0d95b4fd7b437e7cec3ba8b0&type=note
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
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """检查大小写配置是否与文件名一致"""
  # 大小写格式是由使用者传入的，但这里没有显式地检查是否与checkpoint文件一致。
  # 大小写的信息应该被写入bert的配置文件中，但实际上没有，所以我们需要进行验证

  # 如果没有checkpoint文件或者文件名不是标准的，则直接返回
  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  # 这几个模型是要求小写的
  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def convert_to_unicode(text):
  """将文本转为Unicode格式（如果文本并没有转过），假设输入是utf-8格式."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """返回适应打印或者“tf.loggging”的方式的编码文本"""

  # 最终想要的是“str”格式的，但有可能是Unicode字符串的或者是字节字符串
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """加载词表"""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """根据词表，将标志/id转为id/标志."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  # 将标志转为id
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  # 将id转为标志
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """去除前后空格，并用空格分割文本"""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """端到端地进行标记"""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)    # 加载词表
    self.inv_vocab = {v: k for k, v in self.vocab.items()}    # 反向词表，key，value反转
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)    # 基础标记
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)    # 词片段标记

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      # 先将文本切分成一个词一个词的
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        # 再将每个词切分成词片段
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    # 将标志转为id
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    # 将id转为标志
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """进行基础标志（标点分割，小写化等等）"""

  def __init__(self, do_lower_case=True):
    """

    Args:
      do_lower_case: 是否小写化输入.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """标记一些列文本"""
    text = convert_to_unicode(text)   # 将文本转为Unicode格式
    text = self._clean_text(text)    # 删除处理空白字符与无效字符等

    # 这是2018年11月1号，为了多语言模型与中文模型添加的。现在也可以应用在英文模型上。
    text = self._tokenize_chinese_chars(text)    # 在中文字符两边加上空格

    orig_tokens = whitespace_tokenize(text)    # 使用空格切分后的结果
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)    # 去除文本中的重（zhong）音符号
      split_tokens.extend(self._run_split_on_punc(token))    # 根据标点符号分割文本

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """去除文本中的重（zhong）音符号"""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """根据标点符号分割文本"""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """在中日韩文字符两边加上空格"""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """判断是否是中日韩字符"""
    # 这里的中日韩字符并不是包含了所有的韩文与日文，因为有部分韩文与日文也是
    # 使用空格切分的，这部分不需要额外处理。
    # 相关定义如下：
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)

    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """删除无效字符并处理空白字符"""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab    # 词表
    self.unk_token = unk_token    # 未知词的标志
    self.max_input_chars_per_word = max_input_chars_per_word    # 被切分词的最大长度，超过则认为是未知词

  def tokenize(self, text):
    """将一段文本标记为词片段形式

    基于给定词表，使用贪心最长匹配优先算法进行标记

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: 单个词或者使用空格分割之后的词序，输入应该已经经过BasicTokenizer处理。

    Returns:
      词片段序列
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        # 单个词的字符数超过最大限度，则认为是未知词
        output_tokens.append(self.unk_token)
        continue

      is_bad = False    # 是否可以切分成词片段
      start = 0
      sub_tokens = []    # 切分后的词片段
      # 循环判断词的片段是否在给定的词表里
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            # 如果不是开头，在在词片段前加上“##”，为了方便后面进行全词mask
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          # 如果没有找到在给定词表中的词片段，则认为是坏词
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad: # 如果认为是坏词，则标志位未知
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """判断是否是空格字符"""
  # \t, \n, and \r也被认为是空格字符.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """判断是否是控制字符"""
  # 我们将"\t""\n""\r"控制字符视为空格字符
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """判断是否是标点符号字符"""
  cp = ord(char)
  # 将所有非数字与字母的都认为是标点，"^", "$"和 "`"也认为是
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False
