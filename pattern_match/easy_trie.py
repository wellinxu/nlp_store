#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:wellin
# datetime:2021/5/21 2:08 下午
# software: PyCharm

"""
参考：
https://mp.weixin.qq.com/s?__biz=MzIyNzMyODQwMQ==&mid=2247483669&idx=1&sn=d31fb4f6b0689c42a4d1f9f8c9fad14f&chksm=e863a9ebdf1420fd8b903acc22db67b14a33bbfc337592ac1887f863941b543bb960d018bdd2&token=411288412&lang=zh_CN#rd
https://zhuanlan.zhihu.com/p/75733209

trie树的相关实现，Trie,双数组Trie数，AC自动机
is_suffix：是否构建后缀树, case_sensitive：是否大小写敏感
search：搜索文本中所有字典中存在的词，包括有覆盖的词
match：正向最长匹配（后缀树就是后向最长匹配）
prefix_search:是否是词典中某个词的前缀（后缀数就是判断是否是后缀）

todo 双数组实现时，给每个key添加一个尾字符，ac搜索的时候能够直接获取字符长度
"""


class Node(object):
    """树节点"""
    def __init__(self, char, value=None):
        self.char = char
        self.value = value    # 只有叶子节点才有value
        self.child = {}
        self.end = False

    def is_root(self):
        return self.char is None

    def has_child(self, char: str):
        return char in self.child.keys()

    def get_child(self, char: str):
        return self.child.get(char, None)

    def add_child(self, char: str, value=None):
        if not self.has_child(char):
            self.child[char] = Node(char, value)


class BaseTrie(object):
    """各个Trie树实现的基类"""
    def __init__(self, keys, values, is_suffix=False, case_sensitive=False):
        self.keys = keys    # keys不重复
        self.values = values    # todo 同key不同value目前不支持
        self.root = Node(None)
        self.is_suffix = is_suffix   # 是否构建后缀数
        self.case_sensitive = case_sensitive    # 大小写是否敏感，默认不敏感
        self.build()

    def build(self):
        raise NotImplementedError

    def get_word(self, key: str):
        if not self.case_sensitive:
            key = key.lower()    # 不敏感就全部改为小写
        if self.is_suffix:
            key = key[::-1]
        return key

    def get_match_pair(self, match_pair, tlen):
        if self.is_suffix:
            match_pair = [(tlen - b, tlen - a) for a, b in match_pair]
            match_pair.sort()
        return match_pair

    def search(self, text: str):
        """搜索text中所有存在字典中的词，包括折叠覆盖"""
        match_pair = []
        text = self.get_word(text)
        tlen = len(text)
        for start, w in enumerate(text):
            self.start_search(start, tlen, text, match_pair)
        return self.get_match_pair(match_pair, tlen)

    def start_search(self, start, tlen, text, match_pair):
        raise NotImplementedError

    def match(self, text: str):
        """前向最长匹配, is_suffix为true时，反向最长匹配"""
        match_pair = []
        text = self.get_word(text)
        tlen = len(text)
        for start, w in enumerate(text):
            self.start_match(start, tlen, text, match_pair)
        return self.get_match_pair(match_pair, tlen)

    def start_match(self, start, tlen, text, match_pair):
        raise NotImplementedError


class Trie(BaseTrie):
    """Trie树"""

    def build(self):
        for key, value in zip(self.keys, self.values):
            tem_node = self.root
            for char in self.get_word(key):
                tem_node.add_child(char)
                tem_node = tem_node.get_child(char)
            tem_node.end = True
            tem_node.value = value

    def start_search(self, start, tlen, text, match_pair):
        node = self.root
        for i in range(start, tlen):
            w = text[i]
            if node.has_child(w):  # 此时node是前一个节点
                node = node.get_child(w)
                if node.end:  # 此时node是当前节点
                    match_pair.append((start, i + 1))
            else:
                return

    def start_match(self, start, tlen, text, match_pair):
        if match_pair and start < match_pair[-1][1]:
            return
        node = self.root
        end = start
        for i in range(start, tlen):
            w = text[i]
            if node.has_child(w):  # 此时node是前一个节点
                node = node.get_child(w)
                if node.end:  # 此时node是当前节点
                    end = i + 1
            else:
                break
        if end > start:
            match_pair.append((start, end))


class DoubleArrayTrie(BaseTrie):
    """双数组Trie树"""
    def __init__(self, keys, values, is_suffix=False, case_sensitive=False):
        # keys.sort()    # todo 判断keys是按照顺序排序的
        self.char_map = {}    # 保存每个char的索引，ord的值太大，浪费空间
        self.check_key(keys)    # todo 这个方法，后缀报错
        self.node_list = [0] * 20000
        self.check_list = [-1] * 20000
        super(DoubleArrayTrie, self).__init__(keys, values, is_suffix, case_sensitive)

    def check_key(self, keys):
        for i in range(1, len(keys)):
            if keys[i-1] > keys[i]:
                raise Exception('keys的指需要由小到大排序，keys shoule be sotred')
        key_set = set()
        for key in keys:
            for char in key:
                key_set.add(char)
        key_list = list(key_set)
        key_list.sort()
        for char in key_list:
            self.char_map[char] = len(self.char_map) + 2

    def build(self):
        self.keys = [self.get_word(key) for key in self.keys]
        trie = Trie(self.keys, self.keys)
        self.node_list[0] = self.get_value(0, 0, trie.root)
        del trie

    def get_value(self, parent_value, parent_index, node):
        while True:
            flag = True
            for w in node.child.keys():
                index = self.get_index(w)
                while len(self.node_list) < index + parent_value:
                    self.node_list.extend([0] * 20000)
                    self.check_list.extend([-1] * 20000)
                if self.node_list[index + parent_value] != 0:
                    flag = False
                    break
            if flag:
                break
            parent_value += 1
        for w in node.child.keys():
            self.node_list[self.get_index(w) + parent_value] = self.get_value(
                1, self.get_index(w) + parent_value, node.get_child(w))
            if node.get_child(w).end:
                self.node_list[self.get_index(w) + parent_value] *= -1
            self.check_list[self.get_index(w) + parent_value] = parent_index
        return parent_value

    def get_child_index(self, parent_index, w_index):
        parent_value = abs(self.node_list[parent_index])
        if parent_value + w_index >= len(self.node_list):
            return -1
        if self.node_list[parent_value + w_index] != 0 and self.check_list[parent_value + w_index] == parent_index:
            return parent_value + w_index
        return -1

    def start_search(self, start, tlen, text, match_pair):
        parent_index = 0
        for i in range(start, tlen):
            w_index = self.get_index(text[i])
            parent_index = self.get_child_index(parent_index, w_index)
            if parent_index > -1:
                if self.node_list[parent_index] < 0:
                    match_pair.append((start, i + 1))
            else:
                return

    def start_match(self, start, tlen, text, match_pair):
        if match_pair and start < match_pair[-1][1]:
            return
        parent_index = 0
        end = start
        for i in range(start, tlen):
            w_index = self.get_index(text[i])
            parent_index = self.get_child_index(parent_index, w_index)
            if parent_index > -1:
                if self.node_list[parent_index] < 0:
                    end = i + 1
            else:
                break
        if end > start:
            match_pair.append((start, end))

    def get_index(self, w):
        """ 0作为根节点的index 1作为深度的index """
        return ord(w) + 1
        # return self.char_map.get(w, 1)


class AhoCorasickAutoMation(DoubleArrayTrie):
    """AC自动机"""
    def __init__(self, keys, values, is_suffix=False, case_sensitive=False):
        super(AhoCorasickAutoMation, self).__init__(keys, values, is_suffix, case_sensitive)
        self.next_list = [0 for v in self.check_list]
        self.build_next()

    def build_next(self):
        for key in self.keys:
            for i in range(1, len(key)):
                key_index = self.prefix_search(key[:i+1])
                for j in range(1, i+1):
                    tem_index = self.prefix_search(key[j:i+1])
                    if tem_index > 0:
                        self.next_list[key_index] = tem_index
                        break

    def prefix_search(self, key):
        parent_index = 0
        for i in range(len(key)):
            w_index = self.get_index(key[i])
            parent_index = self.get_child_index(parent_index, w_index)
            if parent_index < 0:
                return -1
        return parent_index

    def find_depth(self, index):
        if index > len(self.check_list): return -1
        depth = 0
        while index > 0:
            depth += 1
            index = self.check_list[index]
        if index == 0: return depth
        return -1

    def search(self, text: str):
        """搜索text中所有存在字典中的词，包括折叠覆盖"""
        match_pair = []
        text = self.get_word(text)
        parent_index = 0
        for i, wi in enumerate(text):
            w_index = self.get_index(text[i])
            current_index = self.get_child_index(parent_index, w_index)
            while current_index == -1 and parent_index > 0:
                parent_index = self.next_list[parent_index]
                if self.node_list[parent_index] < 0:
                    start = i - self.find_depth(parent_index)
                    match_pair.append((start, i))
                current_index = self.get_child_index(parent_index, w_index)
            if current_index > -1:
                parent_index = current_index
                if self.node_list[current_index] < 0:
                    start = i + 1 - self.find_depth(current_index)
                    match_pair.append((start, i + 1))
        return self.get_match_pair(match_pair, len(text))

    def match(self, text: str):
        """前向最长匹配, is_suffix为true时，反向最长匹配"""
        match_pair = []
        text = self.get_word(text)
        parent_index = 0
        for i, wi in enumerate(text):
            w_index = self.get_index(text[i])
            current_index = self.get_child_index(parent_index, w_index)
            while current_index == -1 and parent_index > 0:
                parent_index = self.next_list[parent_index]
                if self.node_list[parent_index] < 0:
                    start = i - self.find_depth(parent_index)
                    if match_pair and match_pair[-1][0] == start:
                        match_pair.pop()
                    if not match_pair or match_pair[-1][1] <= start:
                        match_pair.append((start, i))
                current_index = self.get_child_index(parent_index, w_index)
            if current_index > -1:
                parent_index = current_index
                if self.node_list[current_index] < 0:
                    start = i + 1 - self.find_depth(current_index)
                    if match_pair and match_pair[-1][0] == start:
                        match_pair.pop()
                    if not match_pair or match_pair[-1][1] <= start:
                        match_pair.append((start, i + 1))
        return self.get_match_pair(match_pair, len(text))



chars = "abcdefghijklmnopqrstuvwxyz"
def create_char(length):
    word = ""
    for i in range(length):
        word += random.choice(chars)
    return word

def test():
    keys = []
    for i in range(1000):
        keys.append(create_char(random.choice([3, 4, 5, 6, 7, 8, 9])))
    keys.sort()
    texts = []
    for i in range(100000):
        texts.append(create_char(random.choice([i in range(60, 100)])))

    # trie = Trie(keys, keys, True)  # Current memory usage is 0.001472MB; Peak was 0.001802MB
    trie = Trie(keys, keys, False)  # Current memory usage is 0.001472MB; Peak was 0.001802MB
    # dtrie = DoubleArrayTrie(keys, keys, True)
    dtrie = DoubleArrayTrie(keys, keys, False)    # Current memory usage is 0.000504MB; Peak was 0.000834MB
    # atrie = AhoCorasickAutoMation(keys, keys, True)
    atrie = AhoCorasickAutoMation(keys, keys, False)  # Current memory usage is 0.00052MB; Peak was 0.000782MB
    print("trie size:", sys.getsizeof(trie))
    print("dtrie size:", sys.getsizeof(dtrie))
    print("atrie size:", sys.getsizeof(atrie))
    for text in texts:
        match_pair = trie.search(text)
        dmatch_pair = dtrie.search(text)
        amatch_pair = atrie.search(text)
        if str(match_pair) == str(dmatch_pair) == str(amatch_pair):
            continue
        print("text:\t", text)
        print("trie result:")
        for i, j in match_pair:
            print(text[i:j])
        print("dtrie result:")
        for i, j in dmatch_pair:
            print(text[i:j])
        print("atrie result:")
        for i, j in amatch_pair:
            print(text[i:j])
        print()
    # match_pair = trie.search(text)
    # for i, j in match_pair:
    #     print(text[i:j])


if __name__ == '__main__':
    import sys
    import random
    keys = ["南京大学", "南理工", "南京", "南京大", "大南京"]
    keys.sort()

    test()

    # trie = Trie(keys, keys, True)
    # trie = Trie(keys, keys, False)    # Current memory usage is 0.001472MB; Peak was 0.001802MB
    # trie = DoubleArrayTrie(keys, keys, True)
    # trie = DoubleArrayTrie(keys, keys, False)    # Current memory usage is 0.000504MB; Peak was 0.000834MB
    # trie = AhoCorasickAutoMation(keys, keys, False)    # Current memory usage is 0.00052MB; Peak was 0.000782MB
    # trie = AhoCorasickAutoMation(keys, keys, True)
    # text = "大南京大学跟南理工"
    # match_pair = trie.match(text)
    # match_pair = trie.search(text)
    # for i, j in match_pair:
    #     print(text[i:j])