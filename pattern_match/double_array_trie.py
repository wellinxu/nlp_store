from pattern_match.trie import Trie


class DoubleArrayTrie(object):
    def __init__(self, keys):
        keys.sort()
        self.keys = keys
        self.node = [0 for v in range(20000)]
        self.check = [0 for v in range(20000)]
        self.build()

    def build(self):
        trie = Trie(self.keys)
        self.node[0] = self.get_value(0, 0, trie.root)
        del trie

    def get_value(self, parent_value, parent_index, node):
        while True:
            flag = True
            for w in node.child_key:
                index = self.get_index(w)
                while len(self.node) < index + parent_value:
                    self.node.extend([0 for i in range(20000)])
                    self.check.extend([0 for i in range(20000)])
                if self.node[index + parent_value] != 0:
                    flag = False
                    break
            if flag:
                break
            parent_value += 1
        for w in node.child.keys():
            self.node[self.get_index(w) + parent_value] = self.get_value(
                1, self.get_index(w) + parent_value, node.get_child(w))
            if node.get_child(w).end:
                self.node[self.get_index(w) + parent_value] *= -1
            self.check[self.get_index(w) + parent_value] = parent_index
        return parent_value

    def match(self, text):
        match_pair = []
        start = parent_index = parent_value = 0
        for i in range(len(text)):
            w = text[i]
            w_index = self.get_index(w)
            if parent_value + w_index < len(self.node) and self.node[parent_value + w_index] != 0 and self.check[parent_value + w_index] == parent_index:
                if parent_index == 0:
                    start = i
                parent_index = parent_value + w_index
                parent_value = abs(self.node[parent_index])
                if self.node[parent_index] < 0:
                    match_pair.append((start, i + 1))
            elif parent_index > 0:
                i = start
                parent_index = parent_value = 0
        return match_pair

    def get_index(self, w):
        """
        0作为跟节点的index
        1作为深度的index
        :param w:
        :return:
        """
        return ord(w) + 1