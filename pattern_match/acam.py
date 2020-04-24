
from pattern_match.double_array_trie import DoubleArrayTrie


class AhoCorasickAutoMation(DoubleArrayTrie):
    def __init__(self, keys):
        super(AhoCorasickAutoMation, self).__init__(keys)
        self.next = [0 for v in self.check]
        self.build_next()

    def build_next(self):
        for key in self.keys:
            key_index = self.prefix_search(key)
            if len(key) < 2:
                self.next[key_index] = 0
                continue
            tem_index = 0
            for i in range(1, len(key)):
                tem_index = self.prefix_search(key[i:])
                if tem_index > 0:
                    break
            self.next[key_index] = tem_index if tem_index > 0 else 0

    def match(self, text):
        match_pair = []
        start = parent_value = parent_index = 0
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
            else:
                while parent_index > 0:
                    parent_index = self.next[parent_index]
                    parent_value = abs(self.node[parent_index])
                    if parent_value + w_index < len(self.node) and self.node[parent_value + w_index] != 0 and \
                            self.check[parent_value + w_index] == parent_index:
                        for tem_i in range(start, i):
                            if self.prefix_search(text[start:i+i]) > 0:
                                start = i
                                break
                        parent_index = parent_value + w_index
                        parent_value = abs(self.node[parent_index])
                        if self.node[parent_index] < 0:
                            match_pair.append((start, i + 1))
                        break
        return match_pair

    def prefix_search(self, key):
        parent_value = parent_index = 0
        for i in range(len(key)):
            w = key[i]
            w_index = self.get_index(w)
            if w_index + parent_value < len(self.check) and self.node[w_index + parent_value] != 0 and self.check[
                w_index + parent_value] == parent_index:
                parent_index = w_index + parent_value
                parent_value = abs(self.node[parent_index])
            else:
                return -1
        return parent_index

    def match_exact(self, key):
        parent_index = self.prefix_search(key)
        if self.node[parent_index] < 0:
            return True
        return False


if __name__ == '__main__':
    keys = ["南京大学",  "南理工", "南京"]
    # keys.sort()
    # print(keys)
    # trie = Trie(keys)
    trie = AhoCorasickAutoMation(keys)
    text = "南京大学跟南理工"
    match_pair = trie.match(text)
    for i,j in match_pair:
        print(text[i:j])


