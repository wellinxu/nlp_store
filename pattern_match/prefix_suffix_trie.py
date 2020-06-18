class Trie:

    def __init__(self, suffix=False):
        self.node = {}
        self.end = False
        self.value = None
        self.suffix = suffix

    def insert(self, word, value):
        word = self.get_word(word)
        cur_node = self
        for w in word:
            if w in cur_node.node:
                cur_node = cur_node.node[w]
            else:
                cur_node.node[w] = Trie()
                cur_node = cur_node.node[w]
        cur_node.end = True
        cur_node.value = value

    def is_prefix_or_suffix(self, word):
        word = self.get_word(word)
        cur_node = self
        for w in word:
            if w not in cur_node.node:
                return -1
            cur_node = cur_node.node[w]
        if cur_node.end:
            return 1
        return 0

    def has_prefix_or_suffix(self, word):
        word = self.get_word(word)
        cur_node = self
        for w in word:
            if w not in cur_node.node:
                return False
            cur_node = cur_node.node[w]
            if cur_node.end:
                return True
        if cur_node.end:
            return True
        return False

    def get_word(self, word):
        if self.suffix:
            word = word[::-1]
        return word

    def search(self, text):
        text = self.get_word(text)
        result = []
        text_len = len(text)
        cur_index = 0
        for i in range(text_len):
            if i < cur_index:
                continue
            indexs = (0, 0)
            for j in range(i+1, text_len+1):
                word = text[i: j]
                flag = self.is_prefix_or_suffix(self.get_word(word))
                if flag == -1:
                    break
                elif flag == 1:
                    indexs = (i, j)
            if indexs[1] > 0:
                result.append(indexs)
                cur_index = indexs[1]
        if self.suffix:
            result = [(text_len - b, text_len - a) for a, b in result]
        return result


class PrefixTrie(Trie):
    def __init__(self):
        super(PrefixTrie, self).__init__()


class SuffixTrie(Trie):
    def __init__(self):
        super(SuffixTrie, self).__init__(True)


if __name__ == '__main__':
    trie = SuffixTrie()
    # trie = PrefixTrie()
    ws = ["南京", "南京大学", "南理工", "江苏"]
    for w in ws:
        trie.insert(w, "")
    text = "南京大学3江苏"
    # print(trie.has_prefix_or_suffix(text))
    print(trie.search(text))
