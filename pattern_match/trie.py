

class Trie(object):
    def __init__(self, keys):
        self.keys = keys
        self.root = Node(None)
        self.build()

    def build(self):
        for key in self.keys:
            tem_node = self.root
            for w in key:
                if not tem_node.has_child(w):
                    tem_node.add_child(w)
                tem_node = tem_node.get_child(w)
            tem_node.end = True

    def match(self, text):
        match_pair = []
        node = self.root
        start = 0
        for i in range(len(text)):
            w = text[i]
            if node.has_child(w):
                if not node.value:
                    start = i
                node = node.get_child(w)
                if node.end:
                    match_pair.append((start, i+1))
            elif node.value:
                i = start
                node = self.root
        return match_pair


class Node(object):
    def __init__(self, value):
        self.value = value
        self.child = {}
        self.child_key = []    # 为了保证key的顺序
        self.end = False

    def has_child(self, key):
        if key in self.child.keys():
            return True
        return False

    def get_child(self, key):
        return self.child[key]

    def add_child(self, key):
        node = Node(key)
        self.child[key] = node
        self.child_key.append(key)




chars = "abcdefghijklmnopqrstuvwxyz"
def create_char(length):
    word = ""
    for i in range(length):
        word += random.choice(chars)
    return word


if __name__ == '__main__':
    import random
    from pattern_match.double_array_trie import DoubleArrayTrie
    lengths = [3,4,5,6,7,8,9]
    keys = []
    for i in range(1000):
        keys.append(create_char(random.choice(lengths)))
    text = create_char(10000)
    trie = DoubleArrayTrie(keys)
    match_pair = trie.match(text)
    for i,j in match_pair:
        print(text[i:j])
    keys = ["南京大学",  "南理工", "南京"]
    # keys.sort()
    # print(keys)
    # trie = Trie(keys)
    trie = DoubleArrayTrie(keys)
    text = "南京大学跟南理工"
    match_pair = trie.match(text)
    for i,j in match_pair:
        print(text[i:j])


