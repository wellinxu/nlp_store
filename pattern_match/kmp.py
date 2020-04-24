

class KMP(object):
    def __init__(self, key):
        self.key = key
        self.length = len(self.key)
        self.next = [0 for i in self.key]
        self.self_match()
        print(self.next)

    def self_match(self):
        if self.length < 2:
            return
        for i in range(2, self.length):
            for j in range(1, i):
                if self.key[:i-1 - j] == self.key[j:i - 1]:
                    # 当前位置的后缀是关键词的前缀
                    self.next[i] = i-j
                    break

    def match(self, text):
        match_pair = []
        current_ind = 0
        for i in range(len(text)):
            if text[i] == self.key[current_ind]:
                if self.stop_ind(current_ind):
                    match_pair.append((i - self.length + 1, i + 1))
                    current_ind = 0
                else:
                    current_ind += 1
            else:
                while current_ind > 0:
                    current_ind = self.next[current_ind]
                    if text[i] == self.key[current_ind]:
                        current_ind += 1
                        break
        return match_pair

    def stop_ind(self, current_ind):
        return current_ind + 1 == self.length




if __name__ == '__main__':
    kmp = KMP("abdabc")
    text = "abdabdabdabeabdabdabcabdabchabdabc"
    match_pair = kmp.match(text)
    for i,j in match_pair:
        print(text[i:j])






