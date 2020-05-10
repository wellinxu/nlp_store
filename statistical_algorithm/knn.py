

class KNN(object):
    def __index__(self, samples):
        self.samples = samples
        self.feature_len = len(samples[0])
        self.root = self.kd_tree(self.samples, 0)

    def kd_tree(self, samples, i):
        deepth = i % self.feature_len
        left_size = len(samples)
        samples.sort(key=lambda x:x[deepth])
        if left_size == 1:
            node = Node(samples[0], i, samples[0][deepth])
        elif left_size % 2 == 0:
            node = Node(None, i, samples[left_size/2][deepth])
            node.set_left(self.kd_tree(samples[:left_size / 2], i+1))
            node.set_right(self.kd_tree(samples[left_size / 2:], i+1))
        else:
            node = Node(samples[left_size / 2], i, samples[left_size/2][deepth])
            node.set_left(self.kd_tree(samples[:left_size / 2], i+1))
            node.set_right(self.kd_tree(samples[left_size / 2 + 1:], i+1))
        return node

    def search_k(self, sample, k, node):
        result = []
        first_node = node
        value = sample[node.deepth]
        while not node.is_end_node():
            if value < node.value:
                node = node.left
            else:
                node = node.right
            value = sample[node.deepth]
        max_dis = self.distance(node.sample, sample)
        result.append((node, max_dis))
        while node != first_node:
            parent_node = node.parent
            if parent_node.sample:
                tem_distance = self.distance(parent_node.sample, sample)
                if tem_distance < max_dis or len(result) < k:
                    result.append((parent_node, tem_distance))
                    result.sort(key=lambda x: x[1])
                    if len(result) > k:
                        result = result[:k]
                    max_dis = result[-1][1]
            brother_node = node.get_brother()
            abs_dis = abs(sample[parent_node.deepth] - parent_node.value)
            if abs_dis < max_dis or len(result) < k:
                tem_result = self.search_k(sample, k, brother_node)
                result.extend(tem_result)
                result.sort(key=lambda x: x[1])
                if len(result) > k:
                    result = result[:k]
                max_dis = result[-1][1]
            node = parent_node
        return result

    def distance(self, sample1, sample2):
        return 1


class Node(object):
    def __init__(self, sample, deepth, value):
        self.sample = sample
        self.value = value
        self.deepth = deepth
        self.parent = None
        self.left = None
        self.right = None

    def is_end_node(self):
        if self.left or self.right:
            return False
        return True

    def set_left(self, left):
        self.left = left
        self.left.parent = self

    def set_right(self, right):
        self.right = right
        self.right.parent = self

    def get_brother(self):
        if self.parent:
            if self.parent.left == self:
                return self.parent.right
            return self.parent.left
        return None