import numpy as np
from copy import deepcopy


class ID3(object):
    def __init__(self):
        self.tree = None

    def test(self, train):
        right = 0
        wrong = 0
        for i in range(train.shape[0]):
            if self.tree.classify(train[i]) == train[i][0]:
                right += 1
            else:
                wrong += 1
        print('right answers: ', right)
        print('wrong answers: ',  wrong)

    def train(self, data):
        self.tree = TreeModel()
        self.tree.build(data)


class TreeModel(object):
    def __init__(self, feature_used=[]):
        self.threshold = None
        self.index = None
        self.left = None
        self.right = None
        self.label = None
        self.feature_used = feature_used

    def build(self, data):
        # checking if all in the same label
        if self.feature_used == [23,24, 21, 1, 5, 10]:
            print(1)
        if self.same_label(data):
            self.label = data[0][0]  #all the same
            return
        # splitting the the tree
        self.threshold, self.index, lower, higher = self.information_gain(data, self.feature_used)
        self.feature_used.append(self.index)
        if len(self.feature_used) == data.shape[1]-1:  # not consistent(-1 because label is not feature) :
            self.label = 0
            return
        if len(lower)==0 or len(higher)==0:
            print(1)
        self.left = TreeModel(deepcopy(self.feature_used))
        self.right = TreeModel(deepcopy(self.feature_used))
        self.left.feature_used.append(10)
        self.right.build(higher)

    def classify(self, sample):

        if self.label is None and self.right is None and self.left is None:
            print('not initail')
            return -1
        if self.label is not None:
            return self.label
        elif sample[self.index] <= self.threshold:
            return self.left.classify(sample)
        return self.right.classify(sample)

    @staticmethod
    def same_label(data):
        num_of_ones = np.count_nonzero(data[:, :1])
        if num_of_ones == len(data) or num_of_ones == 0:
            return True
        return False


    @staticmethod
    def information_gain(data, feature_used):
        size = len(data)
        min_entropy = float('inf')
        best_feature_index = -1
        threshold = None
        for i in range(1, len(data[0])):  #skipping label
            if i in feature_used:
                continue
            min_entropy_local = 1
            threshold_local = None
            best_lower = None
            best_higher = None
            unique_val = np.unique(data[:, i])
            for val in unique_val:
                lower = data[data[:, i] <= val]
                higher = data[data[:, i] > val]
                lower_ones = np.count_nonzero(lower[:, :1])
                lower_zeros = len(lower) - lower_ones
                higher_ones = np.count_nonzero(higher[:, :1])
                higher_zeros = len(higher) - higher_ones

                lower_entropy_score = entropy(lower_zeros, lower_ones)
                higher_entropy_score = entropy(higher_zeros, higher_ones)
                entropy_score = (len(lower) * lower_entropy_score + (len(higher)) * higher_entropy_score) / size
                if entropy_score <= min_entropy_local:
                    min_entropy_local = entropy_score
                    threshold_local = val
                    best_lower = lower
                    best_higher = higher
            if min_entropy_local < min_entropy:
                threshold = threshold_local
                min_entropy = min_entropy_local
                best_feature_index = i

        return threshold, best_feature_index, best_lower, best_higher


def entropy(a, b):
    if a == 0 or b == 0:
        return 0
    total = a + b
    return (-a/total * np.log2(a/total)) - (b/total * np.log2(b/total))


def prepare_date():
    train_data = np.loadtxt('train.csv', delimiter=',', skiprows=1)
    test_data= np.loadtxt('test.csv', delimiter=',', skiprows=1)
    # train_y, train_x = np.split(train_data, [1], axis=1)
    test_y, test_x = np.split(test_data, [1], axis=1)
    return train_data, test_y, test_x


def main():
    id3_tree = ID3()
    train, test_y, test_x = prepare_date()
    id3_tree.train(train)



if __name__ == '__main__':
    main()
