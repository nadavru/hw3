import numpy as np
from copy import deepcopy


class ID3_epsilon(object):
    def __init__(self):
        self.tree = None

    def test(self, test):
        right = 0
        wrong = 0
        for i in range(test.shape[0]):
            results = self.tree.classify(test[i])
            if (results[0] + results[1] > 1):
                # print(i ,'affected', results)
                x = 1

            if (results[0] > results[1] and test[i][0] == 0) or (results[1] >= results[0] and test[i][0] == 1):
                right += 1
            else:
                wrong += 1
                print('wrongly classified ', i, results)
        print('right answers: ', right)
        print('wrong answers: ',  wrong)

    def train(self, data):
        self.tree = TreeModel()
        self.tree.build(data)


class TreeModel(object):
    def __init__(self, min_size=9):
        self.threshold = None
        self.index = None
        self.left = None
        self.right = None
        self.label = None
        self.orig_data = None
        self.epsilon = None
        self.min_size = min_size

    def build(self, data):
        if self.orig_data is None:  # first time init
            self.orig_data = data
            self.epsilon = [np.std(data[:, i]) * 0.1 for i in range(data.shape[1])]
            # self.epsilon = np.zeros(data.shape[1])  # for testing

        # checking if all in the same label
        if self.same_label(data) or self.min_size >= data.shape[0]:
            self.label = self.get_common_label(data)  #all the same
            return
        # splitting the the tree

        self.threshold, self.index, lower, higher = self.information_gain(data)


        self.left = TreeModel()
        self.right = TreeModel()
        self.left.build(lower)
        self.right.build(higher)

    def classify(self, sample):
        # print('index:', self.index, ' threshold:', self.threshold)
        results = [0, 0]
        if self.label is None and self.right is None and self.left is None:
            print('not initail')
            return -1
        if self.label is not None:
            results[self.label] += 1
            return results
        if sample[self.index] <= self.threshold + self.epsilon[self.index]:
            temp = self.left.classify(sample)
            results[0] += temp[0]
            results[1] += temp[1]
        if sample[self.index] + self.epsilon[self.index] >= self.threshold:
            temp = self.right.classify(sample)
            results[0] += temp[0]
            results[1] += temp[1]
        return results

    @staticmethod
    def same_data(data):
        try:
            if len(data) < 2:
                return True
            last = data[0]
            for sample in data[1:]:
                    if (last != sample).any():
                        return False
                    last = sample
            return True
        except:
            return False

    @staticmethod
    def same_label(data):
        num_of_ones = np.count_nonzero(data[:, :1])
        if num_of_ones == len(data) or num_of_ones == 0:
            return True
        return False

    @staticmethod
    def get_common_label(data):
        num_of_ones = np.count_nonzero(data[:, :1])
        if num_of_ones > len(data) - num_of_ones:
            return 1
        return 0



    @staticmethod
    def information_gain(data):
        size = len(data)
        min_entropy = float('inf')
        best_feature_index = -1
        threshold = None
        for i in range(1, len(data[0])):  #skipping label
            min_entropy_local = float('inf')
            threshold_local = None
            unique_val = np.unique(data[:, i])
            for j in range(len(unique_val)-1):
                lower = data[data[:, i] <= unique_val[j]]
                higher = data[data[:, i] > unique_val[j]]
                lower_ones = np.count_nonzero(lower[:, :1])
                lower_zeros = len(lower) - lower_ones
                higher_ones = np.count_nonzero(higher[:, :1])
                higher_zeros = len(higher) - higher_ones

                lower_entropy_score = entropy(lower_zeros, lower_ones)
                higher_entropy_score = entropy(higher_zeros, higher_ones)
                entropy_score = (len(lower) * lower_entropy_score + (len(higher)) * higher_entropy_score) / size
                if entropy_score < min_entropy_local:
                    min_entropy_local = entropy_score
                    threshold_local = (unique_val[j]  + unique_val[j+1]) /2
                    best_lower_local = lower
                    best_higher_local = higher
            if min_entropy_local < min_entropy:
                threshold = threshold_local
                min_entropy = min_entropy_local
                best_lower = best_lower_local
                best_higher = best_higher_local
                best_feature_index = i

        return threshold, best_feature_index, best_lower, best_higher


def entropy(a, b):
    if a == 0 or b == 0:
        return 0
    total = a + b
    return (-a/total * np.log2(a/total)) - (b/total * np.log2(b/total))


def prepare_date():
    train_data = np.loadtxt('train.csv', delimiter=',', skiprows=1)
    test_data =  np.loadtxt('test.csv', delimiter=',', skiprows=1)
    return train_data, test_data


def main():
    id3_tree = ID3_epsilon()
    train, test = prepare_date()
    id3_tree.train(train)
    id3_tree.test(test)

if __name__ == '__main__':
    main()
