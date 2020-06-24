import numpy as np
from scipy.stats import entropy, entr


class ID3(object):
    def __init__(self):
        self.model = None

    def test(self, data):
        if self.model is None:
            return


    def train(self, data, label):




class TDIDT(object):
    def __init__(self, feature_used = []):
        self.threshold = None
        self.index = None
        self.left = None
        self.right = None
        self.labal = 0
        self.feature_used = feature_used

    def build(self, data):
        self.threshold, min_entropy, self.index = self.information_gain(data, self.feature_used)
        self.feature_used.append(self.index)
        self.left = TDIDT(self.feature_used).build(data)
        self.right = TDIDT(self.feature_used).build(data)

    def classify(self, sample):
        if self.right is None and self.left is None:
            return self.labal
        elif sample[self.index] < self.threshold:
            return  self.left.classify()
        return self.right.classify()

    @staticmethod
    def same_label(data):
        if np.count_nonzero(data[:, :0]) == len(data):
            return True
        return False


    @staticmethod
    def information_gain(data, feature_used):
        size = len(data)
        min_entropy = float('inf')
        best_feature_index = -1
        threshold = None
        for i in range(len(data[0])):
            if i in feature_used:
                continue
            temp_data = data[np.argsort(data[:, i])]
            min_entropy_local = 1
            threshold_local = None
            for j in range(1 , len(data)):
                lower, higher = np.split(temp_data, [j], axis=0)
                lower_ones = np.count_nonzero(lower[:, :1])
                lower_zeros = len(lower) - lower_ones
                higher_ones = np.count_nonzero(higher[:, :1])
                higher_zeros = len(higher) - higher_ones

                lower_entropy_score = entropy(lower_zeros, lower_ones)
                higher_entropy_score = entropy(higher_zeros, higher_ones)
                entropy_score = (j * lower_entropy_score + (size - i) * higher_entropy_score) / size
                if entropy_score <= min_entropy_local:
                    min_entropy_local = entropy_score
                    threshold_local = temp_data[i:i + 1, j:j + 1]
            if min_entropy_local < min_entropy:
                threshold = threshold_local
                min_entropy = min_entropy_local
                best_feature_index = i

            return threshold, min_entropy, best_feature_index


def entropy(a, b):
    if a == 0 or b == 0:
        return 0
    total = a + b
    return (-a * np.log(a/total) - b * np.log(b/total)) / total


def prepare_date(args):
    train_data = np.loadtxt('train.csv', delimiter=',', skiprows=1)
    test_data= np.loadtxt('test.csv', delimiter=',', skiprows=1)
    # train_y, train_x = np.split(train_data, [1], axis=1)
    test_y, test_x = np.split(test_data, [1], axis=1)
    return train_data, test_y, test_x


def main():
    id3_tree = ID3()
    train, test_y, test_x = prepare_date()


if __name__ == '__main__':
    main()
