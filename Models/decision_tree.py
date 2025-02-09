import numpy as np
import matplotlib.pyplot as plt

class Leaf():
    def __init__(self, value, left=None, right=None, feature=None, threshold=None):
        self.value = value
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold


class DecisionTree():
    def __init__(self, root=None, max_depth=100):
        self.root = root
        self.max_depth = max_depth

    def grow_tree(self, X, y, depth=0):
        # If the depth is equal to the max depth or if all the labels are the same, return a leaf node
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Leaf(value=np.mean(y))

        feature, threshold = self.find_best_split(X, y)
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        left = self.grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self.grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Leaf(left=left, right=right, feature=feature, threshold=threshold)

    def find_best_split(self, X, y):
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                gini = self.calculate_gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

