import matplotlib.pyplot as plt
import numpy as np

class Leaf():
    def __init__(self, value=None, left=None, right=None, feature=None, threshold=None):
        self.value = value
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold

class DecisionTree():
    def __init__(self, root=None, max_depth=100):
        self.root = root
        self.max_depth = max_depth

    def grow_tree(self, x, y, depth=0):
        # If the depth is equal to the max depth or if all the labels are the same, return a leaf node
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Leaf(value=np.mean(y))

        feature, threshold, left_indices, right_indices = self.find_best_split(x, y)

        left = self.grow_tree(x[left_indices], y[left_indices], depth + 1)
        right = self.grow_tree(x[right_indices], y[right_indices], depth + 1)

        return Leaf(left=left, right=right, feature=feature, threshold=threshold)

    def find_best_split(self, x, y):
        best_gini = 1
        best_feature = None
        best_threshold = None

        for feature in range(x.shape[1]):
            for threshold in np.unique(x[:, feature]):
                left_indices = x[:, feature] < threshold
                right_indices = x[:, feature] >= threshold

                gini = self.calculate_gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    left_best_indices = left_indices
                    right_best_indices = right_indices

        return best_feature, best_threshold, left_best_indices, right_best_indices

    def calculate_gini(self, left, right):
        # Calculate the gini impurity of the left and right nodes
        # Gini impurity = 1 - sum(p_i^2)
        # p_i = frequency of class i in the node

        left_gini = 1 - np.sum((np.unique(left, return_counts=True)[1] / len(left))**2)
        right_gini = 1 - np.sum((np.unique(right, return_counts=True)[1] / len(right))**2)

        return (len(left) * left_gini + len(right) * right_gini) / (len(left) + len(right)) # We return the weighted average of the gini impurity

    def fit(self, x, y):
        self.root = self.grow_tree(x, y)

    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def predict(self, x):
        return self.traverse_tree(x, self.root)

    def describe_tree(self):
        def print_tree(node, depth=0):
            if node.value is not None:
                print(" " * depth, "Predict", node.value)
            else:
                print(" " * depth, "Feature", node.feature, "<", node.threshold)
                print_tree(node.left, depth + 1)
                print(" " * depth, "Feature", node.feature, ">=", node.threshold)
                print_tree(node.right, depth + 1)

        print_tree(self.root)

    def plot_boundaries(self, x, y):
        def get_features_tresholds(node):
            if node.value is not None:
                return []
            else:
                return [(node.feature, node.threshold)] + get_features_tresholds(node.left) + get_features_tresholds(node.right)
        features_tresholds = get_features_tresholds(self.root)

        # dimension > 2
        if x.shape[1] > 2:
            fig, ax = plt.subplots(x.shape[1], x.shape[1], figsize=(20, 20))
            for i in range(x.shape[1]):
                for j in range(x.shape[1]):
                    if i == j:
                        ax[i, j].axis("off")
                        continue

                    for feature, threshold in features_tresholds:
                        if feature == i:
                            ax[i, j].axvline(threshold, color="red")
                        if feature == j:
                            ax[i, j].axhline(threshold, color="blue")

                    ax[i, j].scatter(x[:, i], x[:, j], c=y)
            plt.show()

        # dimension <= 2
        else:
            plt.figure()
            for feature, threshold in features_tresholds:
                plt.axvline(threshold, color="red")
            plt.scatter(x[:, 0], x[:, 1], c=y)
            plt.show()

def decision_tree_iris_test():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    iris = load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = DecisionTree(max_depth=5)
    model.fit(x_train, y_train)


    y_pred = [model.predict(x) for x in x_test]
    print("Accuracy:", accuracy_score(y_test, y_pred))

    model.describe_tree()
    model.plot_boundaries(x_train, y_train)

def decision_tree_np_test():

    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score

    x,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)

    model = DecisionTree(max_depth=5)
    model.fit(x, y)

    print(f"Accuracy: {accuracy_score(y, [model.predict(x) for x in x])}")

    model.predict(x[0])

    model.describe_tree()
    model.plot_boundaries(x, y)

decision_tree_iris_test()
print()
decision_tree_np_test()