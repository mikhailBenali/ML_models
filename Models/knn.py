from xml.etree.ElementPath import xpath_tokenizer_re

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, x, y, k=5):
        self.x = x
        self.y = y
        self.k = k

    def predict(self, x):
        distances = np.linalg.norm(self.x - x, axis=1)
        indices = np.argsort(distances)[:self.k]
        return np.bincount(self.y[indices]).argmax()

    def batch_predict(self, x):
        return np.array([self.predict(xi) for xi in x])

def test_knn():

    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    knn = KNN(x, y, k=11)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    plt.figure()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm")
    plt.scatter(x_test[0, 0], x_test[0, 1], c="green")
    plt.title("KNN Classifier")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    print(f"Predicted class: {knn.predict(x_test[0])}")
    print(f"Actual class: {y_test[0]}")

    y_preds = knn.batch_predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_preds)}")

def test_k_parameter():
    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    accuracies = []
    for k in range(3,50,2):
        knn = KNN(x, y, k=k)
        accuracy = accuracy_score(y_test, knn.batch_predict(x_test))
        accuracies.append(accuracy)

    plt.bar(range(3,50,2), accuracies)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN Classifier Accuracy vs k")
    plt.show()

if __name__ == "__main__":
    test_knn()
    test_k_parameter()