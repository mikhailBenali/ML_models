import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(x, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        linear_model = np.dot(x, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def test_logistic_regression():
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn import metrics
    import matplotlib.pyplot as plt

    # Load dataset
    bc = datasets.load_breast_cancer()
    x, y = bc.data, bc.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=10**4)
    regressor.fit(x_train, y_train)
    predictions = regressor.predict(x_test)

    cm = metrics.confusion_matrix(y_test, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(cm)
    cm_display.plot()
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='winter')
    plt.show()

if __name__ == "__main__":
    test_logistic_regression()