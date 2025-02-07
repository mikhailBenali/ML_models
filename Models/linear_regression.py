import numpy as np
import matplotlib.pyplot as plt
from random import randint

class linear_regression():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.b0 = np.mean(y)
        self.b1 = 0
        self.error = 0

    def compute_error(self):
        y_pred = self.b0 + self.b1 * self.x
        self.error = y_pred - self.y

    def fit(self, lr, epochs):
        """
        :param x,y: coordinates
        :param b0: intercept
        :param b1: slope
        :param lr: learning rate
        :return: b0,b1
        """
        n = np.size(self.x)
        for _ in range(epochs):
            self.compute_error()
            d_b0 = (2/n) * sum(self.error)
            d_b1 = (2/n) * sum(self.error * self.x)
            self.b0 = self.b0 - lr * d_b0
            self.b1 = self.b1 - lr * d_b1

        self.show_coeffs()
        self.compute_error()


    def show_coeffs(self):
        print(f"y = {self.b0} + {self.b1}x")

    def plot(self, color="red"):
        plt.scatter(self.x, self.y)
        plt.axline((0, self.b0), slope=self.b1, color=color)
        plt.show()

    def predict(self, x):
        return self.b0 + self.b1*x

    def ordinary_least_squares(self):
        """
        :param x,y: coordinates
        :return: b0,b1
        """
        n = np.size(self.x)
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        xy_mean = np.mean(self.x*self.y)
        x_squared_mean = np.mean(self.x**2)

        self.b1 = (x_mean * y_mean - xy_mean) / (x_mean**2 - x_squared_mean)
        self.b0 = y_mean - self.b1 * x_mean

        self.show_coeffs()
        self.plot("green")


def tests():
    for _ in range(10):
        x = np.random.randint(0, 100, 500)
        y = np.random.normal(30, 60, 500)

        # We create a linear regression model
        f = linear_regression(x,y)
        f.show_coeffs()
        f.plot()

        # We fit the model
        f.fit(5*10**-5, 100)
        f.show_coeffs()
        f.plot()

        # We use the ordinary least squares method to compare the results
        f.ordinary_least_squares()

tests()