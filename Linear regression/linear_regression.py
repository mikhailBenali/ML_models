import numpy as np
import matplotlib.pyplot as plt
from random import randint

x = np.random.randint(0, 100, 100)
y = np.random.normal(0, 100, 100)

def least_squares(x,y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)

def gradient_descent(x, y, b0, b1, alpha):
    n = np.size(x)
    y_pred = b0 + b1*x
    d_b0 = (-2/n) * sum(y - y_pred)
    d_b1 = (-2/n) * sum(x * (y - y_pred))
    b0 = b0 - alpha * d_b0
    b1 = b1 - alpha * d_b1
    return (b0, b1)

def show_coeffs(b0, b1):
    print(f"y = {b0} + {b1}x")

f = (randint(0, 10), randint(0, 10))
show_coeffs(f[0], f[1])

print("Let's apply gradient descent to improve the model")
for i in range(100):
    f = gradient_descent(x, y, f[0], f[1], 0.0001)
    show_coeffs(f[0], f[1])
    plt.scatter(x, y)
    plt.axline((0, f[0]), slope=f[1], color='red')
    plt.show()

f = least_squares(x, y)
print(f"With the least squares method we get:")
show_coeffs(f[0], f[1])
plt.scatter(x, y)
plt.axline((0, f[0]), slope=f[1], color='green')
plt.show()