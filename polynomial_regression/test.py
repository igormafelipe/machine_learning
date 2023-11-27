# Multiple Linear Regression: Polynomial, test

# Coded by: Igor Mafra Felipe
# 11/25/2023

# Realized that plotting anything other than 1 feature is hard.

import numpy as np
import matplotlib.pyplot as plt
from polynomial_regression import polynomial_regression

def first_degree(x_i, w, b):
    return np.dot(x_i, w) + b

def second_degree(x_i, w, b):
    return np.dot(x_i ** 2, w) + b

np.random.seed(0)
x = 2 * np.random.rand(100, 2)
y = 4 + np.random.randn(100, 1)

fig, axs = plt.subplots(1, 1, figsize=(12, 5))

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 1], y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample data for Linear Regression')

plt.tight_layout()
plt.show()

for i in range(1, 4):
    for j in range(x.shape[1]):
        w, b, _ = polynomial_regression(x[:, j].reshape(-1, 1), y, [0], 0, 0.001, num_iter=2000, degree=i)

        x_plot = np.linspace(min(x[:, j]), max(x[:, j]), 100).reshape(-1, 1)

        if i == 1:
            y_predict = first_degree(x_plot, w, b)
        elif i == 2:
            x_plot_poly = np.column_stack((x_plot, x_plot ** 2))
            y_predict = second_degree(x_plot_poly[:, 1].reshape(-1, 1), w, b)

        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, j], y)
        plt.plot(x_plot, y_predict, 'r-', label=f'Degree {i}')
        plt.xlabel(f'x_{j+1}')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Polynomial Regression: Degree {i}, Feature {j+1}')
        plt.show()
