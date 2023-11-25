import numpy as np
import matplotlib.pyplot as plt
from multiple_linear_regression import multiple_gradient_descent

# Generating random data for demonstration
np.random.seed(0)
x = 2 * np.random.rand(100, 2)
y = 4 + np.random.randn(100, 1)

# Creating subplots
fig, axs = plt.subplots(1, 1, figsize=(12, 5))

# Plotting the generated data
plt.scatter(x[:, 0], y)
plt.scatter(x[:, 1], y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample data for Linear Regression')

plt.tight_layout()
plt.show()

# Linear Regression: Predicting new values
w, b, cost_history = multiple_gradient_descent(x, y, [0, 0], 0, 0.001, num_iter=2000)

# Plotting the cost function
plt.plot(cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()

# Plotting the regression line
# Creating subplots
fig, axs = plt.subplots(1, 1, figsize=(12, 5))

# Plotting the generated data
y_predict = w * x + b
plt.plot(x, y_predict, 'r-')

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 1], y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()