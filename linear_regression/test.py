import numpy as np
import matplotlib.pyplot as plt
from linear_regression import gradient_descent

# Generating random data for demonstration
np.random.seed(0)
x = 2 * np.random.rand(100, 1)  
y = 4 + 3 * x + np.random.randn(100, 1)

# Plotting the generated data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample Data for Linear Regression')
plt.show()

# Linear Regression: Predicting new values
w, b, cost_history = gradient_descent(x, y, 0, 0, 0.001, num_iter=2000)

# Plotting the cost function
plt.plot(cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()

# Analysis: The algorithm converges too slowly, and the number of iterations is 
#           too high. We can tell it since the convergence happens at around 
#           1250 iterations.

# Plotting the regression line
y_predict = w * x + b
plt.plot(x, y_predict, 'r-')
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()