import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import gradient_descent

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generating two classes of data for logistic regression
x, y = make_classification()
print(x.shape, y.shape)

# Plotting the generated data
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], label='Class 0', c='blue', edgecolor='k')
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], label='Class 1', c='red', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data for Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# Linear Regression: Predicting new values
model = LogisticRegression()
model.fit(x, y)
y_predict = model.predict(x)
print("Accuracy on training set:", model.score(x, y))