# Multiple Linear Regression: Polynomial
# ----------------------------------
# Cost function: MSE
# Optimization algorithm: Gradient Descent
# ----------------------------------
# Coded by: Igor Mafra Felipe
# 11/25/20

import numpy as np

def first_degree(x_i, w, b):
    return np.dot(x_i, w) + b

def second_degree(x_i, w, b):
    return np.dot(x_i ** 2, w) + b

def third_degree(x_i, w, b):
    return np.dot(x_i ** 3, w) + b

DEGREE_FUNCTIONS = {
    1: first_degree,
    2: second_degree,
    3: third_degree    
}

def cost_function(x, y, w, b, degree, regularization):
    m, n = x.shape
    
    total_cost = 0
    for i in range(m):
        f_wb_i = DEGREE_FUNCTIONS[degree](x[i], w, b)
        total_cost += (f_wb_i - y[i]) ** 2
    
    reg = 0
    for j in range(n):
        reg += w[j] ** 2
    reg = reg * regularization / (2 * m)
    
    total_cost = (total_cost / (2 * m)) + reg
    
    return total_cost

def gradient(x, y, w, b, degree, regularization):
    m, n = x.shape # m = number of samples, n = number of features
    dj_dw = np.zeros(n)
    dj_db = 0
    
    for i in range(m):
        f_wb_x = DEGREE_FUNCTIONS[degree](x[i], w, b)
        dj_db += f_wb_x - y[i]
        for j in range(n):
            reg = regularization * w[j] / m
            dj_dw[j] += (f_wb_x - y[i]) * x[i][j] + reg
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db
        

def polynomial_regression(x, y, w=0, b=0, a = 0.003, num_iter=2000, degree=1, regularization=0):
    if degree not in [1, 2, 3]:
        print("Degree must be 1, 2 or 3")
        return
    
    cost_history = []
    for i in range(num_iter):
        dj_dw, dj_db = gradient(x, y, w, b, degree, regularization)
        w = w - a * dj_dw
        b = b - a * dj_db
        
        cost = cost_function(x, y, w, b, degree, regularization)
        cost_history.append(cost)
    
    return w, b, cost_history