# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# one we assume theta_0 and theta_1 params below predict function returns y_hat
def predict(theta_0: float, theta_1: float, x_i: float) -> float:
    return theta_1 * x_i + theta_0

# since we know the actual y_i output we can compute the error  for each pair
def error(theta_0: float, theta_1: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting theta_1 * x_i + theta_0, when actual value is y_i
    """
    return predict(theta_0, theta_1, x_i) - y_i

# we woudl like to know the total error over the entire dataset
# sum of squared errors covers both negative and positive errors, preventing error cancel out. 

from linear_algebra import Vector

def sum_of_sqerrors(theta_0: float, theta_1: float, x: Vector, y: Vector) -> float:
    return sum(error(theta_0, theta_1, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y) )


# +
from typing import Tuple
from statistics import correlation, standard_deviation, mean

# based on the OLS, the error-minimizing theta_0 and theta_1 
def least_squares_fit (x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y, find the least-squares values of theta_0 and theta_1
    """
    theta_1 = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    theta_0 = mean(y) - theta_1 * mean(x)
    return theta_0, theta_1


# -

# Some good intuition notes by the book:
#
# The choice of theta_0 simply states we are trying to predict the averga of the reponse variable via the average the input variable
# The choice of theta_1 means when the input value increases by standard_deviation(x) the prediction then increases by correlation(x, y) * standard_deviation(y)

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

assert least_squares_fit(x, y) == (-5, 3)

# +
from statistics import num_friends_good, daily_minutes_good

theta_0, theta_1 = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < theta_0 < 23.0
assert 0.9 < theta_1 < 0.905
# -

# A common measure of model perfroamnce with OLS is the R-squared, which measures the fraction of the total variation of the output variable that is captured by the model

from statistics import de_mean


# +
def total_sum_squares(y: Vector) -> float:
    """the total squared variation of y_i's from their mean"""
    return sum(v **2 for v in de_mean(y))

def r_squared(theta_0: float, theta_1: float, x: Vector, y: Vector) -> float:
    """
    the fraction of variation in y captured by the model which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(theta_0, theta_1, x, y) /
                  total_sum_squares(y))

rsq = r_squared(theta_0, theta_1, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330
# -

# Using gradient descent

import random
from gradient_descent import gradient_step

# +
num_epochs = 10000
random.seed(0)

theta_guess = [random.random(), random.random()] # choose random value to start

learning_rate = 0.00001

for _ in range(num_epochs):
    theta_0, theta_1 = theta_guess
    # Partial derivatives wirh respect to theta_0 and theta_1
    grad_theta_0 = sum(2 * error(theta_0, theta_1, x_i, y_i) 
                      for x_i, y_i in zip(num_friends_good, daily_minutes_good))
    
    
    grad_theta_1 = sum(2 * error(theta_0, theta_1, x_i, y_i) * x_i
                      for x_i, y_i in zip(num_friends_good, daily_minutes_good))
    
    # compute loss 
    loss = sum_of_sqerrors(theta_0, theta_1, num_friends_good, daily_minutes_good)
    print(f"loss: {loss:.3f}")
    
    # update the guess
    theta_guess = gradient_step(theta_guess, [grad_theta_0, grad_theta_1], -learning_rate) 

theta_guess = theta_0, theta_1

assert 22.9 < theta_0 < 23.0
assert 0.9 < theta_1 < 0.905
# -


