# -*- coding: utf-8 -*-
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

tuples = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]

# tuples example
(0.7, # years experience
 48000, # salary
 1 )# with premium account

# lets mutate the data to the format we require
data = [list(row) for row in tuples]
xs = [[1.0] + row[:2] for row in data]
ys = [row[2] for row in data]

# lets attempt to model the problem with linear regression
from matplotlib import pyplot as plt
from working_with_data import rescale
from multiple_linear_regression import least_squares_fit, predict
from gradient_descent import gradient_step

learning_rate = 0.001
rescaled_xs = rescale(xs)
theta = least_squares_fit(rescaled_xs, ys, learning_rate, 1000, 1)

theta

predictions = [predict(x_i, theta) for x_i in rescaled_xs]
predictions[:10]

plt.scatter(predictions, ys)
plt.xlabel("predicted")
plt.ylabel("actual");

# This approach leads to a couple of immediate problems:
#
# - We’d like for our predicted outputs to be 0 or 1, to indicate class membership. It’s fine  if  they’re  between  0  and  1,  since  we  can  interpret  these  as  probabilities
# - an output of 0.25 could mean 25% chance of being a paid member. But the outputsof  the  linear  model  can  be  huge  positive  numbers  or  even  negative  numbers,which  it’s  not  clear  how  to  interpret.  Indeed,  here  a  lot  of  our  predictions  were negative.
# - The  linear  regression  model  assumed  that  the  errors  were  uncorrelated  with  thecolumns of x. But here, the regression coefficent for experience is 0.43, indicat‐ing that more experience leads to a greater likelihood of a premium account. This means  that  our  model  outputs  very  large  values  for  people  with  lots  of  experi‐ence.  But  we  know  that  the  actual  values  must  be  at  most  1,  which  means  thatnecessarily  very  large  outputs  (and  therefore  very  large  values  of  experience)correspond  to  very  large  negative  values  of  the  error  term.  Because  this  is  thecase, our estimate of beta is biased

# #### The logistic(sigmoid) function

import math


# +
def logistic(x: float) -> float:
    return 1.0/ (1 + math.exp(-x))

# a convenient property to be used later

def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)



# +
from linear_algebra import Vector, dot

def _negative_log_likelihood(x: Vector, y: float, theta: Vector) -> float:
    """The negative log likelihood for one data point"""
    if y==1:
        return -math.log(logistic(dot(x, theta)))
    else:
        return -math.log(1 - logistic(dot(x, theta)))



# -

# if we assume different data points are independent from one another the overall likelihood is just the product of the individual likehoods (naive bayes?)
#
# That means overall likehood is the sum of the individual log likelihoods. 
#
#

from typing import List


# +
def negative_log_likelihood(xs: List[Vector], ys: List[float], theta: Vector) -> float:
    """
    the loss function for logistic regression
    """
    return sum(_negative_log_likelihood(x, y, theta) 
               for x, y in zip(xs, ys))

# now the gradient 

from linear_algebra import vector_sum

def _negative_log_partial_j(x: Vector, y: float, theta: Vector, j: int) -> float:
    """
    The jth partial derivative for one data point.
    Here i is the index of the data point
    """
    return -(y - logistic(dot(x, theta))) * x[j]

def _negative_log_gradient(x: Vector, y: float, theta: Vector) -> Vector:
    """
    The gradient for one data point
    """
    return [_negative_log_partial_j(x, y, theta, j) for j in range(len(theta))]

def negative_log_gradient(xs: List[Vector], ys: List[float], theta: Vector) -> Vector:
    return vector_sum([_negative_log_gradient(x, y, theta) for x, y in zip(xs, ys)])
    

# -

from machine_learning import train_test_split
import random
random.seed(0)

# +
x_train, x_test, y_train, y_test = train_test_split(rescaled_xs, ys, 0.33)

learning_rate = 0.01

# start with a random guess for theta

theta = [random.random() for _ in range(3)]

for epoch in range(5000):
    gradient = negative_log_gradient(x_train, y_train, theta)
    theta = gradient_step(theta, gradient, -learning_rate)
    loss = negative_log_likelihood(x_train, y_train, theta)
    print(f"epoch {epoch}; loss {loss:.3f}")
# -

[round(theta_i, 1) for theta_i in theta]

# +
# transform back to the original data

from working_with_data import scale

means, stdevs = scale(xs)
theta_unscaled = [(theta[0] - theta[1] * means[1] / stdevs[1]
                  - theta[2] * means[2] / stdevs[2]), 
                  theta[1] / stdevs[1], theta[2]/ stdevs[2] ]

# -

theta_unscaled
# not easy to interpret as linear regression, all else equal, an extra year experience adds 1.6 to the input of logistic
# all else equal an extra $10,000 of salary substracts -2.88 from the input of logistic

# +
# lets use the test data and assume paid account whenever probability exceeds 0.5

tp = fp = fn = tn = 0

for x_i, y_i in zip(x_test, y_test):
    prediction = logistic(dot(theta, x_i))
    if y_i == 1 and prediction >= 0.5: # tp: paid and predicted paid
        tp += 1
    elif y_i == 1:                     # fn: paid and predicted unpaid
        fn += 1
    elif prediction >= 0.5:            # fp: unpaid and we predict paid
        fp += 1
    else:
        tn += 1                        # tn: unpaid and predicted unpaid
        

# +
precision = tp / (tp + fp)
recal = tp/ (tp + fn)

precision, recal
# precision when we predicted paid account we were right 75% of times
# recall: when a user has a paid account 80% of the time we were right. 
# -

predictions = [logistic(dot(theta, x_i)) for x_i in x_test]
plt.scatter(predictions, y_test, marker= '+')
plt.xlabel("predicted probability")
plt.ylabel("actual outcome")
plt.title("Logistic regression Predicted vs. Actual")

# Small dive into support vector machines, hyperplanes and kernel trick but not implemented as not suitable for a from scratch routine. 


