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

# Frequently when doing data science, we’ll be trying to the find the best model for a
# certain situation. And usually “best” will mean something like “minimizes the error
# of the model” or “maximizes the likelihood of the data.” In other words, it will represent
# the solution to some sort of optimization problem.

from linear_algebra import Vector, dot


# For functions like ours, the gradient (this is the vector
# of partial derivatives) gives the input direction in which the function most quickly
# increases.

# One approach is to maximizing the above function is to
#
# - pick a random starting point, 
# - compute the gradient, 
# - take a small step in the gradient direction and repeat
#

# #### Estimating the Gradient and know as gradient checking

# Take the function below with one variable. It's derivative at a point x measures how f(x) changes when a make a small change to f(x). 

def sum_of_squares(v: Vector) -> float:
    """ Computes the sum of squared elements in v """
    return dot(v, v)


# The derivative is defined as the slope pf the tangent line at (x, f(x)), while the difference quotiente is the slope of the not-quite-tangent line that runs through (x+h, f(x+h)). 
#
# As the step h gets smaller and smaller, the not so quite tangent line gets coloser and closer. Defining below the not so tangent line approximation of the derivative of f(x).
#

from typing import Callable


def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


# For many functions is easy to claculate the derivates. Example below for the derivate of the square function.

def square(x: float) -> float:
    return x * x


def derivative(x: float) -> float:
    return 2 * x


# Now lets estimate the derivatives by evaluating the difference quotient for a small step e. 

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001)
            for x in xs]
# plot to show they are basically the same value
# %matplotlib inline
import matplotlib.pyplot as plt
plt.title("Actual square derivative vs estimates")
plt.plot(xs, actuals, 'rx', label = 'Actuals')
plt.plot(xs, estimates, 'b+', label = 'Estimates')
plt.legend(loc=9)


# When f is a function of 2 or more variables, it has many *partial derivatives*; each indicating how f changes when we make small changes in just one of the input variables

def partial_difference_quotient(f: Callable[[Vector], float], v: Vector, i: int, h: float) -> float:
    """ Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j==i else 0) # add h to just the ith element of v
        for j, v_j in enumerate(v)]
    return (f(w) - f(v)) /h


def estimate_gradient(f: Callable[[Vector], float], v: Vector, h: float = 0.0001) -> Vector: 
    return [partial_difference_quotient(f, v, i, h)
           for i in range(len(v))]                                          


# The above example is computtatinly expensive, as per each feature we woudl have to compute 2n operations to calculate the tangent line that approximates to the partial derivative. Math derivatives to be used beyond this point. Sum of squares gradient as an example. 

# #### Using the Gradient

# From intuition the function sum_of_squares will be at it's minumum value if the input is a vector of zeroes for all features. 
# Let's prove the below based on the function 

import random
from collections import deque
from linear_algebra import distance, add, scalar_multiply


# +
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Updates the theta parameter after one epoch
     Moves step_size in the gradient direction from 'v'"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# pick a random staring point

v = [random.uniform(-10, 10) for i in range(3)]

last_5_epochs = deque([],5) # using deque to store last 5, not used
for epoch in range(1000):
    grad = sum_of_squares_gradient(v) # compute the gradient at v
    v = gradient_step(v, grad, -0.01) # take a negative learning rate, to optmize minimum
    print(epoch, v)
    last_5_epochs.append(v)
#print(last_5_epochs)

assert distance(v, [0, 0, 0]) < 0.001 # v should be close to 0
# -

# Using Gradient Descent to fit models
#
# - we will have some data set 
# - and some hypothesized function for the data depending or one or more features
# - we will also have a loss function that measures how well the model fits our data
#
# If you assume your data as being fixed; then your loss function tells us how good or bad any particular model parameters are. 
# We will use the gradient descent to find th eparamters that minimize the loss function:
#
# Let's start with an example below:

# +
# x ranges from -50 to 49 and y is always 20 * x + 5
inputs = [(x, 20 *x + 5) for x in range(-50, 50)]

# the function below determines the gradient based on the error from a single data point

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept # the single point prediction of the model
    error = (predicted - y) # error is (predicted - actual)
    squared_error = error ** 2 # we will minimize the square error
    grad = [ 2 * error * x,  2 * error] # using this gradient; 
    return grad


# -

# Now the above computation was for a single data point. For the whole dataset we'll look at the mean squared error. And the gradient of the mean squared error is just the mean of the individual gradients. 
#
# we're going to :
#
# - start with a random value for theta
# - compare the mean of the gradients
# - adjust theta in that direction
# - Repeat
#
# After several epochs (each pass through the dataset) we shoudl learn something approxing the correct parameters. Remmeber we know the correct parameters of theta for comparison. In the below the algorithm will learn from y.
#

# +
from linear_algebra import vector_mean

# Start with random values for slope and intercept
# remmeber we are aiming at a slope around 20 and intercept at 5
theta = [random.uniform(-1, 1), random.uniform(-1, 1)] 
# Set the learning rate
learning_rate = 0.001

for epoch in range(5000):
    grad = vector_mean([linear_gradient(x, y, theta) 
                        for x, y in inputs])  # here grad passes all data points (x,y) -> one batch pass per epoch
                                              # returns 1 x 2 vector gradient
    
    # take a step in that direction
    theta = gradient_step(theta, grad, - learning_rate)  # updates theta parameters
    print(epoch, theta)

slope, intercept = theta
assert 19.99 < slope < 20.01
assert 4.99 < intercept < 5.01
# -

# One Setback of this type of the batch gradient descent is we need to evaluate the whole dataset at the grad step prior to take a gradient step
# this would be prohibitive with larger datasets

# #### Minibatch and Stochastic Gradient Descent

# Minibatch gradient descent technique we compute the gradient and take a gradient step based on a batch of data

from typing import TypeVar, List, Iterator

# +
T = TypeVar('T') # this allows us to type generic functions more in the book pag 108

def minibatches(dataset : List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    """Generates 'batch_size' minibatches from the dataset"""
    # start indexes 0, batch_size, 2 * batch_size...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    
    if shuffle: random.shuffle(batch_starts) # shuffle the batches
    for start in batch_starts:
        end = start + batch_size
        #print(f"{start}:{end}") # assist in viewing the minibatches indexes
        yield dataset[start:end]


# -

# Now we can solve the same problem wth the minibatches

# +
theta = [random.uniform(-1,1), random.uniform(-1,1)]

for epoch in range(1000): # for each epoch we update theta len(datase)/ batch_size; 5 in this case
    for batch in minibatches(inputs, batch_size = 20):
        grad = vector_mean([linear_gradient(x, y, theta)
                           for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1 , "slope should be around 20"
assert 4.9 < intercept < 5.1, "intercept should be around 5"
# -
# #### Stochastic Gradient Descent  

# Another variation is stochastic gradient descent, where gradient steps are taken based on one training example at a time

# +
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

random.shuffle(inputs) #not on book; input observations need to be shuffled
for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta

slope, intercept = theta
assert 19.9 < slope < 20.1 , "slope should be around 20"
assert 4.9 < intercept < 5.1, "intercept should be around 5"
# -












