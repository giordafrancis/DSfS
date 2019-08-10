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

# +
from linear_algebra import dot, Vector

def predict(x: Vector, theta: Vector) -> float:
    """assumes that first element of x is 1"""
    return dot(x, theta)


# -

# Further assumptions of the Least Squares Model:
#
# - the first is that features of vector X are linearly independent;meaning there is no way to write any one as a weighted sum of some of the others. It this assumtion fails it's is impossible to correctly estimate theta
#
# - the second assumption is that the features of X are all uncorrelated with the errors E. If this fals to be the case, our estimate theta will systematiclly be incorrect
#
# pag 191 to 193 have more details on this. Also more detail in this [article](https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/)

# #### Fitting the model

# +
from typing import List
from linear_algebra import Vector

def error(x: Vector, y: float, theta: Vector) -> float:
    return predict(x, theta) - y

def squared_error(x: Vector, y: float, theta: Vector) -> float:
    return error(x, y, theta) ** 2

x = [1, 2, 3]
y = 30
theta = [4, 4, 4] # so prediction = 4 + 8 + 12 = 24

assert error(x, y, theta) == -6
assert squared_error(x, y, theta) == 36


# +
def sqerror_gradient(x: Vector, y: float, theta: Vector) -> Vector:
    err = error(x, y, theta)
    return [2 * err * x_i for x_i in x]

assert sqerror_gradient(x, y, theta) == [-12, -24, -36]
# -

# Using gradient descent we can know compute the optimal theta. first lets write a least_squares_fit function that can work with any dataset:
#

import random
from linear_algebra import vector_mean
from gradient_descent import gradient_step
from multiple_linear_regression_data import inputs
from statistics import num_friends_good, daily_minutes_good


def least_squares_fit(xs: List[Vector], ys: List[float], learning_rate: float = 0.001, num_steps: int = 1000, batch_size: int = 1) -> Vector:
    """
    Find the theta that minimizes the sum of squared errors assuming the model y = dot(x, theta)
    """
    # start with a random guess
    guess = [random.random() for _ in xs[0]]
    for epoch in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                   for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, - learning_rate)
        print(f'epoch is {epoch}; current guess is {guess}')
    return guess  


# +
random.seed(0)

learning_rate = 0.001

theta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
# -

# minutes= 30.58 + 0.972 friends -1.87 work hours + 0.923 phd
assert 30.50 < theta[0] < 30.70
assert 0.96 < theta[1] < 1.00 
assert -1.89 < theta[2] < -1.85
assert 0.91 < theta[3] < 0.94

# You should think of the coefficients of the model as representing all-else-being-equalestimates  of  the  impacts  of  each  factor.  All  else  being  equal,  each  additional  friendcorresponds to an extra minute spent on the site each day. All else being equal, eachadditional hour in a user’s workday corresponds to about two fewer minutes spent onthe  site  each  day.  All  else  being  equal,  having  a  PhD  is  associated  with  spending  an extra minute on the site each day.
#
# What this doesnt capture is interactions between features. It's possible works hours effect is sifferent with people with many friends. One way to handle this is to introduce a new variable with the product of friends and work hours. 
#
# Or  it’s  possible  that  the  more  friends  you  have,  the  more  time  you  spend  on  the  siteup  to  a  point,  after  which  further  friends  cause  you  to  spend  less  time  on  the  site.(Perhaps  with  too  many  friends  the  experience  is  just  too  overwhelming?)  We  couldtry  to  capture  this  in  our  model  by  adding  another  variable  that’s  the  square  of  thenumber of friends.
#
# Once we start adding varaibles we need to worry about weather their coefficients matter. There are no limits to the numbers of products, logs, squares and high powers that can be added. 

# #### Goodness of fit 

from simple_linear_regression import total_sum_squares


def multiple_r_squared(xs: List[Vector], ys:Vector, theta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, theta) ** 2
                                for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_squares(ys)


assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, theta) < 0.68

# R squared tends to increase the more varables are added to the model. Because of this, in a multiple regression, we also need to look at the standard errors ofthe  coefficients,  which  measure  how  certain  we  are  about  our  estimates  of  each  theta_i.
# The regression as a whole may fit our data very well, but if some of the independentvariables are correlated (or irrelevant), their coefficients might not mean much.The typical approach to measuring these errors starts with another assumption—that the errors **εi** are independent normal random variables with mean 0 and some shared(unknown) standard deviation σ. In that case, we (or, more likely, our statistical soft‐ware) can use some linear algebra to find the standard error of each coefficient. Thelarger it is, the less sure our model is about that coefficient. Unfortunately, we’re notset up to do that kind of linear algebra from scratch.

# Digression: The bootstrap
#
# - used below as an estimate to error coeficients for features
#

from typing import TypeVar, Callable

# +
X = TypeVar('X') # Generic type data
Stat = TypeVar('Stat') # Generic type for 'statistic'

def bootstrap_sample(data: List[X]) -> List[X]:
    """ randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X],
                        stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
    """ evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


# -

# 101 points very close to 100
close_to_100 = [99.5 + random.random() for _ in range(101)]
# 101 points, 50 near 0 , 50 near 200
far_from_100 = ([99.5 + random.random()] + [random.random() for _ in range(50)] + [200 + random.random() for _ in range(50)])

from statistics import median, standard_deviation
# both medians are very close values although distribution 
median(close_to_100), median(far_from_100)

# +
# if we compute the bootstrap_statistic

medians_close = bootstrap_statistic(close_to_100, median, 100)
medians_far = bootstrap_statistic(far_from_100, median, 100)

assert standard_deviation(medians_close) < 1
assert standard_deviation(medians_far) > 90
# -

# #### Standard Errors of regression coeficients
#
# We  can  take  the  same  approach  to  estimating  the  standard  errors  of  our  regression coefficients.  We  repeatedly  take  a  bootstrap_sample  of  our  data  and  estimate  theta based on that sample. If the coefficient corresponding to one of the independent vari‐ables (say num_friends) doesn’t vary much across samples, then we can be confident that  our  estimate  is  relatively  tight.  If  the  coefficient  varies  greatly  across  samples,then we can’t be at all confident in our estimate.

from typing import Tuple
import datetime


def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]
    theta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
    print("bootstrap sample", theta)
    return theta


random.seed(0) 
# takes really long to run; run once saved at .py file as an instance
#bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)), 
# estimate_sample_beta, 100)
from multiple_linear_regression_data import bootstrap_betas

#  now we can estimate the standard deviation of each coefficient 
bootstrap_standard_errors = [
    standard_deviation([beta[i] for beta in bootstrap_betas])
    for i in range(len(bootstrap_betas[0]))
]

bootstrap_standard_errors

theta

# +
# now compute the p-values with the approximation to normal_cdf as it gets closer to a 
# t- distributuion for a large number of degrees of freedom ( hard to implement from scratch)

from probability import normal_cdf
# not fully undesrtood the p- value calc below. but good example of accessing 
# coeficients errors 

def p_value(beta_hat_j: float, sigma_hat_j: float)-> float:
    if beta_hat_j > 0: # twice the probability an even *larger* value
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:# otherwise twice the probability of seeing a *smaller* value
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


# +
# based on my theta values and the bootstrap standard errors

assert p_value(30.51, 1.27) < 0.001
assert p_value(0.975, 0.10 ) < 0.001
assert p_value(-1.851, 0.155) < 0.001
assert 0.45 < p_value(0.914, 1.249) < 0.47 # phd theta_3



# -

# Most coeficients have very small p-values (suggesting they are indeed non-zero). The phd coeficient is different then 0 meaning is  random rather then meaningful.

p_value(0.914, 1.249) 


# ## Regularization
#
# Regularization  is  an  approach  in  which  we  add  to  the  error  term  a  penalty  that  getslarger  as  beta  gets  larger.  We  then  minimize  the  combined  error  and  penalty.  Themore importance we place on the penalty term, the more we discourage large coeffi‐cients.For  example,  in  ridge  regression,  we  add  a  penalty  proportional  to  the  sum  of  thesquares  of  the  beta_i.  (Except  that  typically  we  don’t  penalize  beta_0,  the  constant term.)

# ### Ridge regression

# +
# alpha is the tuning parameter aka lambda

def ridge_penalty(theta: Vector, alpha: float) -> float:
    return alpha * dot(theta[1:], theta[1:]) # theta_0 not regularized 

def squared_error_ridge(x: Vector, y: float, theta: Vector, alpha: float) -> float:
    """ estimate error plus ridge penalty on theta"""
    return error(x, y, theta) ** 2 + ridge_penalty(theta, alpha)



# +
# now lets plug this to gradient descent

from linear_algebra import add

# different then Andrew ng overal update to theta
def ridge_penalty_gradient(theta: Vector, alpha: float) -> Vector:
    """ Gradient of just the ridge penalty"""
    return [0.] + [2 * alpha * theta_j for theta_j in theta[1:]]

def sqerror_ridge_gradient(x: Vector, y: float, theta: Vector, alpha: float) -> Vector:
    """
    The gradient corresponding to the ith squared error term including the ridge penalty
    """
    return add(sqerror_gradient(x, y, theta), ridge_penalty_gradient(theta, alpha))



# -

def least_squares_fit_ridge(xs: List[Vector], ys: List[float], alpha: float = 0.0, 
                            learning_rate: float = 0.001, num_steps: int = 1000, batch_size: int = 1) -> Vector:
    """
    Find the theta that minimizes the sum of squared errors assuming the model y = dot(x, theta) using ridge regularization
    """
    # start with a random guess
    guess = [random.random() for _ in xs[0]]
    for epoch in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
                                   for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, - learning_rate)
        #print(f'epoch is {epoch}; current guess is {guess}')
    return guess  


random.seed(0)
theta_t0 = least_squares_fit_ridge(inputs, daily_minutes_good,0.0, learning_rate, 5000, 25)
theta_t0

# as we increase the alpha the goodness of fit gets worst but hte size of theta gets smaller
theta_t1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1, learning_rate, 5000, 25)
theta_t1

# as we increase the alpha the goodness of fit gets worst but the size of theta gets smaller, parameters move towards 0
theta_t2 = least_squares_fit_ridge(inputs, daily_minutes_good, 1, learning_rate, 5000, 25)
theta_t2

# in particular theta_3 phd vanishes which is line line with the previous result as it wasnt significantly different from 0; p value bigger then 0.05

theta_t3 = least_squares_fit_ridge(inputs, daily_minutes_good, 10, learning_rate, 5000, 25)
theta_t3


# #### Lasso regression

# +
# another approach is the lasso regression which uses the penalty:

def lasso_penalty(theta, alpha):
    return alpha * sum(abs(theta_i) for theta_i in theta[1:])
    
# -

# Whereas  the  ridge  penalty  shrank  the  coefficients  overall,  the  lasso  penalty  tends  to force  coefficients  to  be  zero,  which  makes  it  good  for  learning  sparse  models.Unfortunately,  it’s  not  amenable  to  gradient  descent,  which  means  that  we  won’t  beable to solve it from scratch






