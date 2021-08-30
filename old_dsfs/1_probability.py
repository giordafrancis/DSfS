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

import enum, random


# Given a family of two kids , What is the probability of the event both children are girls, conditional on the event "at least one of the children is a girls(L)

# +
# more on class Enum in the book
class KID(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid():
    return random.choice((KID.GIRL, KID.BOY))


# +
random.seed(0)

both_girls = 0
older_girl = 0
either_girl = 0

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == KID.GIRL:
        older_girl += 1
    if younger == KID.GIRL and older == KID.GIRL:
        both_girls += 1
    if younger == KID.GIRL or older == KID.GIRL:
        either_girl += 1        
# -

print("P(both|older):", both_girls/ older_girl)
print("P(both|either):", both_girls/ either_girl)


def uniform_pdf(x: float) -> float:
    return 1 if 0<= x < 1 else 0


def uniform_cdf(x: float) -> float:
    """Returns the probability that a uniform random variable is <=x"""
    if x < 0: return 0    # uniform random is never less than 0
    elif x < 1 : return x # e.g. P(X <= 0.4)
    else: return 1        # uniform random is always less than 1


# +
import math

SQRT_TWO_PI = math.sqrt(2 * math.pi)
def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))


# +
import matplotlib.pyplot as plt
# %matplotlib inline

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0,sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0,sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu=0,sigma=0.5')
plt.plot(xs, [normal_pdf(x,mu=-1, sigma=1) for x in xs], '-.', label='mu=-1,sigma=1')
plt.legend();
plt.title("Various Normal pdfs");

# -
# The normal cdf is the probaility the variable is below a threshold

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


plt.plot(xs, [normal_cdf(x) for x in xs], '-', label="mu=0,sigma=1")
plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label="mu=0,sigma=2")
plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label="mu=0,sigma=0.5")
plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.', label="mu=0,sigma=1")
plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs");


# Sometimes  we’ll  need  to  invert  normal_cdf  to  find  the  value  corresponding  to  aspecified probability. 
# There’s no simple way to compute its inverse, but normal_cdf is continuous and strictly increasing, so we can use a binary search:
#
# Binary search compares the target value to the middle element of the array. If they are not equal, the half in which the target cannot lie is eliminated and the search continues on the remaining half, again taking the middle element to compare to the target value, and repeating this until the target value is found.

def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.00001) -> float:
    """Find approximate inverse using binary search"""
    # if not standard, compute standard and rescale 
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z = -10.0 # normal_cdf(-10) is very close to 0
    hi_z = 10.0  # normal_cdf(10) is very close to 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2 # Consider the midpoint
        mid_p = normal_cdf(mid_z)   # and the cdf's valu there
        
        if mid_p < p:
            low_z = mid_z   # Midpoint too low, search above it
        else:
            hi_z = mid_z    # Midpoint too high, search below it
    return mid_z


assert -0.0001 < inverse_normal_cdf(.5) < 0.0001


# Central Limit theorem

# One  reason  the  normal  distribution  is  so  useful  is  the  central  limit  theorem,  whichsays  (in  essence)  that  a  random  variable  defined  as  the  average  of  a  large  number  ofindependent and identically distributed random variables is itself approximately nor‐mally distributed.

# +
def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))



# -

# The mean of a Bernoulli(p) variable is p, and its standard deviation is sqrt(p(1 - p)). Thecentral  limit  theorem  says  that  as  n  gets  large,  a  Binomial(n,p)  variable  is  approxi‐mately   a   normal   random   variable   with   mean   μ= n* p   and   standard   deviation σ= sqrt(n * p(1 − p)).`

binomial(10, 0.1)

from collections import Counter


def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a Binomial(n, p) and plot their histogram"""
    
    data = [binomial(n, p) for _ in range(num_points)]
    
    # use bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()], 
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    
    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
         for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial distribution vs. Normal Approximation")    


binomial_histogram(0.75, 100, 10000)






