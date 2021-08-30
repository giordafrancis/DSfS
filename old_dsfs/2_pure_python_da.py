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

# #### Exploring one-dimensional data

# The simplest case is when you have a one-dimensional data set, which is just a collection
# of numbers. For example, these could be the daily average number of minutes
# each user spends on your site, the number of times each of a collection of data science
# tutorial videos was watched, or the number of pages of each of the data science books
# in your data science library.
# An obvious first step is to compute a few summary statistics. You’d like to know how
# many data points you have, the smallest, the largest, the mean, and the standard deviation.
# But even these don’t necessarily give you a great understanding. A good next step is to
# create a histogram, in which you group your data into discrete buckets and count how
# many points fall into each bucket:

# +
from typing import List, Dict
from collections import Counter
import math

import matplotlib.pyplot as plt


# +
def bucketize(point: float, bucket_size: float) -> float:
    """ Flow the point to the next multiple of bucket_size"""
    return bucket_size * math.floor(point/ bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)


# -

# Lets consider the below two sets of data

import random
from probability import inverse_normal_cdf, normal_pdf
random.seed(0)

# uniform distribution between -100 and 100
uniform = [200 * random.random() - 100 for _ in range(10_000)]

# normal distribution with mean 0 , standard deviation 57
normal = [57 * inverse_normal_cdf(random.random())
         for _ in range(10_000)]

plot_histogram(uniform, 10, "Uniform Histogram")

plot_histogram(normal, 10, "Normal Histogram")


# #### Two dimensions

# Now imagine you have a data set with two dimensions. Maybe in addition to daily
# minutes you have years of data science experience. Of course you’d want to understand
# each dimension individually. But you probably also want to scatter the data.

# +
def random_normal() -> float:
    """Returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(10_000)]
# -

ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plot_histogram(xs, 10,  "ys1")
#plot_histogram(ys2, 10, "ys1")

plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different Joint Distributions")
plt.show()

# +
from statistics import correlation

print(correlation(xs, ys1))
print(correlation(xs, ys2))
# -

# #### Using Named Tuples

from collections import namedtuple
import datetime

StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
example_stock = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert example_stock.symbol == 'MSFT'
assert example_stock.closing_price == 106.03

# Like regular tuples, namedtuples are immutable, meaning they cannot modify their values once created. Occassionaly this will get in your way, but mostly is a good thing

from typing import NamedTuple


# A great way o use NamedTuples as classes and create methods
class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """As it is a class methods can be added"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']



example_stock2 = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

example_stock2.is_high_tech()

# Dataclasses

from dataclasses import dataclass


# The systax is very similar to NamedTuple. But instead of inheriting from a base class, we use a decorator

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float
    
    def is_high_tech(self) -> bool:
        """As it is a class methods can be added"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


example_stock3 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)

assert example_stock3.symbol == 'MSFT'
assert example_stock3.closing_price == 106.03
assert example_stock3.is_high_tech()

# the big difference is we can modify a dataclass instance values:

# stock split closing price
example_stock3.closing_price /= 2
assert example_stock3.closing_price == 53.015

example_stock3.closing_price = 106.03

# +
# unable set/modify the attribute in the named tupel instance
# example_stock2.closing_price /= 2
# -

# #### Cleaning and Munging

from dateutil.parser import parse
# clunky useing date util parser instead
#datetime.datetime.strptime('2018-04-12', "%Y-%m-%d")
parse('2018-04-12')


def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice2(symbol=symbol,
                      date=parse(date).date(),
                      closing_price=closing_price)


parse_row(['MSFT', "2018-12-14", "106.03"])

# What if we had issues regarding our data? bad data? how best to handle exceptions 

from typing import Optional
import re


def try_parse_row(row: List[str]) -> Optional[StockPrice2]:
    symbol, date_, closing_price_ = row
    
    # Stock symbol should be all capital letters
    if not re.match(r"^[A-Z]+$", symbol):
        return None
    try:
        date = parse(date_).date()
    except ValueError:
        return None
    
    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None

    return StockPrice2(symbol, date, closing_price)


# Should return None for Errors
assert try_parse_row(['MSFTa', "2018-12-14", "106.03"]) is None
assert try_parse_row(['MSFT', "2018-12--14", "106.03"]) is None
assert try_parse_row(['MSFTa', "2018-12-14", "x"]) is None

try_parse_row(['MSFT', "2018-12-14", "106.03"])

# #### We can now attempt to read comma separeted values with bad data and return only the valid rows:

stock_string = """AAPL,6/20/2014,90.91
MSFT,6/20/2014,41.68
FB,6/20/3014,64.15
AAPL,6/19/2014,91.86
MSFT,6/19/2014,10
FB,6/19/2014,64.34
MSFT,6/21/2014,n/a
FB,6/21/3014,61.15
AAPL,6/21/2014,92.82
MSFT,6/22/2014,42.11
FB,6/22/2014,64.89
MSFT,6/22/2014,41.58
FB,6/23/3014,64.16
AAPL,6/23/2014,91.84
MSFT,6/23/2014,42.00
FB,6/24/2014,64.39
MSFT,6/24/2014,41.85
FB,6/24/3014,64.17
AAPL,6/25/2014,91.84
MSFT,6/25/2014,41.60
FB,6/25/2014,64.31
AAPL,7/20/2014,90.91
MSFT,7/20/2014,41.68
FB,7/20/3014,64.15
AAPL,7/19/2014,91.86
MSFT,7/19/2014,10
FB,7/19/2014,64.34
MSFT,7/21/2014,n/a
FB,8/21/3014,61.15
AAPL,8/21/2014,92.82
MSFT,8/22/2014,42.11
FB,8/22/2014,64.89
MSFT,8/22/2014,41.58
FB,8/23/3014,64.16
AAPL,8/23/2014,91.84
MSFT,9/23/2014,42.00
FB,9/24/2014,64.39
MSFT,9/24/2014,41.85
FB,9/24/3014,64.17
AAPL,9/25/2014,91.84
MSFT,9/25/2014,41.60
FB,9/25/2014,64.31
"""

import csv

# create a new file
with open('data/comma_delimited_stock_prices.txt', 'w') as f:
    f.write(stock_string)

# +
data: List[StockPrice] = []
    
with open('data/comma_delimited_stock_prices.txt') as f:
    reader = csv.reader(f)
    for row in reader:
        maybe_stock = try_parse_row(row)
        if not maybe_stock:
            print(f"skipping invalid rows: {row}")
        else:
            data.append(StockPrice(*row))
# year "3014" was not caught! not an exception by parser
# -

# Now how can we calcualte the percentage change per day? 

# +
# lets collect the prices by symbol
from collections import defaultdict

prices: Dict[str, List[StockPrice]] = defaultdict(list)
    
def append_stock(stock: StockPrice) -> StockPrice:
    return StockPrice(stock.symbol, parse(stock.date), float(stock.closing_price))

for sp in data:
    prices[sp.symbol].append(append_stock(sp))

prices = {symbol: sorted(symbol_prices, key= lambda sp: sp.date )
          for symbol, symbol_prices in prices.items()}
# -

prices


# Which we can use to compute a sequence ofday-over-day changes:

# +
def pct_change(yesterday: StockPrice2, today: StockPrice2) -> float:
    return round(today.closing_price / yesterday.closing_price - 1, 3)

class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float

def day_over_day_changes(prices: List[StockPrice2]) -> List[DailyChange]:
    """
    Assumes prices are for one stock and are in order
    """
    return [DailyChange(symbol=today.symbol,
                       date=today.date,
                       pct_change=pct_change(yesterday,today))
           for yesterday, today in zip(prices, prices[1:])]



# -

all_changes = [change
              for symbol_prices in prices.values()
              for change in day_over_day_changes(symbol_prices)]

all_changes

# What is the maximum and minimun daily change?

max_change = max(all_changes, key= lambda change: change.pct_change)
assert max_change.symbol == "MSFT"

min_change = min(all_changes, key = lambda change: change.pct_change)
min_change

all_changes

# +
# changes_by_month: List[DailyChange] = {month: [] for month in range(1,13)}
changes_by_month: Dict[int, List[DailyChange]] = defaultdict(list)

for change in all_changes:
    if change.date.month:
        changes_by_month[change.date.month].append(change)
# -

avg_daily_change = {month:sum(change.pct_change for change in changes)/ len(changes) 
                    for month, changes in changes_by_month.items()}
avg_daily_change

# July is the best_month
assert avg_daily_change[7] == max(avg_daily_change.values())

# #### Rescaling
#
# When dimensions arent comparable with none another we will sometimes rescale our data so that each dimension has mean 0 and standard deviation 1. this effectivily gets reid of the units, converting each dimension to "standard deviations from the mean"

# +
from linear_algebra import distance, vector_mean, Vector
from statistics import standard_deviation

from typing import Tuple


# -

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """Returns the mean and the standard deviation for each position"""
    dim = len(data[0])
    
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data]) 
             for i in range(dim)]
    return means, stdevs


vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors) 

assert means == [-1.0, 0.0, 1.0]
assert stdevs == [2.0, 1.0, 0.0]


# We can use them to create a new dataset:
#

def rescale(data: List[Vector]) -> List[Vector]:
    """Rescales the input data so that each postion has mean 0 and standard deviation 1. 
    (leaves positon as is if its standard deviation is 0)
    """
    dim = len(data[0])
    means, stdevs = scale(data)
    
    # Make a copy of each vector
    rescaled = [v[:] for v in data]
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i] # means and stdevs are vectors with len(dim)
    return rescaled


# checking if it soes it it says
means, stdevs = scale(rescale(vectors))
assert means == [0.0, 0.0, 1.0] 
assert stdevs == [1.0, 1.0, 0.0]

# #### Dimesionality Reduction

# PCA: a techniques used to extract one or more dimensions that capture as much of teh variation in the data as possible

from pca_data import pca_data

pca_data[:5]

# +
# 1st step is to rescale the data by subtracting mean ( or fully normalise)
# if we dont do this our techniques are likely to identify the mean itself rather then the variation (95 to 99% var needs to be retained)
from linear_algebra import subtract

def de_mean(data: List[Vector]) -> List[Vector]:
    """Recenters the data to have mean 0 in every dimension"""
    
    mean= vector_mean(data)
    return [subtract(vector, mean) for vector in data]


# -
# Given a demeaned matrix X, what is the direction that captures the greates variance in the data
#

# +
from linear_algebra import magnitude, dot


# given d direction d (vector of magnitude 1)
# direction each row x in te ematrix extenders dot(x, d) in the d direction
# every nonzero vector below determines a direction if we rescale it to have magnitude 1

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]


# we can compute the variance of our dataset in teh direction determine by w

def directional_variance(data: List[Vector], w: Vector) -> float:
    """
    Returns the variance of x in the direction of w
    """
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)

# we would like to find the direction that maximizes this variance. We can use gradient descent,
# once we have defined the gradient function

def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """
    The gradient of directional variance with respect to w
    """
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data)
                for i in range(len(w))]

# now the first compomenent that we have is just the direction that maximizes the directional_variance function

from gradient_descent import gradient_step

def first_pricipal_component(data:List[Vector], n:int=100, step_size:float=0.1) -> Vector:
    # start with a random guess
    guess = [1.0 for _ in data[0]]
    
    for _ in range(n):
        dv = directional_variance(data, guess)
        gradient = directional_variance_gradient(data, guess)
        guess = gradient_step(guess, gradient, step_size)
    
    return direction(guess)
                


# -

pca_data_demean = de_mean(pca_data)
first_pricipal_component(pca_data_demean)
# see page 148 pic in book
# the below coordenates appear to capture the primary axis along which our demeaned data varies

# +
from linear_algebra import scalar_multiply

# once the above direction for the principal component we can project the data onto it
# to find the values of that component

def project(v: Vector, w: Vector) -> Vector:
    """projects v onto w and subtracts the result from v"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)

# if we want to find further components, we first remove the projections from the data

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """projects v onto w and subtracts result from v"""
    return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]


# -


# On a higher dimension dataset, we can iteratively find as many componets as we want:

# +
def pca(data: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vectors] = []
    for _ in range(num_components):
        component = first_pricipal_component(data)
        components.append(component)
        data = remove_projection(data, component)
    return components

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]

# -


