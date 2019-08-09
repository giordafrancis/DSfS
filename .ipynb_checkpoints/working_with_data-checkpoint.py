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
def bucketize(point: float, bucket_size: float)


