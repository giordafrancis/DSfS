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

# #### Measures of central tendencies

from typing import List

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


# +
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

assert mean([1,1,1]) == 1
assert mean([5,3]) == 4


# -

# underscore is used to denote a private function
def _median_odd(xs:List[float]) -> float:
    """if len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]


def _median_even(xs:List[float]) -> float:
    """If len(xs) is even, it's the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    low_midpoint = hi_midpoint - 1
    return (sorted_xs[hi_midpoint] + sorted_xs[low_midpoint]) / 2


# +
def median(v: List[float]) -> float:
    """Finds the 'middle-most value of v"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)
    

assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2


# +
def quantile(xs: List[float], p: float) -> float:
    """Returns the pth percentile vakue in x"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

assert quantile(num_friends, 0.10) == 1
assert quantile(num_friends, 0.25) == 3
assert quantile(num_friends, 0.75) == 9
assert quantile(num_friends, 0.90) == 13
# -

from collections import Counter


# +
def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more then one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
           if count == max_count]

assert set(mode(num_friends)) == {1,6}


# + {"active": ""}
# Dispersion

# +
def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

assert dat
# -
















