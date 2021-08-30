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
import random

from typing import TypeVar, List, Tuple

X = TypeVar('X') # generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split the data into fractions [prob, 1 - prob]"""
    data = data[:]               # make a shallow copy
    random.shuffle(data)         # shuffle data in place 
    cut = int(len(data) * prob)  # 
    return data[:cut], data[cut:]


# +
data = [i for i in range(1000)]
train, test = split_data(data, prob = 0.75)

# check split numbers are correct
assert len(train) == 750
assert len(test) == 250

# the original data should be preserved ( in some order)
assert sorted(train + test) == data
# -

# Often we have paired input variables and output variables; we need to make sure both corresponding variables are paired together when split

# +
Y = TypeVar('Y') # generic type to represent output

def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # Generate indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    
    return ([xs[i] for i in train_idxs], # x_train
            [xs[i] for i in test_idxs], # x_test
            [ys[i] for i in train_idxs], # y_train
            [ys[i] for i in test_idxs]) # y_test


# +
xs = [x for x in range(1000)] # xs are 1..1000
ys = [2 * x for x in xs]

x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)
# -

assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

# check all data points are paired correctly
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))

# #### Correctness

# +
# see pag 158 for confusion matrix
tp = 70
tn = 981_070
fp = 4930
fn = 13_930

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total
    
assert accuracy(tp, fp, fn, tn) == 0.98114

# impressive number but beware of imbalnces classes!
# looking into precision and recall and f1score

# +
def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)

assert precision(tp, fp, fn, tn) == 0.014


# +
def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)

assert recall(tp, fp, fn, tn) == 0.005

# terrible model! combined f1 score harmonic mean that balances both precision and recall
# usually the choice of model incolves a trade-off between both

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    recall = recall(tp, fp, fn, tn)
    precision = precision(tp, fp, fn, tn)
    return 2 * precision * recall / (precision + recall)
# -






















