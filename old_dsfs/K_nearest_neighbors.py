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

# Nearest neighbours is one of the simplest predictive models. It makes no mathematical assumptions and it doesn't required any sort of heavy machinery. the only things it requires are:
#
# - Some notion of distance
# - An assumption that points are close to one another are similar
#
# That simple assumption also narrows down the understanding of the hypothesis behind the data. Any new points will be classified based on the distance notion above not what causes the classification.
#
# the algorithm trains all labels based on training; one a test set is reviewed it's distance is compared to all training set data then ordered by smallest distance and finally a vote takes place (counter) to ascertain who as the majority of labels within the K smallest distances - that assigns the labelling.
#
#
# from scikit learn doc
# https://scikit-learn.org/stable/modules/neighbors.html
#
# The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data 
#
# Note: assigning the number of K required was not covered. Often is done via  a validation set. 
#
# The curse of dimensionality was also covered : KNN pitfall as it runs into problems in higher dimensions. The higher the dim the vastest is the space.  If KNN is to be used in higher dimensions consider dimensionality reduction first. 
#
# Pag 170 to 173 covers the curse of dimensionality with examples. 
#

# +
from typing import List
from collections import Counter

def majority_vote(labels: List[str]) -> str:
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])


# -

assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'

from typing import NamedTuple
from linear_algebra import Vector, distance


# +
class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:
    # Order the labelled points from nearest to farthest
    by_distance = sorted(labeled_points, key= lambda lp: distance(lp.point, new_point))
    # find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    
    return majority_vote(k_nearest_labels)


# -

# Example: The iris Dataset
#
# We will try to build a model that can preditc the class (species) from the firsts four measurements. Our nearest neighbors function expects a LabeledPoint so lets represent our data that way:
#

# +
import requests

data = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

data
# -

with open('iris.dat', 'w') as f:
    f.write(data.text)

from typing import Dict
import csv
from collections import defaultdict

data.text[:100]


# +
def parse_iris_row(row: List[str]) -> LabeledPoint:
    """ sepal_length, sepal_width, petal_length, petal_width, class
    """
    if row:
        measurements = [float(value) for value in row[:-1]]
        # print(row)
        # class is e.g. "Iris-virginica"; we just want virginica
        label = row[-1].split("-")[-1]
        return LabeledPoint(measurements, label)

with open('iris.dat') as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader if row]
# -

iris_data[:3]

# +
# we'll also group the points by species/ label for ploting

points_by_species : Dict[str, List[Vector]] = defaultdict(list)

for iris in iris_data:
    points_by_species[iris.label].append(iris.point)
# -

# Lattice plot done in the book pag 168; skipped but good example for matplotlib dive in

# +
import random
from machine_learning import split_data

random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.70)
# -

# training set will be the neighbors that we will use to classify the points in  the test set.
# we will need to choose the k 
# in a real aplication a validation set is used to determine this
# here just use k = 5
#

# +
from typing import Tuple

# track how many times we see (predictic, actual)

confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
defaultdict(int)
num_correct = 0

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)
    actual = iris.label
    
    confusion_matrix[(predicted, actual)] += 1
    if predicted == actual:
        num_correct += 1
            
# -

pct_correct = num_correct/ len(iris_test)
print(pct_correct, confusion_matrix)

# +
#iris =  LabeledPoint(point=[6.3, 3.3, 4.7, 1.6], label='versicolor')
#knn_classify(5, iris_train, iris.point)

# +
#by_distance = sorted(iris_train, key= lambda lp: distance(lp.point, iris.point))
#k_nearest_labels = [lp.label for lp in by_distance[:5]]

# +
#majority_vote(k_nearest_labels)
# -


