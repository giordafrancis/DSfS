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

# One of the simplest clustering methods is k-means, in which the number of clusters is chosen in advance, after shich the goal is to partition the inputs into sets S1, ... Sk in a way that minimizes the total sum of squared distances from each point to the mean of its assigned cluster. 
#
# We will set for an iterative algorithm, that usually finds a good clustering:
#
# - Start with a set of k-means randomly assigned, which are points in d-dimensional space
# - assign each point to the mean to which is closest centerpoint
# - if no point assigment has changed, stop and keep the clusters.
# - if some point assignment has changed, recompute the means and return to step2

from linear_algebra import Vector


# +
# helper function that dectects if any centerpoint assigmmnet has changed
def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])

assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0
# -

from typing import List
from linear_algebra import vector_mean
import random


def cluster_means(k: int, inputs: List[Vector], assignments: List[int]) -> List[Vector]:
    # clusters[i] contains the inputs whose assignment is i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        #print(input, assignment)
        clusters[assignment].append(input)
        #print(clusters)
        #break
    # if a cluster is empty, just use a ramdom point
    return [vector_mean(cluster) if cluster else random.choice(inputs)
           for cluster in clusters]


inputs = [[-1, 1], [-2,3], [-3, 4], [4, 5], [-2, 6], [0, 3]]
assignments = [0, 0, 2, 2, 2, 1]
cluster_means(6, inputs, assignments)

# +
import itertools

from linear_algebra import squared_distance


# +
# I undertand the intuition and the code main points
#

class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k #number of clusters
        self.means = None
    
    def classify(self, input: Vector) -> int:
        """return the index of the cluster closest to the input"""
        # means method woudl be already computed as claissfy method is called
        # means len == k
        return min(range(self.k),
                   key= lambda i: squared_distance(input, self.means[i]))
    def train(self, inputs: List[Vector]) -> None:
        # start with a random assignments
        assignments = [random.randrange(self.k) for _ in inputs]
        for _ in itertools.count():
            # print(assignments)
            # compute means
            self.means = cluster_means(self.k, inputs, assignments)
            #  and find new assignments
            new_assignments = [self.classify(input) for input in inputs]
            # check how many assignments have changed and if we're done
            num_changed = num_differences(assignments, new_assignments)
            if num_changed == 0: 
                return
            # otherwise keep the new assignments, and compute new means
            assignments = new_assignments
            self.means = cluster_means(self.k, inputs, assignments)
            print(f"changed: {num_changed} / {len(inputs)}")


# -

inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],
                             [-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

# +
k = 2
random.seed(0)
clusterer = KMeans(k)
clusterer.train(inputs)
means = sorted(clusterer.means)

assert len(means) == k
means
# -

# check that the measures are close to what we expect
squared_distance(means[0], [-44, 5]) 

# #### Chosing K
#
# There are various ways to choose a k. One that is reasonably easy to develop intuition involves plotting the sum of squared errors (between each 
#

# +
from matplotlib import pyplot as plt

def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
    """finds the total squared error from k-means clustering the inputs
    """
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    # there isnt an assignment attribute
    assignments = [clusterer.classify(input) for input in inputs]
    return sum(squared_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))


# -

clusterer = KMeans(3)
clusterer.train(inputs)

means = clusterer.means
means

assignments = [clusterer.classify(input) for input in inputs]
assignments
