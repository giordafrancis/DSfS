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

from typing import List

Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]


assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "Vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]


assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def vector_sum(vectors:List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # check that vector is not empty
    assert vectors, "no vectors provided!"
    # check that vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) 
            for i in range(num_elements)]



assert vector_sum([[1,2], [3, 4], [5, 6], [7,8]]) == [16,20]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c """
    return [c * v_i for v_i in v]


assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


assert vector_mean([[1,2], [3, 4], [5, 6]]) == [3, 4]


# A less obvious tool is the dot product. The dot product of two vectors is the sum of
# their componentwise products:

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([1, 2, 3], [4, 5, 6]) == 32


def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


assert sum_of_squares([1, 2, 3]) == 14

# +
import math

# Which we can use to compute its magnitude (or length):
def magnitude(v: Vector) -> float:
    """Returns the magnitude ( or length) of v"""
    return math.sqrt(sum_of_squares(v))


# -

assert magnitude([3, 4]) == 5


# +
# We now have all the pieces we need to compute the distance between two vectors

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1)**2 + ... + (v_n - w_n)**2 """
    return sum_of_squares(subtract(v, w))


# -

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))


Matrix = List[List]

from typing import Tuple

A = [[1, 2, 3],[4, 5, 6]]


def shape(A: Matrix) -> Tuple:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in first row, 0 is returns if empty lists of list are present
    return num_rows, num_cols


assert shape(A) == (2, 3)


# +
def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]

def get_column(A: Matrix, j:int) -> Vector:
    """Returns the j-th columns of A (as a Vector)"""
    return [A_i[j] # jth element of row vector A_i
           for A_i in A] # for each row vector


# -

assert get_row(A, 1) == [4, 5, 6] 

assert get_column(A,1) == [2, 5]

from typing import Callable


def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """Returns a num_rows x num_cols matrix(i, j) whose (i,j)- th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j)  # entry_fn on each i,j pair
           for j in range(num_cols)] # entry_fn(i,0),...,entry_fn(i,j)
           for i in range(num_rows)] # create one list for each i


def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i,j: 1 if i==j else 0)


assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]
