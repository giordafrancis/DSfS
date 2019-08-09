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

from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# +
# create line chart, years on x axis, gdp on y-axis
plt.plot(years, gdp, color="green", marker='o', linestyle='solid')

# add a tile
plt.title("Nominal GDP")

# add a label to the y-axis
plt.ylabel("Billions of $")
# -

# #### Bar Charts

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# +
# bars are by default width 0.8, so we'll add 0.1 to the left coordinates
# so that each bar is centered
xs = [i + 0.1  for i, _ in enumerate(movies)]

# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.bar(xs, num_oscars)

plt.ylabel("# of Academy Awards")
plt.title("My Favourite Movies")

# label x-axis with movie names at bar centers
plt.xticks([i + .1 for i, _ in enumerate(movies)], movies);
# -

# #### Histograms

from collections import Counter

# +
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]

# bucket grade by decile, but place the 100 in the 90s
histogram = Counter(min(grade//10 * 10, 90) for grade in grades)
histogram

# +
plt.bar([x + 5 for x in histogram.keys()], # shift bars to the right by 5
        histogram.values(), # give the bars the correct values
        10,# increase the width
        edgecolor=(0, 0, 0))  # black edges for the bars

plt.axis([-5, 105, 0, 5])
plt.xticks([i for i in range(0,110,10)]);
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")



# -

# #### Line Charts

variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x,y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# +
# we can make multiple calls to plt.plot
# to show multiple series on the same chart
plt.figure(figsize=(10,6))
plt.plot(xs, variance, 'g-', label = 'variance') # green solid line
plt.plot(xs, bias_squared, 'r-', label = 'bias^2') # red dotted line 
plt.plot(xs, total_error, 'b:', label = 'total error') # blue dotted line 

# because we've assigned labels to each series
# we can get a legend for free
# loc=9 means "top center"
plt.legend(loc=9)
plt.xlabel("model complexity")
plt.title("The Bias-Variance tradeoff")
# -

# #### Scatterplots

friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

# +
plt.scatter(friends, minutes)

# label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
    plt.annotate(label,
                xy=(friend_count,minute_count), # put the label with its point
                xytext=(5, -5),
                textcoords="offset points")
    
plt.title("Daily Minutes vs. Number of Friends");
plt.xlabel("# of Friends")
plt.ylabel("daily minutes spend on the website");
#plt.axis('equal')

# + {"active": ""}
# Warning for variables that are comparable with axis that aren't
# -

test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]
plt.scatter(test_1_grades, test_2_grades)
#plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
#plt.axis("equal") # turn this command on and off to see the difference

plt.show() # for ipython

# %whos








