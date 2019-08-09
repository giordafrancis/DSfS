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

# Create a dict of dicts that represents the users if and name

users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
   ]

# Friendship data is also represented as a ist of pairs ID's

friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), 
                    (2, 3), (3, 4), (4, 5), (5, 6), 
                    (5, 7), (6, 8), (7, 8), (8, 9)]

# Let's create a dict to look at every pair and finds the friendship of every user

# initialize the dict with an empty list for each user id
friendships = {user["id"]: [] for user in users}

friendships

# +
# loop over friendship_pairs and completed it

for i, j in friendship_pairs:
    friendships[i].append(j) # add j as friend of i
    friendships[j].append(i) # add i as friend of j
# -

friendships


# #### What's the total number of connections?

def number_of_friends(user):
    """How many friens does _user_ have?"""
    user_id = user["id"]  # users is list of dicts
    friend_ids = friendships[user_id]
    return len(friend_ids)


total_connections = sum(number_of_friends(user)
                       for user in users)
total_connections

# now divide by the number of users
num_users = len(users)
avg_connections = total_connections / num_users
avg_connections

# #### Who are the most connected people?

num_friends_by_id = [(user["id"], number_of_friends(user)) 
                     for user in users]

# +
num_friends_by_id.sort(
    key= lambda id_and_friends: id_and_friends[1],
    reverse=True
)

num_friends_by_id


# -

# #### Data Scientist you may know suggester? hint your instinct is to suggest that users might know the friends of theirs friends. 

def foaf_ids_bad(user):
    """foaf short for friend of a friend"""
    return [foaf_id
        for friend_id in friendships[user["id"]] #for my users return friends
        for foaf_id in friendships[friend_id]    #for each friend return their friendships (foaf_id)
        ]


# +
#for user[0]

foaf_ids_bad(users[0]) #needs some works as the output is not clean 

# +
from collections import Counter

def foaf_ids(user):
    """foaf short for friend of a friend"""
    user_id = user["id"]
    return Counter(
            foaf_id 
            for friend_id in friendships[user_id]    #for each of my friends
            for foaf_id in friendships[friend_id]    #find their friends
            if foaf_id != user_id                    #who arent me
            and foaf_id not in friendships[user_id]  #and aren't my friends
    )


# -

foaf_ids(users[0])
foaf_ids(users[3])

interests = [
(0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
(0, "Spark"), (0, "Storm"), (0, "Cassandra"),
(1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
(1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
(2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
(3, "statistics"), (3, "regression"), (3, "probability"),
(4, "machine learning"), (4, "regression"), (4, "decision trees"),
(4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
(5, "Haskell"), (5, "programming languages"), (6, "statistics"),
(6, "probability"), (6, "mathematics"), (6, "theory"),
(7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
(7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
(8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
(9, "Java"), (9, "MapReduce"), (9, "Big Data")
]


# ## Interests

# #### How to find user that have shared interests?

def data_scientists_who_like(target_interest):
    """find the ids of all users who like the target interest"""
    return [user_id
           for user_id, user_interest in interests
           if user_interest == target_interest]


data_scientists_who_like("R")

# The above works but has the lookup is week as it has to loop over all values in the list. Let's build an index of interests to users:

from collections import defaultdict

# +
user_ids_by_interests = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interests[interest].append(user_id)
# -

user_ids_by_interests

# And another that maps users to interest

# +
interest_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interest_by_user_id[user_id].append(interest)
# -

interest_by_user_id


# Now it’s easy to find who has the most interests in common with a given user:
#
# - Iterate over the user’s interests.
# - For each interest, iterate over the other users with that interest.
# - Keep count of how many times we see each other user.

def most_common_interests_with(user):
    return Counter(
        interested_user_id
        for interest in interest_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interests[interest]
            if interested_user_id != user["id"]
    )


most_common_interests_with(users[2])

# #### What are the most popular interests? 

# One simple (if not particularly exciting) way to find the most popular interests is simply
# to count the words:
#
# - Lowercase each interest (since different users may or may not capitalize their
# interests).
# -  Split it into words.
# - Count the results.

words_and_counts = Counter(
    word
    for user, interest in interests
    for word in interest.lower().split()
)

for word, count in words_and_counts.most_common(5):
    if count > 1:
        print(word, count)

# ## Salaries and Experience

# How to calculate average salaries based on tenure? hint tenure will vary so a bucket might be required

salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
(48000, 0.7), (76000, 6),
(69000, 6.5), (76000, 7.5),
(60000, 2.5), (83000, 10),
(48000, 1.9), (63000, 4.2)]


def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more then five"


salary_by_tenure = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure[bucket].append(salary)


# +
average_salary_by_bucket = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure.items()
}

average_salary_by_bucket
# -




