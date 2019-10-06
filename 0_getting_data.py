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

# #### stdin and stdout ** skipped**
#
# piping data at the command line if you run Python scripts through it. 

import sys, re

sys.argv[0]

# +
# sys.argv is the list of command-line arguments
# sys.argv[0] is the name of the program itself
# sys.argv[1] will be the regex specified at the command line
regex = sys.argv[1]

for line in sys.stdin:
    if re.search(regex, line):
        sys.stdout.write(line)


# -

# #### Reading and writting files

# For example, imagine you have a file full of email addresses, one per line, and that
# you need to generate a histogram of the domains. The rules for correctly extracting
# domains are somewhat subtle (e.g., the Public Suffix List), but a good first approximation
# is to just take the parts of the email addresses that come after the @. (Which
# gives the wrong answer for email addresses like joel@mail.datasciencester.com.)

# +
def get_domain(email_address: str) -> str:
    """Split on '@' and return the las piece"""
    return email_address.lower().split("@")[-1]

# a couple of tests
assert get_domain("abola@gmail.com") == "gmail.com"
assert get_domain("abola@hotmail.com") == "hotmail.com"

# -

# Just stick some data there
with open('data/email_addresses.txt', 'w') as f:
    f.write("joelgrus@gmail.com\n")
    f.write("joel@m.datasciencester.com\n")
    f.write("this is a fake line\n")
    f.write("joelgrus@m.datasciencester.com\n")
    f.write("joel@hotmail.com\n")


# +
from collections import Counter

with open('data/email_addresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
                            for line in f
                            if "@" in line)
# -

domain_counts

# Delimited files

import csv

with open('data/colon_delimited_stock_prices.txt', 'w') as f:
    f.write("""date:symbol:closing_price
6/20/2014:AAPL:90.91
6/20/2014:MSFT:41.68
6/20/2014:FB:64.5
""")


def process_row(closing_price:float) -> float:
    return closing_price > 61


with open('data/colon_delimited_stock_prices.txt') as f:
    colon_reader  = csv.DictReader(f, delimiter = ":")
    for dict_row in colon_reader:
        #print(dict_row)
        date = dict_row["date"]
        symbol = dict_row["symbol"]
        closing_price = float(dict_row["closing_price"])
        bigger_61 = process_row(closing_price)
        print(bigger_61)

today_prices = { 'AAPL' : 90.91, 'MSFT' : 41.68, 'FB' : 64.5 }

with open("data/comma_delimited_stock_prices.txt", "w") as f:
    csv_writer = csv.DictWriter(f, delimiter = ',', fieldnames=["stock", "price"])
    csv_writer.writeheader()
    for stock, price in today_prices.items():
        csv_writer.writerow({"stock": stock, "price": price})

#
# #### Scrapping the web **fun but not completed now**

# Using an Unauthenticated API **same as above**
#
# - goot twitter authentication example and use of Twython API
#

import requests, json

github_user = "giordafrancis"
endpoint = f"https://api.github.com/users/{github_user}"

repos = json.loads(requests.get(endpoint).text)
repos


