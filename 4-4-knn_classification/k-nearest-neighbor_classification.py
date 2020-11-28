import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import itertools
from pprint import pprint
import matplotlib.pyplot as plt

def fib(num):
  if num <= 1:
    return num
  else:
    return fib(num-1)+fib(num-2)

possible_k_values = [fib(x) for x in range(2,15)]
possible_weights = ["uniform"] #, "distance"]
possible_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]

dataset = pd.read_csv('online_shoppers_intention.csv')

# TODO transform all non-numeric data into numeric data using
# preprocessing.LabelEncoder()
le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)

# select the last column as label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# TODO split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

results = []

for k_val in possible_k_values:
  for weight in possible_weights:
    for metric in possible_metrics:
      # TODO create classifier and fit to the training set (use MAX_NUM_NEIGHBORS, WEIGHTS and METRIC)
      classifier = neighbors.KNeighborsClassifier(n_neighbors=k_val, weights=weight, metric=metric)
      classifier.fit(X_train, y_train)

      # predict test set results
      y_pred = classifier.predict(X_test)
      f1 = f1_score(y_test, y_pred)
      results.append((k_val, weight, metric, f1))

pprint(results)

fig, ax = plt.subplots()

k_vals = [entry[0] for entry in results]
weights = [entry[1] for entry in results]
metrics = [entry[2] for entry in results]
f1s = [1000*entry[3]**5 for entry in results]
ax.scatter(k_vals, metrics, c=f1s, s=f1s)
ax.set_xscale('log')
plt.show()
