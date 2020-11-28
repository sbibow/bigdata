import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import graphviz

dataset = pd.read_csv('online_shoppers_intention.csv')

# TODO transform all non-numeric data into numeric data using preprocessing.LabelEncoder()
dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)

# select the last column as label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# TODO split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y)

# TODO create classifier and fit to the training set
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# TODO predict test set results
y_pred = classifier.predict(X_test)

# TODO get the F1 score for the predicted test results
print(f1_score(y_test, y_pred))

# TODO create tree visualization and dump into file
dot_data = tree.export_graphviz(classifier)
graph = graphviz.Source(dot_data)
graph.render(format="svg")
