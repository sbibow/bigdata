import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import graphviz

dataset = pd.read_csv('online_shoppers_intention.csv')

# transform all non-numeric data into numeric data
dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)

# select the last column as label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

pca = PCA(n_components=17)
X = pca.fit(X).transform(X)

# split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0, stratify=y)

# create classifier and fit to the training set
classifier = tree.DecisionTreeClassifier()
path = classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

cross_val_scores = []
for ccp_alpha in ccp_alphas:
    classifier = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    classifier.fit(X_train, y_train)

    # cross-validate classifier
    cross_val_scores.append(np.mean(cross_val_score(classifier, X_train, y_train, cv=10, scoring='f1_macro')))

    # predict test set results
    y_pred = classifier.predict(X_test)

    # get the F1 score for the predicted test results
    print(f1_score(y_test, y_pred, average='macro'))

# create tree visualization and dump into file
dot_data = tree.export_graphviz(classifier, feature_names=dataset.columns[:-1], out_file=None)
graph = graphviz.Source(dot_data)
graph.render('tree')
