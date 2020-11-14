import pandas as pd
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

MAX_NUM_NEIGHBORS = 20  # von 1-20 in Schleife erhöhen
WEIGHTS = 'uniform'  # wem langweilig ist: diesen Hyperparameter ebenfalls optimieren in weiterer Schleife
METRIC = 'euclidean'

dataset = pd.read_csv('online_shoppers_intention.csv')

# transform all non-numeric data into numeric data
dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)

# select the last column as label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

for num_neighbors in range(1, MAX_NUM_NEIGHBORS):  # [1,2,3,4,...,20]
    print('-----------------------------------------')
    print('number of neighbors: ' + str(num_neighbors) + ', weights: ' + WEIGHTS + ', metric: ' + METRIC)

    # create classifier and fit to the training set
    classifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, weights=WEIGHTS, metric=METRIC)

    # predict test set results
    y_pred = classifier.predict(X_test)

    # get the F1 score for the predicted test results
    print(f1_score(y_test, y_pred))

    # -> was ist das optimale k für unseren Datensatz?
