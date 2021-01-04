import pandas as pd
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import train_test_split, GridSearchCV

MAX_NUM_NEIGHBORS = 15

dataset = pd.read_csv('online_shoppers_intention.csv')

# transform all non-numeric data into numeric data
dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)

# select the last column as label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0, stratify=y)

parameters = [{'n_neighbors': range(1, MAX_NUM_NEIGHBORS), 'weights': ['uniform', 'distance'],
               'metric': ['euclidean', 'manhattan', 'minkowski']}]

grid_search = GridSearchCV(estimator=neighbors.KNeighborsClassifier(),
                           param_grid=parameters,
                           scoring='f1_macro',
                           cv=10,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters = grid_search.best_params_

print('optimized F1 score: ' + str(best_score))
print('optimized parameters: ' + str(best_parameters))
