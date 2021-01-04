# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors

NUM_NEIGHBORS = 1

# import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# fit k-Nearest Neighbor Regression to the dataset
regressor = neighbors.KNeighborsRegressor(NUM_NEIGHBORS, weights='uniform')
regressor.fit(X, y)

# plot the results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Regression using k-Nearest Neighbors')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predict a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)
