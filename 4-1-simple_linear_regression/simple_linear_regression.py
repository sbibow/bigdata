# Simple Linear Regression
# import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import dataset
dataset = pd.read_csv('salary_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# TODO split the dataset into training and test set (use train_test_split(...))
X_train, X_test, y_train, y_test = ...

# TODO fit Simple Linear Regression to the training set
regressor = ...
regressor.fit(...)

# TODO predict the test set results
y_pred = regressor....

# plot the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# TODO plot the test set results (similar to training set resolts)
plt.scatter(...)
# plot the trained regression again to compare with the test set
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

