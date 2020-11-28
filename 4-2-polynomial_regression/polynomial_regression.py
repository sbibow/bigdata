import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px

# this makes the plot show in the browser even if run from inside PyCharm (without this, the plot does not open)
import plotly.io as pio

pio.renderers.default = 'browser'

# Load the Amazon dataset
dataset = np.loadtxt('amazon_revenue_billion_usd.csv', delimiter=',')

# split into X and y
amazon_X = dataset[:, np.newaxis, 0]
amazon_y = dataset[:, np.newaxis, 1]

# generate plot with input data
fig = px.bar(x=amazon_X.flatten(), y=amazon_y.flatten())

# split the data into training/testing sets - use every third value for testing
amazon_X_test = amazon_X[::3]
amazon_X_train = [x for x in amazon_X if x not in amazon_X_test]

# split the targets into training/testing sets - use every third value for testing
amazon_y_test = amazon_y[::3]
amazon_y_train = [x for x in amazon_y if x not in amazon_y_test]

# TODO add some more future x values to amazon_X_test to get predictions for
# (e.g. for 2021, 2022,...)
amazon_X_test=np.append(amazon_X_test, [[2020],[2021],[2022],[2023]],axis=0)

# LINEAR REGRESSION
# TODO create linear regression object
regression = linear_model.LinearRegression()

# TODO train the model using the training sets
regression.fit(amazon_X_train, amazon_y_train)

# TODO make predictions using the testing set
amazon_y_pred = regression.predict(amazon_X_test)

# TODO plot outputs
fig.add_scatter(x=amazon_X_test.flatten(), y=amazon_y_pred.flatten(), name='predictions (linear regression)')

# QUADRATIC REGRESSION
regression = linear_model.LinearRegression()
# transform input data into format for quadratic polynomial
# TODO generate polynomial transform
poly = PolynomialFeatures(degree=10)

# now transform all data
X = poly.fit_transform(amazon_X)
y = poly.fit_transform(amazon_y)
X_pred = poly.fit_transform(amazon_X_pred)

# with this transformed data, the normal linear regression model can be used
# TODO fit model
regression.fit(X, y)
# TODO make predictions using the testing set
amazon_y_pred_poly = regression.predict(X_test)

# add another plot
fig.add_scatter(x=amazon_X_pred.flatten(), y=amazon_y_pred_poly[:, 1].flatten(),
                name='predictions (quadratic regression)')

# show the whole plot
fig.show()
