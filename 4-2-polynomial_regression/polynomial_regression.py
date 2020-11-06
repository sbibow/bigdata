import numpy as np
from sklearn import linear_model
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

# TODO add some more future x values to amazon_X_test to get predictions for (e.g. for 2021, 2022,...)
amazon_X_test = ...

# LINEAR REGRESSION
# TODO create linear regression object
regression = ...

# TODO train the model using the training sets
regression.fit(...)

# TODO make predictions using the testing set
amazon_y_pred = regression.predict(...)

# TODO plot outputs
fig.add_scatter(x=....flatten(), y=....flatten(), name='predictions (linear regression)')

# QUADRATIC REGRESSION
# transform input data into format for quadratic polynomial
# TODO generate polynomial transform
poly = PolynomialFeatures(degree=...)
# now transform all data
X = poly.fit_transform(amazon_X_train)
y = poly.fit_transform(amazon_y_train)
X_test = poly.fit_transform(amazon_X_test)

# with this transformed data, the normal linear regression model can be used
# TODO fit model
regression.fit(...)
# TODO make predictions using the testing set
amazon_y_pred_poly = ...

# add another plot and show the whole figure
fig.add_scatter(x=amazon_X_test.flatten(), y=amazon_y_pred_poly[:, 1].flatten(),
                name='predictions (quadratic regression)')
fig.show()
