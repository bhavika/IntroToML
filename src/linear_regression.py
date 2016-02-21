from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2]

#split into test and train data
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

#create object of Linear Regression
regr = LinearRegression()

#Train the model using training sets
regr.fit(diabetes_X_train, diabetes_Y_train)

plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()