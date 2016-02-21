__author__ = 'bhavika'

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

print digits.data

clf = svm.SVC(gamma=0.1, C=100)
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)

print ("Prediction: ", clf.predict(digits.data[-4]))
plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

