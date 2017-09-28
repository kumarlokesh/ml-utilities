"""
Building and evaluating machine learning model
using k nearest neighbors classification algorithm.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
# Building training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris['data'],
                                                    iris['target'],
                                                    random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
# Model building
knn.fit(X_train, y_train)
# Computing test set accuracy
print("iris test set accuracy: {0}".format(knn.score(X_test, y_test)))