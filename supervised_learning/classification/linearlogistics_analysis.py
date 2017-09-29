from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
stratify=cancer.target, random_state=42)

lr001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("logistics regression with C=0.01")
print("training set score: %f" % lr001.score(X_train, y_train))
print("test set score: %f" % lr001.score(X_test, y_test))

lr = LogisticRegression().fit(X_train, y_train)
print("logistics regression with C=1")
print("training set score: %f" % lr.score(X_train, y_train))
print("test set score: %f" % lr.score(X_test, y_test))

lr100 = LogisticRegression(C=100).fit(X_train, y_train)
print("logistics regression with C=100")
print("training set score: %f" % lr100.score(X_train, y_train))
print("test set score: %f" % lr100.score(X_test, y_test))

plt.plot(lr.coef_.T, 'o', label="C=1")
plt.plot(lr100.coef_.T, 'o', label="C=100")
plt.plot(lr001.coef_.T, 'o', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.ylim(-5, 5)
plt.legend()
plt.show()