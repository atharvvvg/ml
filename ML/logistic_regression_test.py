import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression_scratch import LogisticRegression

bc=datasets.load_breast_cancer()
X, y=bc.data, bc.target
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

model=LogisticRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

print([y_test, y_pred])

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_pred)


acc=accuracy(y_pred, y_test)
print(acc)