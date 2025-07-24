'''
uses sigmoid function to generate a curve: s(x) = 1/(1+e^-x)

thus predicted value: y_pred = 1/(1+e^-(wx+b))
'''
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    # initializing hyperparameters
    def __init__(self, lr=0.001, n_iter=1000):
        self.weight=None
        self.bias=None
        self.lr=lr
        self.n_iter=n_iter

    def fit(self, X, y):
        n_samples, n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iter):
            linear_pred=np.dot(X, self.weight)+self.bias
            prediction=sigmoid(linear_pred)

            dw=(1/n_samples)*np.dot(X.T, (prediction-y))
            db=(1/n_samples)*np.sum(prediction-y)

            self.weight=self.weight-(self.lr*dw)
            self.bias=self.bias-(self.lr*db)
            

    def predict(self, X):
        linear_pred=np.dot(X, self.weight)+self.bias
        y_pred=sigmoid(linear_pred)

        class_pred=np.array([0 if y<0.5 else 1 for y in y_pred])
        return class_pred