import numpy as np

class LinearRegression:
    # initialized with learning rate and number of iterations (hyperparameters)
    # change learning rate to improve model based on data
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr=lr
        self.n_iter=n_iter
        self.weight=None
        self.bias=None

    # fit method is used to train the model
    def fit(self, X, y):
        n_samples, n_features=X.shape
        # initialize weight and bias to 0
        self.weight=np.zeros(n_features)
        self.bias=0

        # we run this loop for number of iterations
        for _ in range(self.n_iter):
            # using y=wx+b (w=weight, b=bias) 
            y_pred=np.dot(X, self.weight)+self.bias

            '''
            gradient descent calculation to minimize cost function, to reduce diff between pred and actual values
            we calculate db (bias gradient) and dw (weight gradient)

            dw=(1/N)*(x_input*(y_pred-y_input))
            db=(1/N)*(y_pred-y_input)

            did X.T to transpose the input samples array
            '''
            dw=(1/n_samples)*np.dot(X.T, (y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)
    
            '''
            now we update weight and bias
            
            w_new=w_old-(learning_rate*dw)
            b_new=b_old-(learning_rate*db)
            '''
            self.weight=self.weight-(self.lr*dw)
            self.bias=self.bias-(self.lr*db)

    def predict(self, X):
        y_pred=np.dot(X, self.weight)+self.bias
        return y_pred