import pandas as pd
from sklearn.model_selection import train_test_split
from linear_regression_scratch import LinearRegression

df=pd.read_csv("linear_data.csv")
''' 
double brackets for x variable created a dataframe (2D array)
single bracket for y creates a series (1D array)
this is because sklearn expects 2D array for 'x' variable as it will be used to train our model
(model can be trained on multiple features)
'y' variable is the target, thus simple series is enough
'''
x=df[["Hours"]]
y=df["Scores"]

# random_state is a seed which ensures that the same data is used for training and testing. this helps with reproducibility, collaborating and debugging.
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42) 


model=LinearRegression()
LinearRegression.fit(X_train, y_train)
predictions=model.predict(X_test)

def mse(y_test, predictions):
    return np̣.mean((y_test-predictions)**2)
    

mse=mse(y_test, predictions)
print(mse)
