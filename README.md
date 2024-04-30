` import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_t`

///

class my_gradient:
    def __init__(self, X,Y, lrate, initial_b, weights, niteration):
        self.X = X
        self.Y = Y
        self.lrate = lrate
        self.initial_b = initial_b
        self.weights = weights
        self.niteration = niteration


    def predict(updatead_weights, x):
        return (updatead_weights[0] + sum((x * updatead_weights[1:])))


    def compute_error(updatead_weights, X, Y):
        totalError=0
        for i in range(0,len(X)):
            totalError +=(Y[i] - (updatead_weights * X[i] + updatead_weights[0]))**2
        return sum(totalError/float(len(X)))

    def gradientStep(weights, X, Y, lrate):
        dm = np.zeros(len(weights))
        N=float(len(X))
        copy_weights = weights
        updatead_weights = weights

        for i in range(0, len(X)):
            for j in range(0, len(weights)):
                dm[j] +=-(2/N) * X[i][j] * (Y[i] -(np.dot(weights, X[i])+ updatead_weights[0]))

            updatead_weights = np.array(copy_weights - (lrate * dm))

        return updatead_weights

    def gradientRun(X, Y, weights, lrate, numiteration, error_list):
        updatead_weights = weights
        for i in range(numiteration):
            updatead_weights = my_gradient.gradientStep(updatead_weights ,X ,Y ,lrate)
            error = my_gradient.compute_error(updatead_weights, X, Y)
            error_list.append(error)
            print('Iteration number ' , str(i),": The error--> ", error)
        return[updatead_weights,error_list]

    def fit(self):
        error_list = []
        [updatead_weights,final_error_list] = my_gradient.gradientRun(self.X, self.Y, self.weights, self.lrate, self.niteration, error_list)
        print(updatead_weights)

        return updatead_weights
///

df = pd.read_csv(r'/content/Cancer_Data.csv')
df

//

df = pd.read_csv(r'/content/Cancer_Data.csv')

Y = np.array(df['fractal_dimension_worst'])

df.drop(['Unnamed: 32' , 'diagnosis','id'] , axis = 1, inplace = True)
X = np.c_[np.ones(df.shape[0]), df]
lrate=0.00001
initial_b=0
#--------------------------------
n_features = df.shape[1]
weights = np.zeros(n_features + 1)
#--------------------------------
niteration=100
print(weights)

my_model = my_gradient(X ,Y ,lrate, initial_b, weights, niteration)
model1_weights= my_model.fit()

// 
#POLY

df = pd.read_csv(r'/content/Cancer_Data.csv')
df
//
X, y = df[["radius_mean", "texture_mean"]], df["fractal_dimension_worst"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=42)
//
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)
//
poly_reg_y_predicted = poly_reg_model.predict(X_test)
poly_reg_y_predicted
//
poly_reg_model.score(X_test,y_test)


