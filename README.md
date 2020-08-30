# Polynomial-Regression
# Hare Krishna
# Radhe Radhe
# What is Polynomial Regression?
Polynomial regression is a special case of linear regression where we fit a polynomial equation on the data with a curvilinear relationship between the target variable and the independent variables.

In a curvilinear relationship, the value of the target variable changes in a non-uniform manner with respect to the predictor (s).

In Linear Regression, with a single predictor, we have the following equation:

# linear regression equation
        Y= mx+c
        
where,

          Y is the target,

          x is the predictor,

          c is the bias,

          and m is the weight in the regression equation

This linear equation can be used to represent a linear relationship. But, in polynomial regression, we have a polynomial equation of degree n represented as:

polynomial regression equation
       Y= c+ m1x + m2x^2 + m3x^3...........

Here:

          c is the bias,

          m1, m2, …, mn are the weights in the equation of the polynomial regression,

          and n is the degree of the polynomial

The number of higher-order terms increases with the increasing value of n, and hence the equation becomes more complicated.

 

Polynomial Regression vs. Linear Regression
Now that we have a basic understanding of what Polynomial Regression is, let’s open up our Python IDE and implement polynomial regression.

I’m going to take a slightly different approach here. We will implement both the polynomial regression as well as linear regression algorithms on a simple dataset where we have a curvilinear relationship between the target and predictor. Finally, we will compare the results to understand the difference between the two.

First, import the required libraries and plot the relationship between the target variable and the independent variable:

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for calculating mean_squared error
from sklearn.metrics import mean_squared_error

# creating a dataset with curvilinear relationship
x=10*np.random.normal(0,1,70)
y=10*(-x**2)+np.random.normal(-100,100,70)

# plotting dataset
plt.figure(figsize=(10,5))
plt.scatter(x,y,s=15)
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.show()
polynomial regression dataset

Let’s start with Linear Regression first:

# Importing Linear Regression
from sklearn.linear_model import LinearRegression

# Training Model
lm=LinearRegression()
lm.fit(x.reshape(-1,1),y.reshape(-1,1))
Let’s see how linear regression performs on this dataset:

y_pred=lm.predict(x.reshape(-1,1))

# plotting predictions
plt.figure(figsize=(10,5))
plt.scatter(x,y,s=15)
plt.plot(x,y_pred,color='r')
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.show()
linear regression predictions

print('RMSE for Linear Regression=>',np.sqrt(mean_squared_error(y,y_pred)))
rmse linear regression

Here, you can see that the linear regression model is not able to fit the data properly and the RMSE (Root Mean Squared Error) is also very high.

Now, let’s try polynomial regression.
The implementation of polynomial regression is a two-step process. First, we transform our data into a polynomial using the PolynomialFeatures function from sklearn and then use linear regression to fit the parameters:

polynomial regression pipeline

We can automate this process using pipelines. Pipelines can be created using Pipeline from sklearn.

Let’s create a pipeline for performing polynomial regression:

# importing libraries for polynomial transform
from sklearn.preprocessing import PolynomialFeatures
# for creating pipeline
from sklearn.pipeline import Pipeline
# creating pipeline and fitting it on data
Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(x.reshape(-1,1),y.reshape(-1,1))
Here, I have taken a 2-degree polynomial. We can choose the degree of polynomial based on the relationship between target and predictor. The 1-degree polynomial is a simple linear regression; therefore, the value of degree must be greater than 1.

With the increasing degree of the polynomial, the complexity of the model also increases. Therefore, the value of n must be chosen precisely. If this value is low, then the model won’t be able to fit the data properly and if high, the model will overfit the data easily.

Read more about underfitting and overfitting in machine learning here.

Let’s take a look at our model’s performance:

poly_pred=pipe.predict(x.reshape(-1,1))
#sorting predicted values with respect to predictor
sorted_zip = sorted(zip(x,poly_pred))
x_poly, poly_pred = zip(*sorted_zip)
#plotting predictions
plt.figure(figsize=(10,6))
plt.scatter(x,y,s=15)
plt.plot(x,y_pred,color='r',label='Linear Regression')
plt.plot(x_poly,poly_pred,color='g',label='Polynomial Regression')
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.legend()
plt.show()
polynomial regression predictions

print('RMSE for Polynomial Regression=>',np.sqrt(mean_squared_error(y,poly_pred)))
rmse polynomial regression

We can clearly observe that Polynomial Regression is better at fitting the data than linear regression. Also, due to better-fitting, the RMSE of Polynomial Regression is way lower than that of Linear Regression.

But what if we have more than one predictor?
For 2 predictors, the equation of the polynomial regression becomes:

two degree polynomial regression

where,

          Y is the target,

          x1, x2 are the predictors,

          𝜃0 is the bias,

          and, 𝜃1, 𝜃2, 𝜃3, 𝜃4, and 𝜃5 are the weights in the regression equation

For n predictors, the equation includes all the possible combinations of different order polynomials. This is known as Multi-dimensional Polynomial Regression.

But, there is a major issue with multi-dimensional Polynomial Regression – multicollinearity. Multicollinearity is the interdependence between the predictors in a multiple dimensional regression problem. This restricts the model from fitting properly on the dataset.
