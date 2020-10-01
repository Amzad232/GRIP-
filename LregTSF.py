# we will predict the percentage marks that a student is expected 
# to score based upon the number of hours they studied.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/shubhamv/Desktop/student_scores - student_scores.csv")
data.shape
data.head()
data.info()
data.describe()

plt.hist(data.Scores)
plt.hist(data.Hours)
data.boxplot('Scores')
data.boxplot('Hours', vert = False)

# pairplot
import seaborn
seaborn.pairplot(data) # histograms + scatterplot


Y = data[['Scores']]
X = data[['Hours']]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)
X_train
X_test
Y_train
Y_test

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,Y_train)
Y_pred = lr.predict(X_test)
Y_test.head(),Y_pred[0:5]

#(    Scores
# 2       27
# 24      86
# 15      95
# 5       20
# 12      41,
#             array([[34.76756311],
#                    [77.52719777],
#                    [87.75232779],
#                    [18.96508943],
#                    [46.85180769]]))

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, Y_pred)

# 30.946738482961003 

# Which is quite good.


