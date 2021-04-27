import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/'
train = pd.read_csv(path +'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

# train = train.iloc[:10000]

# print(train.info())
 
# ## Basic EDA

import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(rc={'figure.figsize':(11.7,8.27)})

# # distribution of labels in training data
# sns.distplot(train.label)
# plt.xticks(rotation=45)
# plt.show() 
 
# # unique values in label coulum
# np.unique(train.label)
 
# sns.countplot(train.label)
# plt.xticks(rotation=45)
# plt.show()
 
# # What percentage of labels in training set equal to 0?
# train[train.label == 0].shape[0] / train.shape[0]

# # scatterplot + histogram of time and label
# sns.jointplot(x='time', y='label', data=train)
# plt.show()

# # scatterplot + histogram of s1 and label
# sns.jointplot(x='s1', y='label', data=train)
# plt.show()

print("----------- Step 1 Complete -----------") 

## Step 2: Define the Model 

X = train.copy()
x_cols = ['s'+ str(i) for i in list(range(1,17,1))]
X = X[x_cols]
X.head()

y = train['label']

# # Pearson correlation coefficient
# corr = train.corr(method='pearson')
# print(corr.label)
 
# # select columns for plotting
# plt_cols = ['s'+ str(i) for i in [1, 5, 6, 14]]
# plt_cols = ['label'] + plt_cols
# plt_cols
 
# sns.pairplot(train[plt_cols])
# plt.show()
 
# Train with selected columns
sub_x_cols = ['s1'] # ['s1', 's5', 's6', 's14']
sub_X = train[sub_x_cols]
sub_X.head()

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor   
from lightgbm import LGBMRegressor 

# model = DecisionTreeRegressor()
# model = RandomForestRegressor(n_estimators=10, random_state=0)
# model = GradientBoostingRegressor() 
# model = AdaBoostRegressor() 
# model = LinearRegression() 
# model = Ridge() 
# model = Lasso() 
model = KNeighborsRegressor() 
# model = LGBMRegressor(n_estimators=1000, learning_rate=0.01)  
# model.fit(sub_X, y)

model_name = type(model).__name__
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

X_train, X_valid, y_train, y_valid = train_test_split(sub_X, y, train_size=0.8, test_size=0.2, random_state=0)

import time 

start_time = time.time() 
model.fit(X_train, y_train)
training_time = int(time.time() - start_time) 
print("training time: %s seconds" % training_time)

valid_preds = model.predict(X_valid)
MAE = mean_absolute_error(y_valid, valid_preds)
print('MAE: {}'.format(MAE))

new_X = test[sub_x_cols]
new_y = model.predict(new_X)

submission_dt = submission.copy()
submission_dt['label'] = new_y
 
submission_dt.head()

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
 
# dd/mm/YY H:M:S
dt_string = now.strftime("%m-%d-%y_%H-%M-%S")

submission_dt.to_csv(path + 'submissions/submission_' + 'model_' + model_name + '_MAE_' + str(MAE) + '_training_' + str(training_time) + '_' + dt_string + '.csv', index=False)

print("----------- Step 2 Complete -----------") 

# ## Step 3: Model Comparison Through Pycaret 

# from pycaret.regression import *

# reg = setup(data = train, 
#              target = 'label',
#              numeric_imputation = 'median',
#              categorical_features = [], 
#              ignore_features = [],
#              normalize = True,
#              silent = True)

# # Returns the best model 
# best = compare_models() 

# print("----------- Step 3 Complete -----------") 
