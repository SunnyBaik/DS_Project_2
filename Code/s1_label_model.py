import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/'
train = pd.read_csv(path + 's1_label.csv')

X = train[['s1_peak_trough']]
y = train['label']

from sklearn.model_selection import train_test_split 

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor   
from lightgbm import LGBMRegressor 
from sklearn.metrics import mean_absolute_error 

# model = DecisionTreeRegressor()
# model = RandomForestRegressor(n_estimators=100, random_state=0)
# model = GradientBoostingRegressor() 
# model = AdaBoostRegressor() 
# model = LinearRegression() 
# model = Ridge() 
# model = Lasso() 
# model = KNeighborsRegressor(n_neighbors=4)  
# model = LGBMRegressor(n_estimators=1000, learning_rate=0.01)  

model.fit(train_X, train_y)
preds = model.predict(val_X)
print(mean_absolute_error(val_y, preds))
