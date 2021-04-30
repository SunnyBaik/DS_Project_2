import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/CSV/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

x_cols = ['s1'] # ['s1', 's5', 's6', 's14']
# X = train[x_cols].copy()
X = test[x_cols].copy() 

# # 1. sensor 데이터를 1000개씩 묶어서 평균 취해 새 컬럼에 저장하기 

# sum = 0
# X['s1_mean'] = 0
# for i in X.index:
#     sum += X.iloc[i, 0]
#     if (i%1000 == 999):
#         X.iloc[(i-999):(i+1), 1] = sum / 1000
#         sum = 0
#     elif (i == X.index[-1]):
#         X.iloc[(i-i%1000):(i+1), 1] = sum/1000
#         sum = 0

# print("----------- Preprocessing Complete -----------")
# # Save the resulting dataframe as a csv file to save time 
# X.to_csv(path + 'preprocessed_test.csv') 
# # X.to_csv(path + 'preprocessed_train.csv') # , index=False)

X = pd.read_csv(path + 'preprocessed_test.csv') 
X['label'] = 0

# 2. 올라가는 추세이면 구간 확장하기

start = 0 
end = 1000 
increasing = 1
prev_s1 = X.iloc[0, 2]
peak = prev_s1 
trough = prev_s1 

# path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/'
# train = pd.read_csv(path + 's1_peak_label.csv')

# model_X = train[['s1_peak']]
# model_y = train['label']

sub_x_cols = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16'] # ['s1', 's5', 's6', 's14']
sub_X = train[sub_x_cols]
sub_y = train['label']

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
model = LGBMRegressor(n_estimators=1000, learning_rate=0.01)  

# model.fit(sub_X, sub_y)

def predict_label(x):
    if (x < 2100):
        return 0
    elif (x >= 2100 and x < 2400):
        return 200 
    elif (x >= 2400 and x < 2700):
        return 213.33 
    elif (x >= 2700 and x < 3000):
        return 240 
    elif (x >= 3000 and x < 3100):
        return 266.67
    elif (x >= 3100 and x < 3200):
        return 306.67 
    else:
        return 533.33 

for i in range(0, X.index[-1]+1, 1000):
    next_s1 = X.iloc[i, 2]
    # 이전 구간이 올라가는 구간일 경우 
    if (increasing):
        # s1_mean이 내려가면 올라가는 구간을 끝내고 내려가는 구간 시작하기
        if (next_s1 < prev_s1 and (end - start) >= 8000):
            # 올라가는 구간 내의 label 값 설정하기
            predicted_label = predict_label(peak) 
            # input = test.iloc[i-1000, 2:18]
            # predicted_label = (model.predict([input]))[0]
            # predicted_label = (peak-1500)/5 + (peak-2500)*0.1
            print("Rising interval  - start : ", int(start/100), " end: ", int(end/100), " peak: ", peak, " predicted_label: ", predicted_label)
            if (peak <= 2100):
                predicted_label = 0
            if (start/100 >= 10):
                start = start - 1000
                end = end - 1000
            X.iloc[start:end, 3] = predicted_label

            increasing = 0
            start = i
            end = i + 1000
            trough = next_s1 
        # s1_mean이 올라가면 end를 업데이트해 구간 확장하기
        else:
            peak = next_s1 
            end = i + 1000

    # 이전 구간이 내려가는 구간일 경우 
    else:
        # s1_mean이 올라가면 내려가는 구간을 끝내고 올라가는 구간 시작하기
        if (next_s1 >= prev_s1 and (end - start) >= 8000):
            # 내려가는 구간 내의 label 값 설정하기 
            predicted_label = predict_label(trough) 
            # input = test.iloc[i-1000, 2:18]
            # predicted_label = (model.predict([input]))[0]
            # predicted_label = (trough-1500)/5 + (trough-2500)*0.1
            print("Falling interval - start : ", int(start/100), " end: ", int(end/100), " trough: ", trough, " predicted_label: ", predicted_label)
            if (trough <= 2100):
                predicted_label = 0
            if (start/100 >= 10):
                start = start - 1000
                end = end - 1000
            X.iloc[start:end, 3] = predicted_label 
            
            increasing = 1
            start = i 
            end = i + 1000
            peak = next_s1 
        # s1_mean이 내려가면 end를 업데이트해 구간 확장하기 
        else:
            trough = next_s1 
            end = i + 1000

    prev_s1 = next_s1 

X['id'] = X.index 
X = X[['id', 'label']].copy() 
X.to_csv(path + 'predicted_test.csv', index=False)

# 3. 구간의 max 값을 찾으면 max 값을 이용해 label 값 계산하기 
#   3-1. label = (s1-1500)/5 + (s1-2500)*0.1
# 4. 내려가는 시간을 100초로 잡아 구간 설정하기 
# 5. 구간의 label을 모두 4-1에서 계산한 값으로 놓기 
# 6. 끝까지 반복하기 

# 초반 0부터 1500까지 올려주는 값을 따로 처리해줘야 할 수도 
# 올라가다 내려가는 것을 2-3번 또는 5-6번 정도 봐주는게 나을 수도, noise가 워낙 커서 
# 마지막에 end가 i + 999보다 작으면 어떻게 할지 

# preprocess test and try prediction 