import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/CSV/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

x_cols = ['label', 's1'] # ['s1', 's5', 's6', 's14']
X = train[x_cols].copy()

# # 1. Data Preprocessing
# # 데이터를 50개씩, 1000개씩 묶어서 처리 

# # 1-1. Training Data
# sum = 0
# X['s1_mean_50'] = 0
# for i in X.index:
#     sum += X.iloc[i, 1]
#     if (i%50 == 49):
#         X.iloc[(i-49):(i+1), 2] = sum / 50
#         sum = 0
#     elif (i == X.index[-1]):
#         X.iloc[(i-i%50):(i+1), 2] = sum / 50
#         sum = 0

# X['s1_mean_1000'] = 0
# for i in X.index:
#     sum += X.iloc[i, 1]
#     if (i%1000 == 999):
#         X.iloc[(i-999):(i+1), 3] = sum / 1000
#         sum = 0
#     elif (i == X.index[-1]):
#         X.iloc[(i-i%1000):(i+1), 3] = sum / 1000
#         sum = 0

# # 1-2. Test Data
# X_test = test[['s1']].copy()

# sum = 0
# X_test['s1_mean_50'] = 0
# for i in X_test.index:
#     sum += X_test.iloc[i, 0]
#     if (i%50 == 49):
#         X_test.iloc[(i-49):(i+1), 1] = sum / 50
#         sum = 0
#     elif (i == X_test.index[-1]):
#         X_test.iloc[(i-i%50):(i+1), 1] = sum / 50 
#         sum = 0

# X_test['s1_mean_1000'] = 0
# for i in X_test.index:
#     sum += X_test.iloc[i, 0]
#     if (i%1000 == 999):
#         X_test.iloc[(i-999):(i+1), 2] = sum / 1000
#         sum = 0
#     elif (i == X_test.index[-1]):
#         X_test.iloc[(i-i%1000):(i+1), 2] = sum / 1000
#         sum = 0     

# print("----------- Preprocessing Complete -----------")

# # Save the resulting dataframe as a csv file to save time 
# X.to_csv(path + 'window_preprocessed_train.csv') 
# X_test.to_csv(path + 'window_preprocessed_test.csv')

X = pd.read_csv(path + 'window_preprocessed_train.csv') 
X_test = pd.read_csv(path + 'window_preprocessed_test.csv')

# 2. Time Dependency
# 좌우로 20개씩, 총 41개의 시간에서 sensor 값을 보고 label 예측
# LinearRegression, CatBoost, MLP

# # 2-1. Training Data
# import math 

# window_df = pd.DataFrame(index=range(math.ceil(X.index[-1]/50)), columns = ['label', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40'])

# for i in range(math.ceil(X.index[-1]/50)):
#     for j in range(0, 41):
#         count = -1000
#         if (50*i + 50*j + count >= 0 and 50*i + 50*j + count <= X.index[-1]):
#             window_df.iloc[i, j+1] = X.iloc[50*i + 50*j + count, 3]
#         else:
#             window_df.iloc[i, j+1] = 1403.75
#     window_df.iloc[i, 0] = X.iloc[50*i, 1] 

# window_df.to_csv(path + 'window_sensor.csv') 

window_df = pd.read_csv(path + 'window_sensor.csv')

window_X = window_df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40']].copy()
window_y = window_df[['label']].copy() 

from sklearn.linear_model import LinearRegression 

model = LinearRegression()
# CatBoost

model.fit(window_X, window_y) 

# # 2-2. Test Data

# test_window_df = pd.DataFrame(index=range(math.ceil(X_test.index[-1]/50)), columns = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40'])

# for i in range(math.ceil(X_test.index[-1]/50)):
#     for j in range(0, 41):
#         count = -1000
#         if (50*i + 50*j + count >= 0 and 50*i + 50*j + count <= X_test.index[-1]):
#             test_window_df.iloc[i, j] = X_test.iloc[50*i + 50*j + count, 2]
#         else:
#             test_window_df.iloc[i, j] = 1403.75

# test_window_df.to_csv(path + 'test_window_sensor.csv')

test_window_df = pd.read_csv(path + 'test_window_sensor.csv')

test_window_X = test_window_df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39', 't40']].copy()
preds = model.predict(test_window_X)

# print(preds) 
# print(preds[0])
# print(len(preds[0]))

for i in range(len(preds)):
    if (i*50+49 <= submission.index[-1]):
        if (preds[i][0] <= 100):
            preds[i][0] = 0 
        submission.iloc[(i*50):(i*50+50), 1] = preds[i][0]    
    else:
        if (preds[i][0] <= 100):
            preds[i][0] = 0 
        submission.iloc[(i*50):(submission.index[-1]+1), 1] = preds[i][0] 

submission.to_csv(path + 'window_predicted_test.csv', index=False)

# submission
# 0-49, 50-99, 100-149, ... 

# 3. Reduce Noise
# 이동평균법 
# 예측값이 100 이하면 0으로 처리 
  
# 4. Box Prediction
# 구간별 알고리즘 적용 
# 위 예측값으로 구간 통일 
# start, end, height 예측 

# Sliding Window 개념 찾아보기 

# Smooth Curve by savgol_filter 
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

# Ethylene 농도를 측정하는 센서 