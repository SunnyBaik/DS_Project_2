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

# sum = 0
# X['s1_mean_50'] = 0
# for i in X.index:
#     sum += X.iloc[i, 1]
#     if (i%50 == 49):
#         X.iloc[(i-49):(i+1), 2] = sum / 50
#         sum = 0
#     elif (i == X.index[-1]):
#         X.iloc[(i-i%50):(i+1), 2] = sum/50
#         sum = 0

# X['s1_mean_1000'] = 0
# for i in X.index:
#     sum += X.iloc[i, 1]
#     if (i%1000 == 999):
#         X.iloc[(i-999):(i+1), 3] = sum / 1000
#         sum = 0
#     elif (i == X.index[-1]):
#         X.iloc[(i-i%1000):(i+1), 3] = sum/1000
#         sum = 0

# print("----------- Preprocessing Complete -----------")

# # Save the resulting dataframe as a csv file to save time 
# X.to_csv(path + 'window_preprocessed_train.csv') 

X = pd.read_csv(path + 'window_preprocessed_train.csv') 

# 2. Time Dependency
# 좌우로 20개씩, 총 41개의 시간에서 sensor 값을 보고 label 예측
# LinearRegression, CatBoost, MLP

# 시작과 끝쪽 데이터에 21, 22... 41개씩 있을 때는 어떻게 처리를 할 것인가?
# 없는 값들은 if문 처리를 통해 NULL로 해야 할지

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