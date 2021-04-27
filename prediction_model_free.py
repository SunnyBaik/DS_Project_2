import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/'
train = pd.read_csv(path +'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

x_cols = ['s1'] # ['s1', 's5', 's6', 's14']
X = train[x_cols].copy()

value_counts = train['label'].value_counts()
print(value_counts)

print(train.head())
print(X.head())

# 1. sensor 데이터를 1000개씩 묶어서 평균 취해 새 컬럼에 저장하기 

sum = 0
X['s1_mean'] = 0
for i in X.index:
    sum += X.iloc[i, 0]
    if (i%1000 == 999):
        X.iloc[(i-999):(i+1), 1] = sum / 1000
        sum = 0
    elif (i == X.index[-1]):
        X.iloc[(i-i%1000):(i+1), 1] = sum/1000
        sum = 0

print("----------- Preprocessing Complete -----------")
# Save the resulting dataframe as a csv file to save time 
X.to_csv(path + 'preprocessed_train.csv') # , index=False)

# 2. 해당 시간으로부터 +- 
# 3. 올라가는 추세이면 구간 확장하기
# 4. 구간의 max 값을 찾으면 max 값을 이용해 label 값 계산하기 
#   4-1. label = (s1-1500)/5 + (s1-2500)*0.1
# 5. 내려가는 시간을 100초로 잡아 구간 설정하기 
# 6. 구간의 label을 모두 4-1에서 계산한 값으로 놓기 