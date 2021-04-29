import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

path = '/Users/jonghyunchoe/Documents/Others/GitHub/DS_Project_2/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')

x_cols = ['s1'] # ['s1', 's5', 's6', 's14']
# X = train[x_cols].copy()
X = test[x_cols].copy() 

value_counts = train['label'].value_counts()
# print(value_counts)

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
# 0, 1000, 2000, ... 마지막까지
start = 0 
end = 1000 
increasing = 1
prev_s1 = X.iloc[0, 2]
peak = prev_s1 
trough = prev_s1 

for i in range(0, X.index[-1]+1, 1000):
    next_s1 = X.iloc[i, 2]
    # 이전 구간이 올라가는 구간일 경우 
    if (increasing):
        # s1_mean이 올라가면 end를 업데이트해 구간 확장하기
        if (next_s1 >= prev_s1):
            peak = next_s1 
            end = i + 1000
        # s1_mean이 내려가면 올라가는 구간을 끝내고 내려가는 구간 시작하기
        else:
            # 올라가는 구간 내의 label 값 설정하기
            predicted_label = (peak-1500)/5 + (peak-2500)*0.1
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

    # 이전 구간이 내려가는 구간일 경우 
    else:
        # s1_mean이 올라가면 내려가는 구간을 끝내고 올라가는 구간 시작하기
        if (next_s1 >= prev_s1):
            # 내려가는 구간 내의 label 값 설정하기 
            predicted_label = (trough-1500)/5 + (trough-2500)*0.1
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
# for i in range(0, X.index[-1]+1, 1000):
#     next_s1 = X.iloc[i, 2]
#     # 이전 구간이 올라가는 구간일 경우 
#     if (increasing):
#         # s1_mean이 올라가면 end를 업데이트해 구간 확장하기
#         if (next_s1 >= prev_s1):
#             peak = next_s1 
#             end = i + 1000
#         # s1_mean이 내려가면 올라가는 구간을 끝내고 내려가는 구간 시작하기
#         else:
#             # 올라가는 구간 내의 label 값 설정하기
#             predicted_label = (peak-1500)/5 + (peak-2500)*0.1
#             if (predicted_label < 0):
#                 predicted_label = 0
#             X.iloc[start:end, 3] = predicted_label

#             increasing = 0
#             start = i
#             end = i + 1000
#             trough = next_s1 

#     # 이전 구간이 내려가는 구간일 경우 
#     else:
#         # s1_mean이 올라가면 내려가는 구간을 끝내고 올라가는 구간 시작하기
#         if (next_s1 >= prev_s1):
#             # 내려가는 구간 내의 label 값 설정하기 
#             predicted_label = (trough-1500)/5 + (trough-2500)*0.1
#             if (predicted_label < 0):
#                 predicted_label = 0
#             X.iloc[start:end, 3] = predicted_label 
            
#             increasing = 1
#             start = i 
#             end = i + 1000
#             peak = next_s1 
#         # s1_mean이 내려가면 end를 업데이트해 구간 확장하기 
#         else:
#             trough = next_s1 
#             end = i + 1000

#     prev_s1 = next_s1 

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