# MIT License
# Copyright (c) 2023 박진홍
# 이 코드는 MIT 라이센스에 따라 배포됩니다.

import csv
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
np.seterr(divide='ignore', invalid='ignore')

# CSV 파일에서 데이터 불러오기
data = []  # 데이터를 저장할 리스트
with open('Seoulmetro_Congestion_by_station_and_time_zone.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # 헤더 행 읽기
    for row in reader:
        # 빈 문자열이나 None을 0으로 대체
        row = [0 if value == '' or value == None else value for value in row]
        data.append(row)

# 필요한 정보 추출
lines = [row[2] for row in data]
DayOfWeek = [row[1] for row in data]
times = np.array([list(map(float, row[6:])) for row in data])
congestions = np.array([list(map(float, row[6:])) for row in data])

# 지하철 노선별
time_labels = header[6:] # 시간대 레이블

x = np.arange(len(time_labels))
# 각 지하철 노선에 대한 데이터 추출

user_input = int(input("Select the line of the subway you want to take (lines 1 to 8).\n If you want to see congestion statistics for the entire subway, enter 0.\n"))
user_day = int(input("Choose today's day : 1.Weekday 2.Saturday 3.Holiday (Please input number)\n"))
if user_input != 0:
    user_time_x, user_time_y = map(int, input("Enter the current time between 5:30 today and 00:30 the next day. (ex. 17:30) \n").split(":"))
    convert_to_input = user_time_x*2 + user_time_y/60 - 10.5
    if convert_to_input < 0:
        convert_to_input += 48

if user_input == 0:
    plt.figure(figsize=(12, 6))
    for line in range(1, 9):  # 1부터 8까지의 지하철 노선 반복하며 전체 혼잡도 그래프 그리기
        if user_day == 1:
            mask = (np.array(lines) == str(line)) & (np.array(DayOfWeek) == "weekday")
        elif user_day == 2:
            mask = (np.array(lines) == str(line)) & (np.array(DayOfWeek) == "Saturday")
        elif user_day == 3:
            mask = (np.array(lines) == str(line)) & (np.array(DayOfWeek) == "holiday")
        congestion = np.mean(times[mask][:, :len(time_labels)], axis=0)
        congestion = [val for val in congestion if val is not None and val != 0]
        plt.plot(x[:len(congestion)], congestion, label=f'Line {line}')
    plt.xticks(x[:39], time_labels[:39], rotation=90)
    plt.xlabel('Time Slot')
    plt.ylabel('Average Congestion')
    plt.title('Subway Congestion by Time Slot - Total')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    plt.figure(figsize=(12, 6))
    if user_day == 1:
        mask = (np.array(lines) == str(user_input)) & (np.array(DayOfWeek) == "weekday")
    elif user_day == 2:
        mask = (np.array(lines) == str(user_input)) & (np.array(DayOfWeek) == "Saturday")
    elif user_day == 3:
        mask = (np.array(lines) == str(user_input)) & (np.array(DayOfWeek) == "holiday")
    congestion = np.mean(times[mask][:, :len(time_labels)], axis=0)
    arr = times[mask][:, :len(time_labels)]

    for e in arr:
        plt.plot(x[:len(e)], e, '.', color='red')
    plt.plot(x[:len(congestion)], congestion, label='median')
    y = [y for x in arr for y in x]
    x = []
    while len(x) != len(y):
        for j in range(39):
            x.append(j)

    mid = sum(y)/len(y)
    x = np.array(x)
    y = np.array(y)
    # 다항 회귀 모델 생성
    degree = 9  # 다항식의 차수 설정
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))

    # 선형 회귀 모델 생성 및 훈련
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)

    new_X = np.array([[convert_to_input]])
    new_X_poly = poly_features.transform(new_X)
    predicted_y = poly_reg.predict(new_X_poly)

    # 결과 시각화
    X_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    predicted_y_range = poly_reg.predict(X_range_poly)

    if predicted_y[0] >= 36:
        congestion_expect = ('매우 혼잡')
    elif predicted_y[0] >= 30:
        congestion_expect = ('혼잡')
    elif predicted_y[0] >= 25:
        congestion_expect = ('보통')
    elif predicted_y[0] >= 15:
        congestion_expect = ('쾌적')
    elif predicted_y[0] < 15:
        congestion_expect = ('매우 쾌적')
    print(f"Current expected congestion : {congestion_expect}")

    plt.scatter(x, y, label='raw data', s = 1)
    plt.plot(X_range, predicted_y_range, label='Polynomial Regression', color='black')
    plt.axhline(y=mid, color='green', linestyle='--', label='Total median line')
    plt.xticks(x[:39], time_labels[:39], rotation=90)
    plt.xlabel('Time Slot')
    plt.ylabel('Congestion')
    plt.title(f'Subway Congestion by Subway Line and Time Slot - {user_input} Line')
    plt.legend()
    plt.show()



