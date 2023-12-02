import csv
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
np.seterr(divide='ignore', invalid='ignore')

# Load data from CSV file
data = []  # List to store the data
with open('Seoulmetro_Congestion_by_station_and_time_zone.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the header row
    for row in reader:
        # Replace empty strings with None
        row = [0 if value == '' or value == None else value for value in row]
        data.append(row)

# Extract necessary information
lines = [row[2] for row in data]  # Line column
DayOfWeek = [row[1] for row in data]
times = np.array([list(map(float, row[6:])) for row in data])
congestions = np.array([list(map(float, row[6:])) for row in data])  # Convert time data to a NumPy array

# Create a plot (e.g., for subway lines)
time_labels = header[6:]  # Time slot labels


x = np.arange(len(time_labels))
# Extract and plot data for each subway line

user_day = int(input("Choose today's day : 1.Weekday 2.Saturday 3.Holiday (Please input number)\n"))
user_input = int(input("Select the line of the subway you want to take (lines 1 to 8).\n If you want to see congestion statistics for the entire subway, enter 0.\n"))

if user_input == 0:
    plt.figure(figsize=(12, 6))
    for line in range(1, 9):  # Iterate over subway lines 1 to 8
        if user_day == 1:
            mask = (np.array(lines) == str(line)) & (np.array(DayOfWeek) == "weekday")
        elif user_day == 2:
            mask = (np.array(lines) == str(line)) & (np.array(DayOfWeek) == "Saturday")
        elif user_day == 3:
            mask = (np.array(lines) == str(line)) & (np.array(DayOfWeek) == "holiday")
        congestion = np.mean(times[mask][:, :len(time_labels)], axis=0)  # Use the minimum time slots available
        congestion = [val for val in congestion if val is not None and val != 0]  # Remove None and 0 values from congestion data
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
    degree = 10  # 다항식의 차수 설정
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))

    # 선형 회귀 모델 생성 및 훈련
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)

    new_X = np.array([[25]])
    new_X_poly = poly_features.transform(new_X)
    predicted_y = poly_reg.predict(new_X_poly)
    print(f"X={new_X[0, 0]}일 때, 예측된 y 값: {predicted_y[0]}")

    # 결과 시각화
    X_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    predicted_y_range = poly_reg.predict(X_range_poly)

    plt.scatter(x, y, label='raw data', s = 1)
    plt.plot(X_range, predicted_y_range, label='Polynomial Regression', color='black')
    plt.axhline(y=mid, color='green', linestyle='--', label='Total median line')
    plt.xticks(x[:39], time_labels[:39], rotation=90)
    plt.xlabel('Time Slot')
    plt.ylabel('Congestion')
    plt.title(f'Subway Congestion by Subway Line and Time Slot - {user_input} Line')
    plt.legend()
    plt.show()



