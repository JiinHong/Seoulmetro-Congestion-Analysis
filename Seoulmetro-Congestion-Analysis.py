import csv
import numpy as np
import matplotlib.pyplot as plt
import warnings

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
    arr = times[mask][:, :len(time_labels)]
    for idx, e in enumerate(arr):
        ret = list(map(float, e))
        print(idx, " : ", ret)
    congestion = np.mean(times[mask][:, :len(time_labels)], axis=0)
    plt.plot(x[:len(congestion)], congestion)
    plt.xticks(x[:39], time_labels[:39], rotation=90)
    plt.xlabel('Time Slot')
    plt.ylabel('Average Congestion')
    plt.title(f'Subway Congestion by Subway Line and Time Slot - {user_input} Line')
    plt.legend()
    plt.grid(True)
    plt.show()