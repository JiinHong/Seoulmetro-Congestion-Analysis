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
times = np.array([list(map(float, row[6:])) for row in data])  # Convert time data to a NumPy array

# Create a plot (e.g., for subway lines)
time_labels = header[6:]  # Time slot labels
x = np.arange(len(time_labels))

# Extract and plot data for each subway line
plt.figure(figsize=(12, 6))
for line in range(1, 10):  # Iterate over subway lines 1 to 9
    mask = (np.array(lines) == str(line))
    congestion = np.mean(times[mask][:, :len(time_labels)], axis=0)  # Use the minimum time slots available
    congestion = [val for val in congestion if val is not None and val != 0]  # Remove None and 0 values from congestion data
    if congestion:  # Check if congestion list is not empty
        plt.plot(x[:len(congestion)], congestion, label=f'Line {line}')
plt.xticks(x[:len(congestion)], time_labels[:len(congestion)], rotation=90)
plt.xlabel('Time Slot')
plt.ylabel('Average Congestion')
plt.title('Subway Congestion by Subway Line and Time Slot')
plt.legend()
plt.grid(True)
plt.show()
