# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import matplotlib.pyplot as plt
from collections import deque
import random

# Create a fixed-length deque of length 50 to store the data points
data_points = deque(maxlen=50)

# Create an empty plot
fig, ax = plt.subplots()

# Set the x-axis and y-axis limits to 100
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Create a scatter plot to visualize the data points
scatter = ax.scatter([], [])

# Iterate through the data points and update the scatter plot
for i in range(100):
    # Generate and add data points to the deque
    new_x = i
    new_y = random.randint(0, 100)
    data_points.append((new_x, new_y))

    # Update the scatter plot with the new data points
    x_values = [x for x, y in data_points]
    y_values = [y for x, y in data_points]
    scatter.set_offsets(list(zip(x_values, y_values)))
    plt.pause(0.1)

# Show the plot
plt.show()