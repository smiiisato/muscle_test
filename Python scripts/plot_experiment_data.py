import matplotlib.pyplot as plt
import pandas as pd
import os

## ------------- CHANGE HERE ------------------------
FILE1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Experiments data/241127/Force Length/8kv.csv"
FILE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/old/8kv_2.csv"
FILE3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/old/rubber_cable_characterization_1128.csv"
WHAT_TO_DRAW = 4
# ---------------------------------------------------1

# Load the CSV data into a Pandas DataFrame
df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)
df3 = pd.read_csv(FILE3)

# Extract time and position columns
time1 = df1["Time(s)"]
position1 = df1["MC SW Overview - Actual Position(mm)"]
velocity1 = df1["MC SW Overview - Actual Velocity(m/s)"]
force1 = df1["MC SW Force Control - Measured Force(N)"]

time2 = df2["Time(s)"]
position2 = df2["MC SW Overview - Actual Position(mm)"]
velocity2 = df2["MC SW Overview - Actual Velocity(m/s)"]
force2 = df2["MC SW Force Control - Measured Force(N)"]

time3 = df3["Time(s)"]
position3 = df3["MC SW Overview - Actual Position(mm)"]
velocity3 = df3["MC SW Overview - Actual Velocity(m/s)"]
force3 = df3["MC SW Force Control - Measured Force(N)"]

# Plot the data
plt.figure(figsize=(10, 6))
if WHAT_TO_DRAW == 1:
    plt.plot(time1, position1, marker='o', linestyle='-', color='b', label='Position1')
    plt.plot(time2, position2, marker='o', linestyle='-', color='r', label='Position2')
    plt.title('Time vs Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (mm)')
elif WHAT_TO_DRAW == 2:
    plt.plot(time1, force1, marker='o', linestyle='-', color='b', label='Force1')
    plt.plot(time2, force2, marker='o', linestyle='-', color='r', label='Force2')
    plt.title('Time vs Force')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
elif WHAT_TO_DRAW == 3:
    plt.plot(position1, force1, marker='o', markersize=0.5, color='b', label='Original')
    plt.plot(position2, force2, marker='o', markersize=0.5, color='r', label='New')
    plt.title('Position vs Force')
    plt.xlabel('Position (mm)')
    plt.ylabel('Force (N)')
else:
    plt.plot(position3, force3, marker='*', markersize=0.5, color='b', label='rubber cable')
    plt.title('Position vs Force')
    plt.xlabel('Position (mm)')
    plt.ylabel('Force (N)')

plt.grid(True)
plt.legend()
plt.show()
