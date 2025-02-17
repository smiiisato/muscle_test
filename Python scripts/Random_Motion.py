import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import time

# ------------CHANGE HERE---------------
MIN_LENGTH = 27.33 # mm
MAX_LENGTH = 39.922 # mm
SAMPLING_RATE = 100    # Hz
DURATION_TIME = 30 # sec
MAX_VELOCITY = 200 # mm/s # originally 200
MAX_ACCELERATION = 400 # mm/s² # originally 400
NUM_WAYPOINTS = 20 # Number of waypoints # originally 30
#1 = 20 way points
#2 = 35 way points
#3 = 50 way points
#4 = 65 way points
#5 = 80 way points

FILE_NAME = "random_motion_profile"
# --------------------------------------

# Parameters
min_length = MIN_LENGTH
max_length = MAX_LENGTH
sampling_rate = SAMPLING_RATE
duration_time = DURATION_TIME
max_velocity = MAX_VELOCITY
max_acceleration = MAX_ACCELERATION
num_waypoints = NUM_WAYPOINTS
random_seed = int(time.time() % 1000)        # Seed for reproducibility
#delta = (max_length - min_length) * 0.05  # 5% of the range
delta = 0
file_name = "../Motion profiles/random_motion/" + FILE_NAME + "_seed_{}_waypoints_{}.csv".format(random_seed, num_waypoints)

# Set random seed
np.random.seed(random_seed)

# Generate time array
t = np.arange(0, duration_time, 1 / sampling_rate)

# Generate waypoint times
waypoint_times = np.linspace(0, duration_time, num_waypoints)

# Generate random positions for waypoints within min+delta and max-delta
waypoint_positions = np.random.uniform(min_length + delta, max_length - delta, num_waypoints)

# Ensure the first waypoint starts at max_length
waypoint_positions[0] = max_length # - delta  # Slightly below max to avoid saturation

# Prevent consecutive waypoints at limits
for i in range(1, num_waypoints):
    if abs(waypoint_positions[i] - min_length) < delta and abs(waypoint_positions[i-1] - min_length) < delta:
        waypoint_positions[i] = min_length + delta
    if abs(waypoint_positions[i] - max_length) < delta and abs(waypoint_positions[i-1] - max_length) < delta:
        waypoint_positions[i] = max_length - delta

# Create cubic spline interpolation with 'not-a-knot' boundary conditions
cs = CubicSpline(waypoint_times, waypoint_positions, bc_type=((2, 0.0), 'not-a-knot'))

# Evaluate spline at sampling points
position = cs(t)

# Ensure positions are within bounds
position = np.clip(position, min_length, max_length)

# Calculate velocity and acceleration
velocity = cs(t, 1)  # First derivative
acceleration = cs(t, 2)  # Second derivative

# Enforce velocity and acceleration constraints
constraints_violated = (
    np.any(np.abs(velocity) > max_velocity) or
    np.any(np.abs(acceleration) > max_acceleration) or
    np.any(position <= min_length) or
    np.any(position >= max_length)
)

if constraints_violated:
    print("Constraints violated. Adjusting spline...")

    # Define objective function to minimize constraint violations
    def objective(waypoint_positions):
        cs = CubicSpline(waypoint_times, waypoint_positions, bc_type=((2, 0.0), 'not-a-knot'))
        pos = cs(t)
        vel = cs(t, 1)
        accel = cs(t, 2)
        # Calculate penalties
        vel_penalty = np.sum(np.maximum(np.abs(vel) - max_velocity, 0)**2)
        accel_penalty = np.sum(np.maximum(np.abs(accel) - max_acceleration, 0)**2)
        limit_penalty = np.sum(np.maximum(min_length - pos, 0)**2 + np.maximum(pos - max_length, 0)**2)
        saturation_penalty = np.sum((waypoint_positions <= min_length + delta) | (waypoint_positions >= max_length - delta))
        deviation_penalty = np.sum((waypoint_positions - original_waypoints)**2)
        # Total penalty
        total_penalty = (
            vel_penalty * 1.0 +
            accel_penalty * 1.0 +
            limit_penalty * 1.0 +
            saturation_penalty * 100.0 +  # High weight to prevent waypoints at limits
            deviation_penalty * 0.1
        )
        return total_penalty

    # Store original waypoints
    original_waypoints = waypoint_positions.copy()

    # Bounds for optimization (min_length + delta to max_length - delta for each waypoint)
    bounds = [(min_length + delta, max_length - delta) for _ in waypoint_positions]

    # Constraints to prevent waypoints at exact limits
    constraints = []

    # Run optimization to adjust waypoints
    result = minimize(
        objective,
        waypoint_positions,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 500}
    )

    # Use optimized waypoints
    waypoint_positions = result.x

    # Recompute spline with adjusted waypoints
    cs = CubicSpline(waypoint_times, waypoint_positions, bc_type=((2, 0.0), 'not-a-knot'))
    position = cs(t)
    position = np.clip(position, min_length, max_length)
    velocity = cs(t, 1)
    acceleration = cs(t, 2)

# Plot position over time
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, position, label='Position')
plt.title('Position vs. Time')
plt.ylabel('Position (mm)')
plt.grid(True)
plt.legend()

# Plot velocity over time
plt.subplot(3, 1, 2)
plt.plot(t, velocity, label='Velocity')
plt.title('Velocity vs. Time')
plt.ylabel('Velocity (mm/s)')
plt.grid(True)
plt.legend()

# Plot acceleration over time
plt.subplot(3, 1, 3)
plt.plot(t, acceleration, label='Acceleration')
plt.title('Acceleration vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (mm/s²)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Check constraints
print(f"Max position: {np.max(np.abs(position)):.2f} mm")
print(f"Min position: {np.min(np.abs(position)):.2f} mm")
print(f"Max velocity: {np.max(np.abs(velocity)):.2f} mm/s")
print(f"Max acceleration: {np.max(np.abs(acceleration)):.2f} mm/s²")

# Save position data to CSV file (only one column)
np.savetxt(file_name, position, delimiter=',', fmt='%.6f')

# Print the latest time
print(f"Latest time: {t[-1]:.3f} seconds")