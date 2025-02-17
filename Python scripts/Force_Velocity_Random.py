import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ------------CHANGE HERE---------------
MIN_LENGTH = 37.543 # mm
MAX_LENGTH = 52.141 # mm
SAMPLING_RATE = 100  # Hz
REST_PERIOD = 1  # seconds
CSV_FILE = "motion_profile_random_velocity.csv"
# --------------------------------------

def generate_random_velocity_profile(total_time, sampling_rate, num_control_points=5, max_velocity=10):
    """
    Generates a random smooth velocity curve using cubic spline interpolation.
    
    Parameters:
    - total_time (float): Total duration for the motion profile.
    - sampling_rate (float): Number of samples per second (Hz).
    - num_control_points (int): Number of random control points to define the spline.
    - max_velocity (float): Maximum allowable velocity for random control points.
    
    Returns:
    - time (numpy.ndarray): Time array.
    - velocity (numpy.ndarray): Generated velocity array.
    """
    # Generate random control points for velocity profile
    time_control_points = np.linspace(0, total_time, num_control_points)
    velocity_control_points = np.random.uniform(0, max_velocity, num_control_points)
    
    # Ensure velocity starts and ends at zero
    velocity_control_points[0] = 0
    velocity_control_points[-1] = 0
    
    # Create cubic spline to interpolate the control points
    cs = CubicSpline(time_control_points, velocity_control_points)
    
    # Generate time array based on sampling rate
    time = np.arange(0, total_time, 1 / sampling_rate)
    
    # Generate velocity profile using the spline
    velocity = cs(time)
    
    return time, velocity

def generate_motion_profile(min_length, max_length, sampling_rate, rest_period, random_velocity_profile=True):
    """
    Generates a position vs. time profile with either a fixed or random velocity curve.
    
    Parameters:
    - min_length (float): Minimum position (Point B) in mm.
    - max_length (float): Maximum position (Point A) in mm.
    - sampling_rate (float): Number of samples per second (Hz).
    - rest_period (float): Rest time at each point in seconds.
    - random_velocity_profile (bool): Whether to generate a random velocity profile.
    
    Returns:
    - data (numpy.ndarray): Array with time, position, velocity, and acceleration columns.
    """
    # Total distance to move (positive value)
    total_distance = abs(max_length - min_length)

    # Random velocity profile parameters
    total_time = 5  # Arbitrary total time for one-way motion (can be adjusted)
    
    # Generate velocity profile (random or fixed)
    if random_velocity_profile:
        time, velocity = generate_random_velocity_profile(total_time, sampling_rate)
    else:
        # Fixed velocity profile for comparison (linear acceleration, cruise, deceleration)
        cruise_velocity = 10  # mm/s
        accel_time = 1  # s
        cruise_time = 2  # s
        decel_time = 1  # s
        time = np.linspace(0, accel_time + cruise_time + decel_time, int(sampling_rate * (accel_time + cruise_time + decel_time)))
        velocity = np.piecewise(time, 
            [time < accel_time, (time >= accel_time) & (time <= accel_time + cruise_time), time > accel_time + cruise_time],
            [lambda t: cruise_velocity * t / accel_time, cruise_velocity, lambda t: cruise_velocity - cruise_velocity * (t - accel_time - cruise_time) / decel_time])
    
    # Integrate velocity to get position
    position = min_length + np.cumsum(velocity) * (1 / sampling_rate)
    
    # Compute acceleration as the derivative of velocity
    acceleration = np.gradient(velocity, time)

    # Combine time, position, velocity, and acceleration into a data array
    data = np.column_stack((time, position, velocity, acceleration))

    return data

# Example usage
if __name__ == "__main__":
    # Parameters
    min_length = MIN_LENGTH  # mm (Point B)
    max_length = MAX_LENGTH  # mm (Point A)
    sampling_rate = SAMPLING_RATE
    rest_period = REST_PERIOD

    # Generate motion profile with a random smooth velocity curve
    data = generate_motion_profile(min_length, max_length, sampling_rate, rest_period)

    # Save to CSV file (time, position, velocity, and acceleration columns)
    np.savetxt(CSV_FILE, data, delimiter=',', fmt='%.6f', header="Time,Position,Velocity,Acceleration", comments='')

    # print the total time
    print("Total time is {}".format(max(data[:, 0])))

    # Plot position, velocity, and acceleration vs. time
    time = data[:, 0]
    position = data[:, 1]
    velocity = data[:, 2]
    acceleration_profile = data[:, 3]

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Position plot
    axs[0].plot(time, position, label='Position', color='blue')
    axs[0].set_ylabel('Position (mm)')
    axs[0].grid(True)
    axs[0].legend()

    # Velocity plot
    axs[1].plot(time, velocity, label='Velocity', color='green')
    axs[1].set_ylabel('Velocity (mm/s)')
    axs[1].grid(True)
    axs[1].legend()

    # Acceleration plot
    axs[2].plot(time, acceleration_profile, label='Acceleration', color='red')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Acceleration (mm/s²)')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()
