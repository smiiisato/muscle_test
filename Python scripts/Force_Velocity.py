import numpy as np
import matplotlib.pyplot as plt

# ------------CHANGE HERE---------------
SPEED = 100
MIN_LENGTH = 37.543 # mm
MAX_LENGTH = 52.141 # mm
CRUISE_VELOCITIES = np.array([SPEED]*9).flatten()  # mm/s
SAMPLING_RATE = 100    # Hz
REST_PERIOD = 1         # seconds
CRUISE_FRACTION = 0.9   # 20% of total distance allocated to cruising
CSV_FILE = "motion_profile_{}mms.csv".format(SPEED)
print('Speed is {}mm/s'.format(SPEED))
# --------------------------------------

def generate_motion_profile(min_length, max_length, cruise_velocities, sampling_rate, rest_period, cruise_fraction=0.2):
    """
    Generates a position vs. time profile starting from max_length to min_length and back.
    Automatically calculates acceleration and deceleration to ensure a cruising phase exists.

    Parameters:
    - min_length (float): Minimum position (Point B) in mm.
    - max_length (float): Maximum position (Point A) in mm.
    - cruise_velocities (list of floats): List of cruise velocities in mm/s.
    - sampling_rate (float): Number of samples per second (Hz).
    - rest_period (float): Rest time at each point in seconds.
    - cruise_fraction (float): Fraction of total distance allocated to the cruise phase (0 < cruise_fraction < 1).

    Returns:
    - data (numpy.ndarray): Array with time, position, velocity, and acceleration columns.
    """
    # Total distance to move (positive value)
    total_distance = abs(max_length - min_length)

    # Time step
    dt = 1 / sampling_rate

    # Initialize lists to store time, position, velocity, and acceleration
    time_list = []
    position_list = []
    velocity_list = []
    acceleration_list = []

    # Current time
    current_time = 0

    for cruise_velocity in cruise_velocities:
        # Ensure cruise_velocity is positive
        cruise_velocity = abs(cruise_velocity)

        # Compute the distances for acceleration and deceleration
        s_cruise = cruise_fraction * total_distance
        s_accel_decel = total_distance - s_cruise

        # Distance for acceleration and deceleration phases
        s_accel = s_accel_decel / 2
        s_decel = s_accel_decel / 2

        # Compute acceleration and deceleration
        # Using kinematic equations: v^2 = 2*a*s
        acceleration = cruise_velocity**2 / (2 * s_accel)
        deceleration = acceleration  # Assuming symmetric acceleration and deceleration

        # Compute times for acceleration, cruise, and deceleration phases
        t_accel = cruise_velocity / acceleration
        t_decel = cruise_velocity / deceleration
        t_cruise = s_cruise / cruise_velocity

        # Total time for one-way motion
        t_total = t_accel + t_cruise + t_decel

        # Time arrays for each phase
        t_phase_accel = np.arange(0, t_accel, dt)
        t_phase_cruise = np.arange(t_accel, t_accel + t_cruise, dt)
        t_phase_decel = np.arange(t_accel + t_cruise, t_total + dt/10, dt)  # Include endpoint

        # Acceleration arrays for each phase (initial motion)
        accel_accel = np.full_like(t_phase_accel, -acceleration)
        accel_cruise = np.zeros_like(t_phase_cruise)
        accel_decel = np.full_like(t_phase_decel, deceleration)

        # Velocity arrays for each phase (initial motion)
        vel_accel = -acceleration * t_phase_accel
        vel_cruise = np.full_like(t_phase_cruise, -cruise_velocity)
        vel_decel = -cruise_velocity + deceleration * (t_phase_decel - t_accel - t_cruise)

        # Position arrays for each phase (initial motion)
        # Acceleration phase
        pos_accel = max_length + vel_accel * t_phase_accel / 2

        # Cruise phase
        pos_cruise = pos_accel[-1] + vel_cruise * (t_phase_cruise - t_accel)

        # Deceleration phase
        pos_decel = pos_cruise[-1] + vel_cruise[-1] * (t_phase_decel - (t_accel + t_cruise)) + 0.5 * deceleration * (t_phase_decel - (t_accel + t_cruise))**2

        # Combine arrays for initial motion
        time_one_way = np.concatenate((t_phase_accel, t_phase_cruise, t_phase_decel)) + current_time
        position_one_way = np.concatenate((pos_accel, pos_cruise, pos_decel))
        velocity_one_way = np.concatenate((vel_accel, vel_cruise, vel_decel))
        acceleration_one_way = np.concatenate((accel_accel, accel_cruise, accel_decel))

        # Rest at min_length
        if rest_period > 0:
            t_rest = np.arange(dt, rest_period + dt, dt) + time_one_way[-1]
            pos_rest = np.full_like(t_rest, min_length)
            vel_rest = np.zeros_like(t_rest)
            accel_rest = np.zeros_like(t_rest)

            time_one_way = np.concatenate((time_one_way, t_rest))
            position_one_way = np.concatenate((position_one_way, pos_rest))
            velocity_one_way = np.concatenate((velocity_one_way, vel_rest))
            acceleration_one_way = np.concatenate((acceleration_one_way, accel_rest))

            current_time = time_one_way[-1]
        else:
            current_time = time_one_way[-1]

        # Append initial motion
        time_list.extend(time_one_way)
        position_list.extend(position_one_way)
        velocity_list.extend(velocity_one_way)
        acceleration_list.extend(acceleration_one_way)

        # Reverse motion (return to max_length)
        # Time arrays for each phase
        t_phase_accel_rev = np.arange(0, t_accel, dt)
        t_phase_cruise_rev = np.arange(t_accel, t_accel + t_cruise, dt)
        t_phase_decel_rev = np.arange(t_accel + t_cruise, t_total + dt/10, dt)

        # Acceleration arrays for each phase (return motion)
        accel_accel_rev = np.full_like(t_phase_accel_rev, acceleration)
        accel_cruise_rev = np.zeros_like(t_phase_cruise_rev)
        accel_decel_rev = np.full_like(t_phase_decel_rev, -deceleration)

        # Velocity arrays for each phase (return motion)
        vel_accel_rev = acceleration * t_phase_accel_rev
        vel_cruise_rev = np.full_like(t_phase_cruise_rev, cruise_velocity)
        vel_decel_rev = cruise_velocity - deceleration * (t_phase_decel_rev - t_accel - t_cruise)

        # Position arrays for each phase (return motion)
        # Acceleration phase
        pos_accel_rev = min_length + vel_accel_rev * t_phase_accel_rev / 2

        # Cruise phase
        pos_cruise_rev = pos_accel_rev[-1] + vel_cruise_rev * (t_phase_cruise_rev - t_accel)

        # Deceleration phase
        pos_decel_rev = pos_cruise_rev[-1] + vel_cruise_rev[-1] * (t_phase_decel_rev - (t_accel + t_cruise)) - 0.5 * deceleration * (t_phase_decel_rev - (t_accel + t_cruise))**2

        # Combine arrays for return motion
        time_return = np.concatenate((t_phase_accel_rev, t_phase_cruise_rev, t_phase_decel_rev)) + current_time + dt
        position_return = np.concatenate((pos_accel_rev, pos_cruise_rev, pos_decel_rev))
        velocity_return = np.concatenate((vel_accel_rev, vel_cruise_rev, vel_decel_rev))
        acceleration_return = np.concatenate((accel_accel_rev, accel_cruise_rev, accel_decel_rev))

        # Rest at max_length
        if rest_period > 0:
            t_rest = np.arange(dt, rest_period + dt, dt) + time_return[-1]
            pos_rest = np.full_like(t_rest, max_length)
            vel_rest = np.zeros_like(t_rest)
            accel_rest = np.zeros_like(t_rest)

            time_return = np.concatenate((time_return, t_rest))
            position_return = np.concatenate((position_return, pos_rest))
            velocity_return = np.concatenate((velocity_return, vel_rest))
            acceleration_return = np.concatenate((acceleration_return, accel_rest))

            current_time = time_return[-1]
        else:
            current_time = time_return[-1]

        # Append return motion
        time_list.extend(time_return)
        position_list.extend(position_return)
        velocity_list.extend(velocity_return)
        acceleration_list.extend(acceleration_return)

    # Combine data
    data = np.column_stack((time_list, position_list, velocity_list, acceleration_list))
    return data

# Example usage
if __name__ == "__main__":
    # Parameters
    min_length = MIN_LENGTH          # mm (Point B)
    max_length = MAX_LENGTH        # mm (Point A)
    cruise_velocities = CRUISE_VELOCITIES
    sampling_rate = SAMPLING_RATE
    rest_period = REST_PERIOD
    cruise_fraction = CRUISE_FRACTION

    # Generate motion profile starting from max_length to min_length
    data = generate_motion_profile(min_length, max_length, cruise_velocities, sampling_rate, rest_period, cruise_fraction)

    # Save to CSV file (time and position columns)
    np.savetxt(CSV_FILE, data[:, 1], delimiter=',', fmt='%.6f') # save only the length data

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