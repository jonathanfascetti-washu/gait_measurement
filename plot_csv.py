import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter


def extract_csv_data(filename):
    """
    Reads a CSV file and creates a NumPy array for each column as a global variable.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        None (Variables are created dynamically in the global scope)
    """
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(filename)

    # Convert each column into a NumPy array and assign it to a variable dynamically
    for col in df.columns:
        globals()[col] = df[col].dropna().to_numpy()
        print(f"Created variable: {col}")


def plot(xaxis, yaxis):
    plt.plot(xaxis, yaxis)
    plt.show()


def plot_all(Time, AccX, AccY, AccZ, GyroX, GyroY, GyroZ):
    names = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    fig, axs = plt.subplots(2, 3)
    for i, data in enumerate([AccX, AccY, AccZ, GyroX, GyroY, GyroZ]):
        axs[i // 3, i % 3].plot(Time, data)
        axs[i // 3, i % 3].set_title(f"{names[i]}")
        axs[i // 3, i % 3].tick_params(axis="x", labelsize=8)
        axs[i // 3, i % 3].set_xlabel("Time (s)", fontsize=9)
    plt.show()


def plot_one(x, y):
    plot(x[2:], y[2:])
    plt.show()


def threshold(data, acc=True):
    # create empty array to return at end of function
    return_data = np.zeros(len(data))

    # threshold will be different for acceleration vs gyroscopic data
    if acc == True:
        for i, num in enumerate(data):
            # set acceleration threshold to 100
            if abs(num) > 100:
                try:
                    # if passes the threshold, just average the two adjacent data points
                    return_data[i] = (data[i - 1] + data[i + 1]) / 2
                except (
                    IndexError
                ):  # if dealing with the last index in the array...
                    return_data[i] = data[i - 1]
            else:
                return_data[i] = data[i]
    else:  # gyroscopic data
        for i, num in enumerate(data):
            # set gyroscopic threshol to 20
            if abs(num) > 20:
                try:
                    # if passes the threshold, just average the two adjacent data points
                    return_data[i] = (data[i - 1] + data[i + 1]) / 2
                except (
                    IndexError
                ):  # if dealing with the last index in the array...
                    return_data[i] = data[i - 1]
            else:
                return_data[i] = data[i]
    return return_data


def parse_and_clean_time(Time):
    # there are always 12 characters in the beginning that can be cut ("xxxx-xx-xx ")
    Time = [s[11:] for s in Time]

    time_as_object = [dt.strptime(t, "%H:%M:%S.%f") for t in Time]
    t0 = time_as_object[0]
    zeroed_times = [
        round((t - t0).total_seconds(), 3) for t in time_as_object
    ]
    # print(zeroed_times)
    return zeroed_times


def smooth_data(Time, data, function):
    if function == "savgol":
        window = 50
        order = 3
        smooth = signal.savgol_filter(data, window, order)
    elif function == "moving_avg":
        window = 10
        smooth = np.convolve(data, np.ones(window) / window, mode="same")
    elif function == "gaussian":
        sigma = 2
        smooth = gaussian_filter(data, sigma)
    elif function == "exp_moving_avg":
        alpha = 0.2
        smooth = np.zeros_like(data)
        smooth[0] = data[0]  # First value remains the same
        for i in range(1, len(data)):
            smooth[i] = alpha * data[i] + (1 - alpha) * smooth[i - 1]
    elif function == "not_smoothed":
        smooth = data

    return (Time, smooth)


def compare_smoothing_functions(Time, data):
    functions = [
        "not_smoothed",
        "savgol",
        "moving_avg",
        "gaussian",
        "exp_moving_avg",
    ]

    fig, axs = plt.subplots(1, len(functions))

    for i, function in enumerate(functions):
        Time, smooth = smooth_data(Time, data, function)
        axs[i].plot(Time, smooth)
        axs[i].set_title(f"{functions[i]}")
        axs[i].tick_params(axis="x", labelsize=8)
        axs[i].set_xlabel("Time (s)", fontsize=9)
    plt.show()


# Example usage
if __name__ == "__main__":
    filename = "~/Downloads/imu_output.csv"
    # Replace with your CSV file name
    extract_csv_data(filename)

    trunc = 10

    # parse the time to remove the date and just have the time itself.  Also "zero" it to the first entry
    Time = parse_and_clean_time(Time[trunc:])
    # all_six = [AccX, AccY, AccZ, GyroX, GyroY, GyroZ]

    # Threshold all directions of data and truncate first few values (not all start right at zero)
    AccX = threshold(AccX[trunc:])
    AccY = threshold(AccY[trunc:])
    AccZ = threshold(AccZ[trunc:])
    GyroX = threshold(GyroX[trunc:], acc=False)
    GyroY = threshold(GyroY[trunc:], acc=False)
    GyroZ = threshold(GyroZ[trunc:], acc=False)

    # smooth data type can be "not_smoothed", "savgol", "moving_avg", "gaussian", "exp_moving_avg"
    # AccX = smooth_data(Time, AccX, "exp_moving_avg")

    compare_smoothing_functions(Time, AccX)

    # plot all on a 6-pane figure
    # plot_all(Time, AccX, AccY, AccZ, GyroX, GyroY, GyroZ)

    # plot_one(Time, AccX)
