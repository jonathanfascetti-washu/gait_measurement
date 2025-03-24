import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import datetime, timedelta

import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

import csv


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
    ynames = [
        "Acceleration (m/$s^2$)",
        "Acceleration (m/$s^2$)",
        "Acceleration (m/$s^2$)",
        "Angular Velocity (rad/s)",
        "Angular Velocity (rad/s)",
        "Angular Velocity (rad/s)",
    ]
    fig, axs = plt.subplots(2, 3)
    for i, data in enumerate([AccX, AccY, AccZ, GyroX, GyroY, GyroZ]):
        axs[i // 3, i % 3].plot(Time, data)
        axs[i // 3, i % 3].set_title(f"{names[i]}", fontsize=15)
        axs[i // 3, i % 3].tick_params(axis="x", labelsize=8)
        axs[i // 3, i % 3].set_xlabel("Time (s)", fontsize=8)
        axs[i // 3, i % 3].set_ylabel(ynames[i], fontsize=12)
    plt.show()


def plot_one(x, y, xname, yname):
    plt.plot(x, y, marker="o", linestyle="-")
    plt.title(f"{yname} over {xname}")
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()


def threshold(data, acc=True):
    log = 0
    val = []
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
                    log += 1
                    val.append(round(num, 3))
                except (
                    IndexError
                ):  # if dealing with the last index in the array...
                    return_data[i] = data[i - 1]
                    log += 1
                    val.append(round(num, 3))
            else:
                return_data[i] = data[i]
    else:  # gyroscopic data
        for i, num in enumerate(data):
            # set gyroscopic threshol to 20
            if abs(num) > 20:
                try:
                    # if passes the threshold, just average the two adjacent data points
                    return_data[i] = (data[i - 1] + data[i + 1]) / 2
                    log += 1
                    val.append(round(num, 3))
                except (
                    IndexError
                ):  # if dealing with the last index in the array...
                    return_data[i] = data[i - 1]
                    log += 1
                    val.append(round(num, 3))
            else:
                return_data[i] = data[i]
    return return_data, log, val


def clean_time(time):
    # there are always 12 characters in the beginning that can be cut ("xxxx-xx-xx ")
    time = [s[11:] for s in time]

    time = list(time)
    print(time[:10])

    time_as_object = [dt.strptime(t, "%H:%M:%S.%f") for t in time]

    return time_as_object


def zero_time(time):
    t0 = time[0]
    zeroed_times = [round((t - t0).total_seconds(), 3) for t in time]
    return zeroed_times


def smooth_data(Time, data, function):
    if function == "savgol":
        window = 25
        order = 3
        smooth = signal.savgol_filter(data, window, order)
    elif function == "moving_avg":
        window = 5
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

    with open("smooth_output.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Time", "data"])
        for i, line in enumerate(smooth):
            writer.writerow([Time[i], smooth[i]])
    return (Time, smooth)


def moving_average(data, window_size):
    return np.convolve(
        data, np.ones(window_size) / window_size, mode="valid"
    )


def exponential_smoothing(data, alpha):
    smooth = np.zeros_like(data)
    smooth[0] = data[0]
    for i in range(1, len(data)):
        smooth[i] = alpha * data[i] + (1 - alpha) * smooth[i - 1]
    return smooth


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


def gpt_compare_smoothing_methods(x, y):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    # Original data (for reference)
    for ax in axs:
        ax.plot(x, y, "k-", alpha=0.5, label="Original Data")

    # Moving Average
    for window_size in [2, 5, 7]:
        y_ma = moving_average(y, window_size)
        x_ma = x[: len(y_ma)]  # Adjust x-axis
        axs[0].plot(x_ma, y_ma, label=f"Win={window_size}")
    axs[0].set_title("Moving Average")
    axs[0].legend()
    axs[0].grid()

    # Savitzky-Golay
    for window_size, poly in [(11, 2), (21, 3)]:
        y_sg = savgol_filter(y, window_size, poly, mode="nearest")
        axs[1].plot(x, y_sg, label=f"Win={window_size}, Poly={poly}")
    axs[1].set_title("Savitzky-Golay")
    axs[1].legend()
    axs[1].grid()

    # Gaussian Smoothing
    for sigma in [1, 1.5, 2]:
        y_gauss = gaussian_filter1d(y, sigma)
        axs[2].plot(x, y_gauss, label=f"Sigma={sigma}")
    axs[2].set_title("Gaussian Smoothing")
    axs[2].legend()
    axs[2].grid()

    # Exponential Smoothing
    for alpha in [0.2, 0.35, 0.5]:
        y_exp = exponential_smoothing(y, alpha)
        axs[3].plot(x, y_exp, label=f"Alpha={alpha}")
    axs[3].set_title("Exponential Smoothing")
    axs[3].legend()
    axs[3].grid()

    # Labels
    for ax in axs:
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("X Axis Acceleration (m/$s^2$)", fontsize=12)

    plt.suptitle("Comparison of Smoothing Methods", fontsize=14)
    plt.tight_layout()
    plt.savefig("compare_smoothing_functions.png")
    plt.show()


def distance_traveled_1d(Time, data):
    distance = 0
    freq = 0.1
    for i, acc in enumerate(data):
        distance = distance + data[i] * freq

    print(distance)
    return distance


def get_period(datax, datay):
    peaks, _ = find_peaks(datay, prominence=15)

    dt = np.zeros(0)
    try:
        for i in range(len(peaks) - 1):
            dt = np.append(dt, datax[peaks[i + 1]] - datax[peaks[i]])
        print(type(dt))
        print(dt)
        stats(dt, "period")

        plt.plot(peaks, datay[peaks], "xr")
        plt.plot(datay)
        plt.suptitle("X Axis Acceleration and Detected Peaks")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s$^2$)")
        plt.show()

    except IndexError:
        print(
            "Error: fewer than two peaks detected.  Cannot calculate period"
        )
        pass


def stats(data, name="name"):
    print(" ~~~ STATS on " + name + " ~~~")
    print("~~ Size: " + str(len(data)))
    if len(data) != 0:
        print("~~ Average: " + str(sum(data) / len(data)))
        print("~~ Max: " + str(max(data)))
        print("~~ Min: " + str(min(data)))
    print("~~ Standard Deviation: " + str(np.std(data)))

    print(" ~~~ ")


# Example usage
if __name__ == "__main__":
    filename = "/Users/jonathanfascetti/Desktop/Senior Design/jonathan_wear_am03242025.csv"
    # Replace with your CSV file name
    extract_csv_data(filename)

    trunc = 0
    length = len(AccX)
    print("length: " + str(length))

    # parse the time to remove the date and just have the time itself.  Also "zero" it to the first entry
    Time_clean = clean_time(Time[trunc:length])
    Time_zeroed = zero_time(Time_clean)

    index_start = 0
    index_end = 0

    # Change between True/False when you want to look at a specific region or not
    segment = False

    if segment == True:
        t_start = 1
        t_end = 130

        for i, t in enumerate(Time_zeroed):
            if t > t_start and index_start == 0:
                index_start = i
            if t > t_end and index_end == 0:
                index_end = i
    else:
        index_start = 0
        index_end = len(Time_zeroed)

    Time_clean = clean_time(Time[trunc:length])
    Time = zero_time(Time_clean[index_start:index_end])

    # This code finds the average and std in time intervals
    """
    diff = []
    for i, t in enumerate(Time):
        if i < 100:
            diff.append(Time[i + 1] - Time[i])

    print("Average difference in time: " + str(sum(diff) / len(diff)))
    print("Standard deviation in difference in times: " + str(np.std(diff)))
    print("Maximum interval: " + str(max(diff)))
    """

    # Threshold all directions of data and truncate first few values (not all start right at zero)
    AccX, thresh_AX, val_AX = threshold(AccX[index_start:index_end])
    AccY, thresh_AY, val_AY = threshold(AccY[index_start:index_end])
    AccZ, thresh_AZ, val_AZ = threshold(AccZ[index_start:index_end])
    GyroX, thresh_GX, val_GX = threshold(
        GyroX[index_start:index_end], acc=False
    )
    GyroY, thresh_GY, val_GY = threshold(
        GyroY[index_start:index_end], acc=False
    )
    GyroZ, thresh_GZ, val_GZ = threshold(
        GyroZ[index_start:index_end], acc=False
    )

    ## print to terminal if any datapoints were removed because they exceeded the threshold
    print(
        f"AccX: {thresh_AX}; exl: {val_AX} \nAccY: {thresh_AY}; exl: {val_AY} \nAccZ: {thresh_AZ}; exl: {val_AZ} \nGyroX: {thresh_GX}; exl: {val_GX} \nGyroY: {thresh_GY}; exl: {val_GY} \nGyroZ: {thresh_GZ}; exl: {val_GZ} \n"
    )

    ## smooth data type can be "not_smoothed", "savgol", "moving_avg", "gaussian", "exp_moving_avg"
    # Time, AccZ = smooth_data(Time, AccZ, "not_smoothed")

    ## plot all on a 6-pane figure
    # plot_all(Time, AccX, AccY, AccZ, GyroX, GyroY, GyroZ)

    plot_one(Time, AccY, "Time", "AccX")
