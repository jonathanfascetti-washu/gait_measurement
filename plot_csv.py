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
import scipy.stats
from scipy.interpolate import interp1d

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
        print("\n~~~\nExtracting data from csv file...")
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
    # plt.plot(x, y, marker="o", linestyle="-")
    plt.plot(x, y)
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

    time_as_object = []
    for t in time:
        try:
            time_as_object.append(dt.strptime(t, "%H:%M:%S.%f"))
        except ValueError:
            time_as_object.append(dt.strptime(t, "%H:%M:%S"))

    return time_as_object


def zero_time(time):
    t0 = time[0]
    zeroed_times = [round((t - t0).total_seconds(), 3) for t in time]
    return zeroed_times


def smooth_data(Time, data, function):
    if function == "savgol":
        window = 8
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
    for window_size, poly in [(11, 2), (8, 3)]:
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

    return distance


def get_period(datax, datay):
    rms = np.sqrt(np.sum(np.square(datay)) / len(datay))
    print("\n~~~\nRoot Mean Squared:\n" + str(rms) + "\n~~~\n")
    peaks, _ = find_peaks(datay, prominence=15, height=rms)

    dt = np.zeros(0)
    try:
        for i in range(len(peaks) - 1):
            dt = np.append(dt, datax[peaks[i + 1]] - datax[peaks[i]])

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


def get_avalanches(datax, datay):
    datay = np.abs(datay)  # take absolute value of all data

    q = 75  # percentile
    thresh = np.percentile(datay, q)
    print("thresh: " + str(thresh))
    plot(datax, datay)
    avalanche = (
        []
    )  # where each index is a tuple, wiht the start and end indieces
    start = 0
    end = 0
    window = 0

    for i, val in enumerate(datay):
        if val > thresh and i > end:
            start = i
            # print("\nstart: " + str(start))

            for j in range(len(datay) - i):
                if i + j >= len(datay):  # prevent out of bounds error
                    break
                if datay[i + j] < thresh:
                    end = i + j
                    # print("end: " + str(end))
                    break
            else:
                end = min(i + window - 1, len(datay) - 1)
                # print("~window maxed out~end: " + str(end))
            avalanche.append((start, end))

    """ Old, overlapping avalanches
    for i, val in enumerate(datay):
        start = 0
        end = 0

        if val > median and i > end:
            start = i
            print("\nstart: " + str(start))

            for j in range(window):
                if i + j >= len(datay):  # prevent out of bounds error
                    break
                if datay[i + j] < median:
                    end = i + j
                    print("end: " + str(end))

                    break
                else:
                    end = min(i + window - 1, len(datay) - 1)
                    print("~window maxed out~end: " + str(end))
            avalanche.append((start, end))
    """

    binary_list = [1 if i > thresh else 0 for i in datay]

    analysis_array = np.array([datay.tolist(), binary_list, datax])
    # print("~~~\nAnalysis numPy array:\n")
    # np.set_printoptions(threshold=np.inf)
    # print(analysis_array)
    # print("~~~")

    # plot

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    # First plot
    ax1.plot(datax, datay.tolist(), label="Dataset Raw", color="blue")
    ax1.set_ylabel("Raw values")
    ax1.set_title("Stacked Plots of Raw and Bianry")
    ax1.legend()
    ax1.grid(True)

    # Second plot
    ax2.plot(datax, binary_list, label="Dataset Binary", color="green")
    ax2.set_xlabel("Time values")
    ax2.set_ylabel("Binary values")
    ax2.legend()
    ax2.grid(True)

    # Third plot
    ax3.plot(datax, datax, label="Dataset Time", color="blue")
    ax3.set_ylabel("Time values")
    ax3.set_title("Stacked Plots of Raw and Time")
    ax3.legend()
    ax3.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return avalanche, analysis_array


def get_avalanches_window(datax, datay):
    datay = np.abs(datay)
    median = np.median(datay)
    fac = 4  # scaling factor
    print("median / fac: " + str(median / fac))
    # plot(datax, datay)
    avalanche = []  # Store (start, end) index pairs
    window = 10
    start = 0
    end = 0

    i = 0
    while i < len(datay):
        if datay[i] > median / fac and i > end:
            start = i  # Start of an avalanche
            end = i  # Initialize end at the same point

            # Look ahead within the window to see where the avalanche extends
            for j in range(1, window):
                if i + j >= len(datay):  # Prevent out-of-bounds access
                    break
                if datay[i + j] > median / fac:
                    end = i + j  # Extend the end position

            avalanche.append((start, end))
            i = end  # Skip ahead to avoid overlapping detections
        i += 1  # Move to the next data point

    """ Old, overlapping avalanches
    for i, val in enumerate(datay):
        start = 0
        end = 0

        if val > median and i > end:
            start = i
            print("\nstart: " + str(start))

            for j in range(window):
                if i + j >= len(datay):  # prevent out of bounds error
                    break
                if datay[i + j] < median:
                    end = i + j
                    print("end: " + str(end))

                    break
                else:
                    end = min(i + window - 1, len(datay) - 1)
                    print("~window maxed out~end: " + str(end))
            avalanche.append((start, end))
    """
    print("~~~\nAvalanches found:")
    print(avalanche)
    print("~~~")
    return avalanche


def plot_avalanches(avalanche):
    # Compute avalanche sizes
    sizes = np.array([end - start for start, end in avalanche])

    # Get unique sizes and their frequencies manually
    unique_sizes = np.unique(sizes)
    frequencies = np.array([np.sum(sizes == size) for size in unique_sizes])

    # Scatter plot on log-log scale
    plt.figure(figsize=(7, 5))
    plt.scatter(unique_sizes, frequencies, color="blue", alpha=0.7)

    # Set log-log scale
    plt.xscale("log")
    plt.yscale("log")

    # Labels and formatting
    plt.xlabel("Avalanche Size (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Log-Log Scatter Plot of Avalanche Sizes")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


def merge_axes(x, y, z):
    merged = (x + y + z) / 3

    return merged


def interpolate_data(data1, data2):
    """
    Takes in two lists of (x, y) tuples of different sizes and randomly selects n values from the larger set,
    where n is the size of the smaller set. Returns the two datasets with matched sizes.
    """
    # Determine the smaller and larger dataset
    if len(data1) < len(data2):
        small_data, large_data = data1, data2
    else:
        small_data, large_data = data2, data1

    # Randomly select n points from the larger dataset
    n = len(small_data)
    sampled_indices = np.random.choice(
        len(large_data), size=n, replace=False
    )
    sampled_data = [large_data[i] for i in sampled_indices]

    return small_data, sampled_data


def compute_icc(data1, data2):
    """
    Computes the Intraclass Correlation Coefficient (ICC) for two datasets of (x, y) tuples.
    The datasets are first resized using interpolate_data() before computing ICC.
    """
    # Ensure both datasets have the same number of points
    matched_data1, matched_data2 = interpolate_data(data1, data2)

    # Extract only the y-values for ICC computation
    y1 = np.array([point[1] for point in matched_data1])
    y2 = np.array([point[1] for point in matched_data2])

    # Compute mean and variance
    mean_y1, mean_y2 = np.mean(y1), np.mean(y2)
    var_y1, var_y2 = np.var(y1, ddof=1), np.var(y2, ddof=1)

    # Compute covariance
    covariance = np.cov(y1, y2, ddof=1)[0, 1]

    # Compute ICC
    icc = (2 * covariance) / (var_y1 + var_y2)

    return icc


# Example usage
if __name__ == "__main__":
    filename = "/Users/jonathanfascetti/Desktop/Senior Design/jonathan_procedure1_04032025.csv"
    # Replace with your CSV file name
    extract_csv_data(filename)

    trunc = 0
    length = len(AccX)
    print("\n~~~\nSize of data set:")
    print("length: " + str(length))
    print("~~~")

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
    diff = []
    for i, t in enumerate(Time):
        if i < 100:
            diff.append(Time[i + 1] - Time[i])

    print("\n~~~\nAverage difference in time: " + str(sum(diff) / len(diff)))
    print("Standard deviation in difference in times: " + str(np.std(diff)))
    print("Maximum interval: " + str(max(diff)))
    print("Minimum interval: " + str(min(diff)))
    print("~~~")

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
    print("\n~~~")
    print("Data points excluded due to crossing threshold for each axis:")
    print(
        f"AccX: {thresh_AX}; exl: {val_AX} \nAccY: {thresh_AY}; exl: {val_AY} \nAccZ: {thresh_AZ}; exl: {val_AZ} \nGyroX: {thresh_GX}; exl: {val_GX} \nGyroY: {thresh_GY}; exl: {val_GY} \nGyroZ: {thresh_GZ}; exl: {val_GZ} \n"
    )
    print("~~~\n")

    # smooth data type can be "not_smoothed", "savgol", "moving_avg", "gaussian", "exp_moving_avg"
    # Time, AccX = gpt_compare_smoothing_methods(Time, AccX)

    # merged = merge_axes(GyroX, GyroY, GyroZ)
    # plot_all(Time, AccX, AccY, AccZ, GyroX, GyroY, GyroZ)

    Time, AccX = smooth_data(Time, np.abs(AccX), "savgol")

    avalanches_1, anaylsis_array = get_avalanches(Time, AccX)
    plot_avalanches(avalanches_1)

    np.save("jonathan_03312025_3xn_x_axis.npy", anaylsis_array)
