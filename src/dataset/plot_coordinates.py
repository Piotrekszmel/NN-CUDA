import numpy as np 
import matplotlib.pyplot as plt

def plot_dataset():
    with open("src/dataset/coordinates_targets_zero.txt") as f:
        lines = f.readlines()
        x_0 = [float(line.split()[0]) for line in lines]
        y_0 = [float(line.split()[1]) for line in lines]

    with open("src/dataset/coordinates_targets_one.txt") as f:
        lines = f.readlines()
        x_1 = [float(line.split()[0]) for line in lines]
        y_1 = [float(line.split()[1]) for line in lines]

    plt.figure(1)
    plt.axvline(x=0, color="black")
    plt.axhline(y=0, color='black', linestyle='-')
    plt.scatter(x_0, y_0, c="red", marker="o", label="0")
    plt.scatter(x_1, y_1, c="blue", marker="o", label="1")
    plt.legend(loc="upper left", bbox_to_anchor=(1.005, 1))


    with open("src/dataset/coordinates_output_zero.txt") as f:
        lines = f.readlines()
        x_0 = [float(line.split()[0]) for line in lines]
        y_0 = [float(line.split()[1]) for line in lines]

    with open("src/dataset/coordinates_output_one.txt") as f:
        lines = f.readlines()
        x_1 = [float(line.split()[0]) for line in lines]
        y_1 = [float(line.split()[1]) for line in lines]

    plt.figure(2)
    plt.axvline(x=0, color="black")
    plt.axhline(y=0, color='black', linestyle='-')
    plt.scatter(x_0, y_0, c="red", marker="o", label="0")
    plt.scatter(x_1, y_1, c="blue", marker="o", label="1")
    plt.legend(loc="upper left", bbox_to_anchor=(1.005, 1))

    plt.show()


plot_dataset()