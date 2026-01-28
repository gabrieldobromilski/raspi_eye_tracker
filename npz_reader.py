import numpy as np
import matplotlib.pyplot as plt


def plot_y(npz_path):
    with np.load(npz_path) as data:
        y = data["y"]

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1])
    plt.xlabel("y_1")
    plt.ylabel("y_2")
    plt.title("Wykres punktowy danych y")
    plt.grid(True)
    plt.show()



def plot_calibration(npz_path):
    with np.load(npz_path) as data:
        X = data["X_left"]
        y = data["y"]

    pupil_x = X[:, 0]
    pupil_y = X[:, 1]

    start_idx = 0
    segment_idx = 0

    for i in range(1, len(y)):
       
        if not np.array_equal(y[i], y[i - 1]):
            show_segment(
                pupil_x[start_idx:i],
                pupil_y[start_idx:i],
                segment_idx
            )
            start_idx = i
            segment_idx += 1

    
    show_segment(
        pupil_x[start_idx:],
        pupil_y[start_idx:],
        segment_idx
    )


def show_segment(px, py, idx):
    
    mean_x = np.mean(px)
    mean_y = np.mean(py)

    plt.figure()


    plt.plot(px, py, marker="o", label="Trajektoria źrenicy")

    plt.scatter(
        mean_x,
        mean_y,
        s=200,
        marker="x",
        color = "red",
        label="Punkt średni"
    )

    plt.xlabel("X środka źrenicy")
    plt.ylabel("Y środka źrenicy")
    plt.title(f"Kalibracja – segment {idx}")
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_y("calibration_data_left.npz")
    plot_calibration("calibration_data_left.npz")

