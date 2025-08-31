import sys
import matplotlib.pyplot as plt
from statistics import mean
from pathlib import Path
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

library_map = {
    "mpi": "MPI",
    "gpuccl": "GPUCCL",
    "nvshmem_h": "GPUSHMEM_Host",
    "nvshmem_d": "GPUSHMEM_Device",
}


def reject_outliers(data, m=0.1):
    data1 = np.array(data)
    low = np.quantile(data1, m)
    high = np.quantile(data1, 1 - m)
    data1 = data1[(data1 >= low)]
    data1 = data1[(data1 <= high)]
    return data1.tolist()


def parse_experiment_file(filename):

    results = {}
    current_lib = None
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip()
            if not line:
                # Skip blank lines
                continue

            # Otherwise, parse the line as data:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            experiment = parts[0]
            lib = parts[1]  # e.g. "unc", "mpi", "nccl"
            size = int(parts[2])  # e.g. 128, 256, ...
            perf_values = float(parts[3])
            if experiment not in results:
                results[experiment] = {}

            if lib != "unc":
                current_lib = lib
            if current_lib not in results[experiment]:
                results[experiment][lib] = {}

            if size not in results[experiment][current_lib]:
                results[experiment][current_lib][size] = {}
                results[experiment][current_lib][size]["baseline"] = []
                results[experiment][current_lib][size]["uniconn"] = []

            if lib != "unc":
                results[experiment][current_lib][size]["baseline"].append(perf_values)
            else:
                results[experiment][current_lib][size]["uniconn"].append(perf_values)

    return results


markers = {"mpi": "D", "gpuccl": "X", "nvshmem_d": "o", "nvshmem_h": "."}
color = {
    "mpi": "blue",
    "gpuccl": "green",
    "nvshmem_d": "magenta",
    "nvshmem_h": "orange",
}


def plot_runtimes(results, ax, title):
    """
    Creates a single figure with multiple lines:
      - One line per (comparison group, experiment type)
      - Y-values are (experiment_value - baseline_value), where baseline is "unc"
      - X-axis is logarithmic
    """
    line_width = 2
    line = []
    x_vals = []

    for size in sorted(results["jacobi"]["mpi"].keys()):
        x_vals.append(size)

    for lib_name, lib_data in results["jacobi"].items():
        baseline_runtimes = []
        uniconn_runtimes = []
        for size in sorted(lib_data.keys()):
            data_dict = lib_data[size]
            baseline_runtimes.append(
                np.mean(reject_outliers(data_dict.get("baseline"))) / 1000
            )
            uniconn_runtimes.append(
                np.mean(reject_outliers(data_dict.get("uniconn"))) / 1000
            )

        (line1,) = ax.loglog(
            x_vals,
            baseline_runtimes,
            color=color[lib_name],
            linestyle="--",
            linewidth=line_width,
            alpha=1,
            label=f"{library_map[lib_name]}: Native",
        )
        (line2,) = ax.loglog(
            x_vals,
            uniconn_runtimes,
            color=color[lib_name],
            linestyle="-",
            linewidth=line_width,
            alpha=1,
            label=f"{library_map[lib_name]}: Uniconn",
        )
        line.append(line1)
        line.append(line2)

    # Final labeling
    ax.set_xticks(x_vals, labels=x_vals)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of GPUs", fontsize=12)
    ax.grid(True)
    return line


def plot_diff(results, inset_ax):

    line_width = 2
    x_vals = []
    for size in sorted(results["jacobi"]["mpi"].keys()):
        x_vals.append(size)

    # inset_ax.set_xscale("log")
    for lib_name, lib_data in results["jacobi"].items():
        diff_runtimes = []
        for gpu_size in sorted(lib_data.keys()):
            data_dict = lib_data[gpu_size]
            diff_percent = (
                (
                    np.mean(reject_outliers(data_dict.get("uniconn")))
                    - np.mean(reject_outliers(data_dict.get("baseline")))
                )
                / np.mean(reject_outliers(data_dict.get("baseline")))
                * 100
            )

            print(lib_name)
            print(diff_percent)

            diff_runtimes.append(
                (
                    np.mean(reject_outliers(data_dict.get("uniconn")))
                    - np.mean(reject_outliers(data_dict.get("baseline")))
                )
                / np.mean(reject_outliers(data_dict.get("baseline")))
                * 100
            )
        inset_ax.semilogx(
            x_vals,
            diff_runtimes,
            color=color[lib_name],
            linestyle="-",
            linewidth=line_width,
            alpha=1,
        )

    inset_ax.set_xticks(x_vals, labels=x_vals)
    inset_ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    inset_ax.set_xlabel("Number of GPUs", fontsize=8)
    inset_ax.set_ylabel("Diff in %", fontsize=8)
    inset_ax.grid(True)


def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    filename3 = sys.argv[3]

    # 1. Parse the file
    results_perlmutter = parse_experiment_file(filename1)
    results_lumi = parse_experiment_file(filename2)
    results_marenostrum = parse_experiment_file(filename3)

    # 2. Plot all groups on one figure
    plt.style.use("seaborn-v0_8-paper")

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), constrained_layout=True)

    line = plot_runtimes(results_perlmutter, ax=axs[0], title="Perlmutter")
    inset_ax = inset_axes(axs[0], width="40%", height="30%", loc="upper right")
    plot_diff(results_perlmutter, inset_ax)

    plot_runtimes(results_lumi, ax=axs[1], title="Lumi")
    inset_ax = inset_axes(axs[1], width="40%", height="30%", loc="upper right")
    plot_diff(results_lumi, inset_ax)

    plot_runtimes(results_marenostrum, ax=axs[2], title="Marenostrum 5")
    inset_ax = inset_axes(axs[2], width="40%", height="30%", loc="upper right")
    plot_diff(results_marenostrum, inset_ax)

    fig.supylabel("Time (s)", fontsize=12)
    fig.legend(
        handles=line,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        frameon=False,
        bbox_transform=fig.transFigure,
        fontsize=12,
    )

    plt.savefig("Jacobi2D_results_10k_iter" + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
