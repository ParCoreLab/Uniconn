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
            data_type = parts[1]  # e.g. "unc", "mpi", "nccl"
            size = int(parts[2])  # e.g. 128, 256, ...
            perf_values = float(parts[3])
            if experiment not in results:
                results[experiment] = {}

            if data_type != "unc":
                current_lib = data_type
            if current_lib not in results[experiment]:
                results[experiment][data_type] = {}

            if size not in results[experiment][current_lib]:
                results[experiment][current_lib][size] = {}
                results[experiment][current_lib][size]["baseline"] = []
                results[experiment][current_lib][size]["uniconn"] = []

            if data_type != "unc":
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


def plot_bandwidth(results, ax):

    line_width = 2
    line = []
    x_vals = []
    for size in sorted(results["bandwidth"]["mpi"].keys()):
        x_vals.append(size / 1000)

    for lib_name, lib_data in results["bandwidth"].items():

        baseline_runtimes = []
        uniconn_runtimes = []

        for size in sorted(lib_data.keys()):
            data_dict = lib_data[size]
            baseline_runtimes.append(
                np.mean(reject_outliers(data_dict.get("baseline")))
            )
            uniconn_runtimes.append(np.mean(reject_outliers(data_dict.get("uniconn"))))
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

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True)
    return line


def plot_latency(results, ax):

    line_width = 2
    line = []
    x_vals = []
    for size in sorted(results["latency"]["mpi"].keys()):
        x_vals.append(size / 1000)

    for lib_name, lib_data in results["latency"].items():

        baseline_runtimes = []
        uniconn_runtimes = []

        for size in sorted(lib_data.keys()):
            data_dict = lib_data[size]
            baseline_runtimes.append(
                np.mean(reject_outliers(data_dict.get("baseline")))
            )
            uniconn_runtimes.append(np.mean(reject_outliers(data_dict.get("uniconn"))))

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

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True)
    return line


def plot_diff_bandwidth(results, inset_ax):

    line_width = 2

    for exp_name, exp_data in results.items():

        inset_ax.set_yscale("linear")
        inset_ax.set_xscale("log")
        for lib_name, lib_data in exp_data.items():

            x_vals = []

            diff_runtimes = []
            for size in sorted(lib_data.keys()):
                x_vals.append(size / 1000)
                data_dict = lib_data[size]
                diff_runtimes.append(
                    (
                        np.mean(reject_outliers(data_dict.get("baseline")))
                        - np.mean(reject_outliers(data_dict.get("uniconn")))
                    )
                    / np.mean(reject_outliers(data_dict.get("baseline")))
                    * 100
                )

            print("Average Diff Bandwidth:" + lib_name)
            print(np.mean(diff_runtimes))
            inset_ax.semilogx(
                x_vals,
                diff_runtimes,
                color=color[lib_name],
                linestyle="-",
                linewidth=line_width,
                alpha=1,
            )

        # Final labeling

        inset_ax.set_xlabel("Message size (KB)")
        inset_ax.set_ylabel("Diff in %")
        inset_ax.grid(True)


def plot_diff_latency(results, inset_ax):

    line_width = 2

    for exp_name, exp_data in results.items():

        inset_ax.set_yscale("linear")
        inset_ax.set_xscale("log")
        for lib_name, lib_data in exp_data.items():

            x_vals = []

            diff_runtimes = []
            for size in sorted(lib_data.keys()):
                x_vals.append(size / 1000)
                data_dict = lib_data[size]
                diff_runtimes.append(
                    -(
                        np.mean(reject_outliers(data_dict.get("baseline")))
                        - np.mean(reject_outliers(data_dict.get("uniconn")))
                    )
                    / np.mean(reject_outliers(data_dict.get("baseline")))
                    * 100
                )
            print("Average Diff Latency:" + lib_name)
            print(np.mean(diff_runtimes))
            inset_ax.semilogx(
                x_vals,
                diff_runtimes,
                color=color[lib_name],
                linestyle="-",
                linewidth=line_width,
                alpha=1,
            )

        # Final labeling
        inset_ax.set_xlabel("Message size (KB)")
        inset_ax.set_ylabel("Diff in %")
        inset_ax.grid(True)


def main():

    # 1. Parse the file
    fig_filename = sys.argv[1]
    results_perlmutter_bandwidth = parse_experiment_file(sys.argv[2])
    results_perlmutter_latency = parse_experiment_file(sys.argv[3])
    results_lumi_bandwidth = parse_experiment_file(sys.argv[4])
    results_lumi_latency = parse_experiment_file(sys.argv[5])
    results_marenostrum_bandwidth = parse_experiment_file(sys.argv[6])
    results_marenostrum_latency = parse_experiment_file(sys.argv[7])

    # 2. Plot all groups on one figure
    plt.style.use("seaborn-v0_8-paper")

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), constrained_layout=True)

    print("Perlmutter")
    curr_ax = axs[0][0]
    line = plot_latency(results_perlmutter_latency, ax=curr_ax)
    curr_ax.set_title("Perlmutter", fontsize=12, fontweight="bold")
    curr_ax.set_ylabel("Time/Iteration (μs)", fontsize=12)

    inset_ax = inset_axes(curr_ax, width="40%", height="30%", loc="upper center")
    plot_diff_latency(results_perlmutter_latency, inset_ax)

    curr_ax = axs[1][0]
    plot_bandwidth(results_perlmutter_bandwidth, ax=curr_ax)
    inset_ax = inset_axes(curr_ax, width="40%", height="30%", loc="right")
    curr_ax.set_xlabel("Message size (KB)", fontsize=12)
    curr_ax.set_ylabel("Bandwidth (MB/s)", fontsize=12)

    plot_diff_bandwidth(results_perlmutter_bandwidth, inset_ax)

    print("Lumi")
    curr_ax = axs[0][1]
    plot_latency(results_lumi_latency, ax=curr_ax)
    curr_ax.set_title("Lumi", fontsize=12, fontweight="bold")

    inset_ax = inset_axes(curr_ax, width="40%", height="30%", loc="upper center")
    plot_diff_latency(results_lumi_latency, inset_ax)

    curr_ax = axs[1][1]
    plot_bandwidth(results_lumi_bandwidth, ax=curr_ax)

    curr_ax.set_xlabel("Message size (KB)", fontsize=12)
    inset_ax = inset_axes(curr_ax, width="40%", height="30%", loc="right")
    plot_diff_bandwidth(results_lumi_bandwidth, inset_ax)

    print("Marenostrum")
    curr_ax = axs[0][2]
    plot_latency(results_marenostrum_latency, ax=curr_ax)  # results_marenostrum_latency
    curr_ax.set_title("Marenostrum 5", fontsize=12, fontweight="bold")

    inset_ax = inset_axes(curr_ax, width="40%", height="30%", loc="upper center")
    plot_diff_latency(results_marenostrum_latency, inset_ax)

    curr_ax = axs[1][2]
    plot_bandwidth(results_marenostrum_bandwidth, ax=curr_ax)
    curr_ax.set_xlabel("Message size (KB)", fontsize=12)

    inset_ax = inset_axes(curr_ax, width="40%", height="30%", loc="right")
    plot_diff_bandwidth(results_marenostrum_bandwidth, inset_ax)

    fig.legend(
        handles=line,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        bbox_transform=fig.transFigure,
        fontsize=12,
    )

    plt.savefig(fig_filename + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
