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


def reject_outliers(data, m=0):
    data = np.array(data)
    low = np.quantile(data, m / 2)
    high = np.quantile(data, 1 - m / 2)
    data = data[(data >= low) & (data <= high)]
    return data.tolist()


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
            matrix_name = parts[2]  # e.g. 128, 256, ...
            gpu_count = int(parts[3])
            perf_values = float(parts[4])
            if experiment not in results:
                results[experiment] = {}

            if lib != "unc":
                current_lib = lib
            if current_lib not in results[experiment]:
                results[experiment][current_lib] = {}

            if matrix_name not in results[experiment][current_lib]:
                results[experiment][current_lib][matrix_name] = {}
            if gpu_count not in results[experiment][current_lib][matrix_name]:
                results[experiment][current_lib][matrix_name][gpu_count] = {}
                results[experiment][current_lib][matrix_name][gpu_count][
                    "baseline"
                ] = []
                results[experiment][current_lib][matrix_name][gpu_count]["uniconn"] = []

            if lib != "unc":
                results[experiment][current_lib][matrix_name][gpu_count][
                    "baseline"
                ].append(perf_values)
            else:
                results[experiment][current_lib][matrix_name][gpu_count][
                    "uniconn"
                ].append(perf_values)

    return results


markers = {"mpi": "D", "gpuccl": "X", "nvshmem_d": "o", "nvshmem_h": "."}
color = {
    "mpi": "blue",
    "gpuccl": "green",
    "nvshmem_d": "magenta",
    "nvshmem_h": "orange",
}


def plot_runtimes(results, ax, mtx, gpu_count, title):

    handles = []
    edge_color = "black"
    line_width = 1.0
    # Bar positioning
    bar_width = 0.18  # Width of a single bar
    group_gap = 0.2  # Gap between groups of bars
    bar_gap = 0.02  # Gap between bars within a group

    n_methods = 2 * len(results["cg"].keys())

    index = np.arange(n_methods)  # The label locations
    total_group_width = n_methods * bar_width + (n_methods - 1) * bar_gap
    # group_starts = index - total_group_width / 2 + bar_width / 2
    i = 0
    for lib_name, lib_data in results["cg"].items():
        bar_position1 = (
            total_group_width / 2 + bar_width / 2 + i * (bar_width + bar_gap)
        )
        i += 1
        bar_position2 = (
            total_group_width / 2 + bar_width / 2 + i * (bar_width + bar_gap)
        )
        i += 1
        data_dict = lib_data[mtx][gpu_count]
        uniconn_res = np.mean(reject_outliers(data_dict.get("uniconn"))) / 1000
        baseline_res = np.mean(reject_outliers(data_dict.get("baseline"))) / 1000
        diff_percent = (
            (
                np.mean(reject_outliers(data_dict.get("uniconn")))
                - np.mean(reject_outliers(data_dict.get("baseline")))
            )
            / np.mean(reject_outliers(data_dict.get("baseline")))
            * 100
        )
        print(mtx + ", " + lib_name)
        print(diff_percent)

        ax.bar(
            bar_position1,
            baseline_res,
            bar_width,
            label=f"{library_map[lib_name]}: Native",
            color=color[lib_name],
            alpha=0.5,
            edgecolor=edge_color,
            linewidth=line_width,
        )
        ax.bar(
            bar_position2,
            uniconn_res,
            bar_width,
            label=f"{library_map[lib_name]}: Uniconn",
            color=color[lib_name],
            alpha=1,
            edgecolor=edge_color,
            linewidth=line_width,
        )
        # ax.bar_label(bars, fmt='%.2f',label_type="center")

    # # Final labeling
    # ax.set_xticks(x_vals, labels=x_vals)
    # ax.tick_params(axis="x", labelsize=10)
    # ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    # ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(mtx, fontsize=10)

    # ax.legend( fontsize=12)
    # ax.grid(True)
    return ax.get_legend_handles_labels()


def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    # filename3 = sys.argv[3]

    # 1. Parse the file
    results_perlmutter = parse_experiment_file(filename1)
    results_lumi = parse_experiment_file(filename2)
    # results_marenostrum = parse_experiment_file(filename3)

    # 2. Plot all groups on one figure
    plt.style.use("seaborn-v0_8-paper")

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 3), constrained_layout=True)

    handles, labels = plot_runtimes(
        results_perlmutter, axs[0], "Serena", 8, title="Perlmutter"
    )
    plot_runtimes(results_perlmutter, axs[1], "Queen_4147", 8, title="")

    plot_runtimes(results_lumi, axs[2], "Serena", 8, title="LUMI")
    plot_runtimes(results_lumi, axs[3], "Queen_4147", 8, title="")
    # print("Marenostrum")
    # # plot_runtimes(results_marenostrum, axs[4], "Serena", 8, title="Marenostrum 5 (tmp)")
    # # plot_runtimes(results_marenostrum, axs[5], "Queen_4147", 8, title="")

    fig.supylabel("Time (s)", fontsize=10)
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        bbox_transform=fig.transFigure,
        fontsize=12,
    )

    plt.savefig("CG_diff" + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
