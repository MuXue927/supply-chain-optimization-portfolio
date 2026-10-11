import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLORS = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759"]


DATASETS = {
    "medium_improvement": {
        "title": "中规模数据集",
        "metric": "改进率 (%)",
        "data": {
            "ALNS-B": [
                5.52,
                4.75,
                1.35,
                6.26,
                5.68,
                0.28,
                1.00,
                6.74,
                0.23,
                1.23,
                4.93,
                4.15,
                6.50,
                1.32,
                1.22,
                3.44,
                4.23,
                1.63,
                0.52,
                8.68,
            ],
            "ALNS-F": [
                4.47,
                4.43,
                0.93,
                4.82,
                7.47,
                -0.71,
                2.40,
                3.07,
                0.00,
                1.23,
                0.63,
                4.07,
                3.29,
                3.62,
                2.30,
                3.02,
                8.77,
                1.02,
                2.99,
                3.26,
            ],
            "ALNS-ML": [
                6.49,
                4.43,
                0.93,
                4.82,
                7.47,
                -0.71,
                2.40,
                3.07,
                0.00,
                1.23,
                0.63,
                4.07,
                3.29,
                3.62,
                2.30,
                3.02,
                8.77,
                1.02,
                2.99,
                3.26,
            ],
            "ALNS-T": [
                4.54,
                8.86,
                4.42,
                1.44,
                1.07,
                0.77,
                -0.20,
                2.50,
                0.23,
                5.92,
                0.96,
                3.89,
                3.08,
                0.87,
                5.87,
                -0.56,
                3.70,
                1.63,
                0.40,
                5.42,
            ],
        },
    },
    "large_improvement": {
        "title": "大规模数据集",
        "metric": "改进率 (%)",
        "data": {
            "ALNS-B": [
                0.34,
                -0.14,
                0.26,
                1.39,
                3.39,
                -0.09,
                1.94,
                -0.78,
                0.59,
                0.39,
                -0.58,
                0.14,
                4.12,
                2.39,
                6.67,
                1.75,
                1.76,
                -1.32,
                -0.61,
                1.48,
            ],
            "ALNS-F": [
                5.91,
                0.58,
                0.07,
                0.54,
                0.79,
                4.12,
                -0.55,
                0.99,
                5.24,
                0.39,
                -0.70,
                0.28,
                -0.08,
                1.15,
                -1.00,
                4.51,
                -0.15,
                -1.40,
                -0.61,
                1.72,
            ],
            "ALNS-ML": [
                5.91,
                0.58,
                0.07,
                0.54,
                0.79,
                4.12,
                -0.55,
                0.00,
                5.24,
                0.39,
                -0.70,
                0.28,
                -0.08,
                1.15,
                -1.00,
                4.51,
                -0.15,
                -1.40,
                -0.61,
                1.72,
            ],
            "ALNS-T": [
                3.01,
                -0.25,
                0.07,
                2.73,
                3.21,
                0.00,
                -0.51,
                -0.78,
                0.59,
                0.55,
                -0.58,
                0.43,
                1.28,
                3.60,
                -1.18,
                3.12,
                3.76,
                -1.40,
                -0.68,
                0.20,
            ],
        },
    },
    "small_gap": {
        "title": "小规模数据集",
        "metric": "相对 ILP 的 Gap (%)",
        "data": {
            "ALNS-B": [
                95.13499,
                13.69408,
                127.93054,
                71.98364,
                -64.36791,
                63.98754,
                38.23529,
                16.80882,
                49.25373,
                -40.39924,
                -45.69150,
                10.00670,
                -90.42655,
                44.17532,
                -93.77114,
                62.11293,
                -7.23355,
            ],
            "ALNS-F": [
                79.09650,
                21.01010,
                156.87410,
                54.19223,
                -62.61600,
                34.39252,
                30.88235,
                18.10786,
                63.24627,
                -36.78707,
                -46.15797,
                0.51374,
                -90.81303,
                26.87428,
                -93.60842,
                76.68488,
                -19.91602,
            ],
            "ALNS-ML": [
                79.09650,
                21.01010,
                156.87410,
                54.19223,
                -62.61600,
                34.39252,
                30.88235,
                18.10786,
                63.24627,
                -36.78707,
                -46.15797,
                0.51374,
                -90.81303,
                26.87428,
                -93.60842,
                76.68488,
                -19.91602,
            ],
            "ALNS-T": [
                79.09650,
                -1.15440,
                80.89725,
                71.77914,
                -63.66519,
                44.48598,
                20.58824,
                18.14722,
                81.90299,
                -40.49430,
                -45.42045,
                10.00670,
                -90.36484,
                58.59285,
                -93.74064,
                49.36248,
                2.18456,
            ],
        },
    },
}


def plot_box_scatter(
    data: dict,
    labels: list[str],
    values: list[list[float]],
    metric: str,
    title_prefix: str,
    out_prefix: str,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        whiskerprops={"color": "#4C4C4C", "linewidth": 1.2},
        medianprops={"color": "#222222", "linewidth": 1.8},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "white",
            "markeredgecolor": "#4C4C4C",
            "alpha": 0.95,
        },
    )

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for i, y in enumerate(values, start=1):
        arr = np.asarray(y, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        inlier_mask = (arr >= lower) & (arr <= upper)

        inlier_vals = arr[inlier_mask]
        x = rng.normal(i, 0.05, size=len(inlier_vals))
        ax.scatter(
            x,
            inlier_vals,
            s=28,
            alpha=0.75,
            color=COLORS[i - 1],
            edgecolors="white",
            linewidths=0.5,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    legend_handles = [
        Patch(facecolor="#A6BDD7", edgecolor="#4C4C4C", alpha=0.45, label="25%~75%"),
        Line2D([0], [0], color="#4C4C4C", lw=1.4, label="1.5IQR内的范围"),
        Line2D([0], [0], color="#222222", lw=1.8, label="中位线"),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="#4C4C4C",
            linestyle="None",
            label="异常值",
        ),
    ]

    ax.set_ylabel(metric)
    ax.set_xlabel("算法变体")
    ax.grid(False)
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    out_path = f"runExperiments/{out_prefix}_boxplot.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_violin_box(
    data: dict,
    labels: list[str],
    values: list[list[float]],
    metric: str,
    title_prefix: str,
    out_prefix: str,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    vp = ax.violinplot(values, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(COLORS[i])
        body.set_edgecolor("black")
        body.set_alpha(0.25)

    ax.boxplot(
        values, positions=np.arange(1, len(labels) + 1), widths=0.18, showmeans=True
    )

    means = [np.mean(v) for v in values]
    ax.scatter(
        np.arange(1, len(labels) + 1),
        means,
        marker="D",
        s=50,
        color="black",
        label="均值",
    )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{title_prefix} - 小提琴图 + 箱线图", pad=12)
    ax.set_ylabel(metric)
    ax.set_xlabel("算法变体")
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_path = f"runExperiments/{out_prefix}_violin_box.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_ecdf(data: dict, metric: str, title_prefix: str, out_prefix: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, vals) in enumerate(data.items()):
        arr = np.sort(np.asarray(vals, dtype=float))
        y = np.arange(1, len(arr) + 1) / len(arr)
        ax.step(arr, y, where="post", linewidth=2, color=COLORS[i], label=label)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{title_prefix} - ECDF 累积分布", pad=12)
    ax.set_xlabel(metric)
    ax.set_ylabel("累计概率")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = f"runExperiments/{out_prefix}_ecdf.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def print_summary(dataset_name: str, data: dict) -> None:
    print(f"\n统计摘要 - {dataset_name}")
    for variant, vals in data.items():
        arr = np.asarray(vals, dtype=float)
        print(
            f"{variant}: n={len(arr)}, mean={arr.mean():.3f}, median={np.median(arr):.3f}, "
            f"std={arr.std(ddof=1):.3f}, min={arr.min():.3f}, max={arr.max():.3f}"
        )


def plot_panel_3x3() -> str:
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    variant_labels = ["ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]
    panel_legend_handles = [
        Patch(facecolor="#A6BDD7", edgecolor="#4C4C4C", alpha=0.45, label="25%~75%"),
        Line2D([0], [0], color="#4C4C4C", lw=1.2, label="1.5IQR内的范围"),
        Line2D([0], [0], color="#222222", lw=1.6, label="中位线"),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor="#4C4C4C",
            linestyle="None",
            label="异常值",
        ),
    ]

    for row, dataset_key in enumerate(DATASETS.keys()):
        conf = DATASETS[dataset_key]
        data = conf["data"]
        labels = list(data.keys())
        values = [data[k] for k in labels]
        metric = conf["metric"]
        title = conf["title"]

        # Col 1: Box + scatter
        ax1 = axes[row, 0]
        bp = ax1.boxplot(
            values,
            tick_labels=labels,
            patch_artist=True,
            showmeans=True,
            whiskerprops={"color": "#4C4C4C", "linewidth": 1.1},
            medianprops={"color": "#222222", "linewidth": 1.6},
            flierprops={
                "marker": "o",
                "markersize": 3.8,
                "markerfacecolor": "white",
                "markeredgecolor": "#4C4C4C",
                "alpha": 0.95,
            },
        )
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

        rng = np.random.default_rng(42)
        for i, y in enumerate(values, start=1):
            arr = np.asarray(y, dtype=float)
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            inlier_mask = (arr >= lower) & (arr <= upper)

            inlier_vals = arr[inlier_mask]
            x = rng.normal(i, 0.05, size=len(inlier_vals))
            ax1.scatter(
                x,
                inlier_vals,
                s=16,
                alpha=0.70,
                color=COLORS[i - 1],
                edgecolors="white",
                linewidths=0.4,
            )
        ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax1.set_ylabel(metric)
        ax1.set_xlabel("算法变体")
        ax1.grid(False)

        # Col 2: Violin + box
        ax2 = axes[row, 1]
        vp = ax2.violinplot(
            values, showmeans=False, showmedians=False, showextrema=False
        )
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(COLORS[i])
            body.set_edgecolor("black")
            body.set_alpha(0.25)
        ax2.boxplot(
            values, positions=np.arange(1, len(labels) + 1), widths=0.18, showmeans=True
        )
        means = [np.mean(v) for v in values]
        ax2.scatter(
            np.arange(1, len(labels) + 1),
            means,
            marker="D",
            s=28,
            color="black",
            label="均值",
        )
        ax2.set_xticks(np.arange(1, len(labels) + 1))
        ax2.set_xticklabels(variant_labels)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax2.set_title(f"{title} - 小提琴+箱线", fontsize=11)
        ax2.set_ylabel(metric)
        ax2.set_xlabel("算法变体")
        ax2.legend(loc="upper left", fontsize=8)

        # Col 3: ECDF
        ax3 = axes[row, 2]
        for i, (label, vals) in enumerate(data.items()):
            arr = np.sort(np.asarray(vals, dtype=float))
            y = np.arange(1, len(arr) + 1) / len(arr)
            ax3.step(arr, y, where="post", linewidth=1.8, color=COLORS[i], label=label)
        ax3.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax3.set_title(f"{title} - ECDF", fontsize=11)
        ax3.set_xlabel(metric)
        ax3.set_ylabel("累计概率")
        ax3.legend(loc="lower right", fontsize=8)

    fig.suptitle("ALNS 各变体分布对比总览（3x3）", fontsize=16, y=0.995)
    fig.legend(
        handles=panel_legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.88, 0.985),
        fontsize=9,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.98))
    out_path = "runExperiments/alns_distribution_panel_3x3.png"
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return out_path


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    saved_files = []
    for dataset_key, conf in DATASETS.items():
        data = conf["data"]
        labels = list(data.keys())
        values = [data[k] for k in labels]

        saved_files.append(
            plot_box_scatter(
                data,
                labels,
                values,
                conf["metric"],
                conf["title"],
                dataset_key,
            )
        )
        saved_files.append(
            plot_violin_box(
                data,
                labels,
                values,
                conf["metric"],
                conf["title"],
                dataset_key,
            )
        )
        saved_files.append(
            plot_ecdf(
                data,
                conf["metric"],
                conf["title"],
                dataset_key,
            )
        )
        print_summary(dataset_key, data)

    saved_files.append(plot_panel_3x3())

    print("\n已保存文件:")
    for path in saved_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
