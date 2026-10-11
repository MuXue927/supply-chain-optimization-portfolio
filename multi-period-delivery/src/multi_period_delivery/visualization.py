"""
visualization
=============

Module role
    Provide visualization outputs for solution processes and result analysis:
      1. Objective-value curves.
      2. Gap convergence curves.
      3. Operator usage and performance statistics (stacked horizontal bars
         plus tables).
      4. Interactive 3D supply-chain network (interactive Plotly HTML).

Design principles
    - Functions are independent and do not use global state.
    - Inputs are read-only; passed objects are not modified.
    - Output directories (images) are created automatically to avoid save
      failures.
    - Matplotlib figures are closed after saving to prevent memory leaks.
    - Plotly HTML includes JavaScript enhancements for edge-click highlighting
      and the information panel.

Dependencies
    - numpy / matplotlib / networkx / plotly
    - DataALNS: builds supply-chain data for the 3D network graph.

Main functions
    plot_objective_changes(result, output_file_loc, tracker=None)
        Prefer the paired search-objective and actual-objective curves. Fall
        back to result.plot_objectives() when tracker data is unavailable.

    plot_gap_changes(tracker, output_file_loc)
        Plot two kinds of gap:
          A. Gap between the current solution and the initial solution.
          B. Improvement percentage from a new best solution over the
             previous best.

    plot_operator_performance(state, result)
        Combine four subplots (two stacked-bar groups and two table groups)
        to show destroy and repair operator performance.

    plot_supply_chain_network_3d_enhanced(data, output_file_loc)
        Build a 3D plant-to-dealer supply network and generate HTML with
        edge-click highlighting.

Notes
    - Core function behavior is unchanged. The cleanup only:
        * Normalizes imports.
        * Adds directory existence checks.
        * Adds detailed documentation and comments.
        * Removes unused imports (fm, Patch).
    - Implemented improvements: dynamic column count, adaptive font scaling,
      and optional moving-average gap smoothing.
"""

# =========================
# Standard library
# =========================
import os

# =========================
# Third-party libraries
# =========================
import numpy as np

# =========================
# Internal project modules
# =========================
from .InputDataALNS import DataALNS
from .alns_config import default_config as ALNSConfig


# ---------------------------------------------------------------------
# Basic curve plotting
# ---------------------------------------------------------------------
def _ensure_image_dir(output_file_loc: str) -> str:
    """
    Ensure the images subdirectory exists under the output directory and
    return its path.
    """
    img_dir = os.path.join(output_file_loc, "images")
    os.makedirs(img_dir, exist_ok=True)
    return img_dir


def _moving_average(series, window: int):
    """Simple moving average; return a copy when window <= 1."""
    arr = np.asarray(series, dtype=float)
    if window <= 1 or arr.size == 0 or window > arr.size:
        return arr.copy(), np.arange(1, arr.size + 1)
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Align x-coordinates to the window center.
    start = window // 2
    x = np.arange(start + 1, start + 1 + smoothed.size)
    return smoothed, x


def _reconstruct_actual_objectives_from_tracker(tracker):
    """
    Extract or reconstruct the actual-objective series from tracker data.

    Priority:
      1) tracker.actual_objectives / actual_objectives_history
      2) Reconstruct from tracker.gaps and tracker.initial_actual_obj.
    """
    for attr in ("actual_objectives", "actual_objectives_history"):
        values = getattr(tracker, attr, None)
        if isinstance(values, list) and len(values) > 0:
            return np.asarray(values, dtype=float)

    gaps = getattr(tracker, "gaps", None)
    init_actual = getattr(tracker, "initial_actual_obj", None)
    if not isinstance(gaps, list) or len(gaps) == 0:
        return np.asarray([], dtype=float)
    if init_actual is None:
        return np.asarray([], dtype=float)
    try:
        base = float(init_actual)
        if not np.isfinite(base):
            return np.asarray([], dtype=float)
        # gap(%) = (current - initial) / |initial| * 100
        return np.asarray([base + (g / 100.0) * abs(base) for g in gaps], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _best_improvement_series_from_objectives(objectives):
    """
    Build the improvement percentage of each new best objective over the
    previous best objective.

    Returns:
      x_best: one-based iteration indices where a new best was found.
      imp_best: corresponding improvement percentages.
    """
    arr = np.asarray(objectives, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    best = float("inf")
    x_best = []
    imp_best = []
    for i, val in enumerate(arr, start=1):
        if not np.isfinite(val):
            continue
        if val < best:
            if np.isfinite(best) and abs(best) > 1e-12:
                imp = (best - val) / abs(best) * 100.0
                x_best.append(i)
                imp_best.append(float(imp))
            best = val
    return np.asarray(x_best, dtype=int), np.asarray(imp_best, dtype=float)


def plot_objective_changes(
    result,
    output_file_loc: str,
    tracker=None,
    smooth_window: int = 1,
    show_raw: bool = True,
) -> None:
    """
        Plot objective-value changes.
        Prefer plotting:
            - Search Objective
            - Actual Objective (Gurobi-aligned)
        Fall back to result.plot_objectives() when tracker data is unavailable.

    Parameters:
        result            Object providing plot_objectives(title=...).
        output_file_loc   Output root directory.

    Output:
        Save an SVG to images/Objective.svg.
    """
    _ensure_image_dir(output_file_loc)
    import matplotlib.pyplot as plt

    # Fallback: use the original logic when tracker data is unavailable.
    if tracker is None or not hasattr(tracker, "objectives"):
        plt.figure(figsize=(10, 6))
        result.plot_objectives(title="Changes of Objective")
        file_path = os.path.join(output_file_loc, "images", "Objective.svg")
        plt.savefig(file_path, dpi=600, bbox_inches="tight")
        plt.close()
        return

    search_obj = np.asarray(getattr(tracker, "objectives", []) or [], dtype=float)
    actual_obj = _reconstruct_actual_objectives_from_tracker(tracker)

    if search_obj.size == 0 and actual_obj.size == 0:
        plt.figure(figsize=(10, 6))
        result.plot_objectives(title="Changes of Objective")
        file_path = os.path.join(output_file_loc, "images", "Objective.svg")
        plt.savefig(file_path, dpi=600, bbox_inches="tight")
        plt.close()
        return

    n = int(max(search_obj.size, actual_obj.size))
    iters = np.arange(1, n + 1)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Search objective (left axis)
    if search_obj.size > 0:
        x_s = np.arange(1, search_obj.size + 1)
        if show_raw:
            ax.plot(
                x_s,
                search_obj,
                color="#1f77b4",
                alpha=0.35,
                linewidth=1.1,
                label="Search Objective (raw)",
            )
        s_sm, xs_sm = _moving_average(search_obj, smooth_window)
        ax.plot(
            xs_sm,
            s_sm,
            color="#1f77b4",
            linewidth=2.0,
            label=(
                "Search Objective"
                if smooth_window <= 1
                else f"Search Objective (smoothed, w={smooth_window})"
            ),
        )
        ax.set_ylabel("Search objective", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")

    # Actual objective (right axis)
    if actual_obj.size > 0:
        ax2 = ax.twinx()
        x_a = np.arange(1, actual_obj.size + 1)
        if show_raw:
            ax2.plot(
                x_a,
                actual_obj,
                color="#d62728",
                alpha=0.35,
                linewidth=1.1,
                label="Actual Objective (raw)",
            )
        a_sm, xa_sm = _moving_average(actual_obj, smooth_window)
        ax2.plot(
            xa_sm,
            a_sm,
            color="#d62728",
            linewidth=2.0,
            label=(
                "Actual Objective"
                if smooth_window <= 1
                else f"Actual Objective (smoothed, w={smooth_window})"
            ),
        )
        ax2.set_ylabel("Actual objective", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    ax.set_title("Objective Convergence (Search vs Actual)")
    ax.set_xlabel("Iteration (#)")
    # Avoid identical x-limits warning when only one iteration point is available.
    ax.set_xlim(1, max(2, n))
    ax.grid(True, linestyle="--", alpha=0.7)

    file_path = os.path.join(output_file_loc, "images", "Objective.svg")
    plt.savefig(file_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_gap_changes(
    tracker, output_file_loc: str, smooth_window: int = 1, show_raw: bool = True
) -> None:
    """
        Plot gap convergence curves with optional smoothing.

        The plot has two parts:
            1) Gap between the current solution and the initial solution.
            2) Improvement percentage from a new best over the previous best.

    Parameters:
        tracker         Tracker containing gaps (list[float]).
        output_file_loc Output root directory.
        smooth_window   Smoothing window; >=2 enables a simple moving average,
                        while 1 disables smoothing.
        show_raw        Whether to show the raw curve in light color.

    Output:
        images/Gaps.svg

    Behavior:
        - Raw curve: uses raw_color when visible or grayscale transparency in
          show_raw mode.
        - Smoothed curve: uses smoothed_color with a thicker line.
        - If smooth_window > len(gaps), fall back to plotting only the raw
          curve.
    """
    _ensure_image_dir(output_file_loc)
    gaps = np.asarray(getattr(tracker, "gaps", []) or [], dtype=float)
    objectives = np.asarray(getattr(tracker, "objectives", []) or [], dtype=float)
    if gaps.size == 0 and objectives.size == 0:
        return

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    # (A) Current vs Initial
    if gaps.size > 0:
        xg = np.arange(1, gaps.size + 1)
        if show_raw:
            ax1.plot(
                xg,
                gaps,
                color="#d62728",
                alpha=0.35,
                linewidth=1.0,
                label="Current vs Initial (raw)",
            )
        g_sm, xg_sm = _moving_average(gaps, smooth_window)
        ax1.plot(
            xg_sm,
            g_sm,
            color="#1f77b4",
            linewidth=2.0,
            label=(
                "Current vs Initial"
                if smooth_window <= 1
                else f"Current vs Initial (smoothed, w={smooth_window})"
            ),
        )
        ax1.axhline(0.0, color="#777", linestyle="--", linewidth=1.0)
        ax1.set_ylabel("Gap (%)")
        ax1.set_title("Gap A: Current Solution vs Initial Solution")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="best")
    else:
        ax1.text(0.5, 0.5, "No gap history available", ha="center", va="center")

    # (B) New best vs previous best
    xb, imp = _best_improvement_series_from_objectives(objectives)
    if imp.size > 0:
        ax2.vlines(xb, 0.0, imp, color="#2ca02c", alpha=0.35, linewidth=1.2)
        ax2.plot(
            xb,
            imp,
            color="#2ca02c",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="New best vs prev best",
        )
        ax2.set_ylabel("Improvement (%)")
        ax2.set_xlabel("Iteration (#)")
        ax2.set_title("Gap B: New Best Feasible vs Previous Best")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="best")
    else:
        ax2.text(
            0.5,
            0.5,
            "No best-improvement events found",
            ha="center",
            va="center",
        )
        ax2.set_xlabel("Iteration (#)")

    plt.tight_layout()
    file_path = os.path.join(output_file_loc, "images", "Gaps.svg")
    plt.savefig(file_path, dpi=600, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Operator-performance visualization
# ---------------------------------------------------------------------
def plot_operator_group(
    ax, operator_counts, title: str, status_labels: list[str], colors: list[str]
) -> None:
    """
    Plot one operator group (destroy / repair) as a stacked horizontal bar
    chart with dynamic column count.

    Parameters:
        ax              Matplotlib Axes
        operator_counts dict[str, sequence]  (s1, s2, ..., sK) per operator.
        title           Subplot title.
        status_labels   Status label for each column (length = K).
        colors          Color list (length >= K; callers must extend if
                        needed).

    Behavior:
        - If an operator count sequence is shorter than K, pad it with zeros.
        - Display all K columns as a stack.
    """
    operator_names = [name.replace("_", " ").title() for name in operator_counts.keys()]
    K = len(status_labels)
    if not operator_counts or K == 0:
        ax.set_title(title, pad=20, weight="bold")
        ax.text(
            0.5,
            0.5,
            "No operator statistics available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xlabel("Iterations (#)", labelpad=10)
        ax.set_ylabel("Operator", labelpad=10)
        ax.grid(False)
        return

    rows = []
    for counts in operator_counts.values():
        c = list(counts or [])
        if len(c) < K:
            c += [0] * (K - len(c))
        else:
            c = c[:K]
        rows.append(c)
    counts_array = np.array(rows, dtype=float)
    cumulative = counts_array.cumsum(axis=1)

    for idx in range(K):
        widths = counts_array[:, idx]
        starts = cumulative[:, idx] - widths
        ax.barh(
            operator_names,
            widths,
            left=starts,
            height=0.5,
            color=colors[idx],
            label=status_labels[idx],
        )

    ax.set_title(title, pad=20, weight="bold")
    ax.set_xlabel("Iterations (#)", labelpad=10)
    ax.set_ylabel("Operator", labelpad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_operator_performance(state, result, cmap_name: str = "tab20") -> None:
    """
    Visualize operator performance (destroy / repair) with dynamic status
    columns and adaptive fonts:
        - Detect the number of status columns from the counts sequence length.
        - Extend labels and colors when there are more than four statuses.
        - Automatically reduce table font size when there are many rows.

    Parameters:
        state       Current solution (must contain state.data.output_file_loc).
        result      Statistics object with destroy_operator_counts and
                    repair_operator_counts.
        cmap_name   Matplotlib colormap used when extra colors are needed.

    Output:
        images/Performance of Operators.svg
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 2, width_ratios=[1.8, 1.2], height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax_table1 = plt.subplot(gs[0, 1])
    ax_table2 = plt.subplot(gs[1, 1])

    destroy_counts = result.statistics.destroy_operator_counts
    repair_counts = result.statistics.repair_operator_counts

    # Determine the status column count from the larger of the two groups.
    def _status_len(count_dict):
        return len(next(iter(count_dict.values()))) if count_dict else 0

    K = max(_status_len(destroy_counts), _status_len(repair_counts))
    if K == 0:
        return

    base_labels = ["Best", "Better", "Accepted", "Rejected"]
    if K <= len(base_labels):
        status_labels = base_labels[:K]
    else:
        extra = [f"S{i}" for i in range(len(base_labels) + 1, K + 1)]
        status_labels = base_labels + extra

    # Color generation
    base_colors = ["#3C9BC9", "#76CBB4", "#FC757B", "#FFE59B"]
    if K <= len(base_colors):
        colors = base_colors[:K]
    else:
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(K)]

    # Draw stacked bars
    plot_operator_group(ax1, destroy_counts, "Destroy operators", status_labels, colors)
    plot_operator_group(ax2, repair_counts, "Repair operators", status_labels, colors)

    # Prepare table data (pad with zeros for alignment)
    def _pad_counts(counts_list, K):
        c = list(counts_list)
        if len(c) < K:
            c += [0] * (K - len(c))
        else:
            c = c[:K]
        return c

    destroy_data = [
        [op_name.replace("_", " ").title()] + _pad_counts(counts, K)
        for op_name, counts in destroy_counts.items()
    ]
    repair_data = [
        [op_name.replace("_", " ").title()] + _pad_counts(counts, K)
        for op_name, counts in repair_counts.items()
    ]
    columns = ["Operator"] + status_labels

    ax_table1.axis("off")
    ax_table2.axis("off")

    # Dynamic column widths
    max_width_first_col = max(
        [len(str(row[0])) for row in destroy_data]
        + [len(str(row[0])) for row in repair_data]
        + [len(columns[0])]
    )
    other_col_widths = []
    for i in range(1, K + 1):
        other_col_widths.append(
            max(
                [len(str(row[i])) for row in destroy_data]
                + [len(str(row[i])) for row in repair_data]
                + [len(columns[i])]
            )
        )
    col_widths = [max_width_first_col / 8]
    col_widths.extend([w / 6 + 0.15 for w in other_col_widths])

    # Adaptive font sizing
    max_rows = max(len(destroy_data), len(repair_data))
    if max_rows <= 10:
        fsize = 9
    elif max_rows <= 20:
        fsize = 8
    elif max_rows <= 30:
        fsize = 7
    else:
        fsize = 6

    def _build_table(ax_table, data):
        tbl = ax_table.table(
            cellText=data,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            bbox=[0.1, 0.15, 1.1, 0.6],
            colWidths=col_widths,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fsize)
        # Scale row height moderately based on the status column count.
        scale_y = 1.2 if max_rows < 18 else 1.0
        tbl.scale(1.0, scale_y)
        for (i, j), cell in tbl._cells.items():
            if i == 0:
                cell.set_facecolor("#E6E6E6")
                cell.set_text_props(weight="bold")
            if j == 0:
                cell.set_text_props(wrap=True)
        return tbl

    _build_table(ax_table1, destroy_data)
    _build_table(ax_table2, repair_data)

    ax_table1.text(
        0.5,
        0.85,
        "Destroy Operators Performance",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax_table2.text(
        0.5,
        0.85,
        "Repair Operators Performance",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    plt.subplots_adjust(
        left=0.1, right=0.98, bottom=0.2, top=0.92, wspace=0.3, hspace=0.4
    )

    # Legend (all statuses)
    handles, _ = ax1.get_legend_handles_labels()
    plt.figlegend(
        handles,
        status_labels,
        loc="lower center",
        ncol=min(K, 6),
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    plt.suptitle("Performance of Operators", y=0.98, weight="bold")

    file_loc = _ensure_image_dir(state.data.output_file_loc)
    file = os.path.join(file_loc, "Performance of Operators.svg")
    plt.savefig(file, dpi=600, bbox_inches="tight", pad_inches=0.2)
    plt.close()


# ---------------------------------------------------------------------
# Interactive 3D supply-chain network
# ---------------------------------------------------------------------
def plot_supply_chain_network_3d_enhanced(data: DataALNS, output_file_loc: str) -> None:
    """
    Build an interactive 3D plant-to-dealer supply network with Plotly.

    Nodes: plant / dealer.
    Edge attributes: total_supply (plant availability: initial inventory plus
    production) and total_demand (dealer demand for shared SKUs).

    Interactions:
      - Hover to show edge metrics.
      - Click an edge to highlight it in red and show details in the upper-right
        information panel.
      - Double-click or click the background to reset.

    Output:
        images/supply_chain_network_3d_enhanced.html
    """
    # 1. Graph construction
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.io as pio

    supply_chain = data.construct_supply_chain()
    G = nx.DiGraph()
    for plant in data.plants:
        G.add_node(plant, node_type="plant")
    for dealer in data.dealers:
        G.add_node(dealer, node_type="dealer")

    for (plant, dealer), skus in supply_chain.items():
        if skus:
            total_demand = sum(data.demands.get((dealer, sku), 0) for sku in skus)
            initial_inv = sum(data.sku_initial_inv.get((plant, sku), 0) for sku in skus)
            total_prod = sum(
                data.sku_prod_each_day.get((plant, sku, day), 0)
                for sku in skus
                for day in range(1, data.horizons + 1)
            )
            total_supply = initial_inv + total_prod
            G.add_edge(
                plant, dealer, total_supply=total_supply, total_demand=total_demand
            )

    # 2. Layout (3D spring)
    pos = nx.spring_layout(G, dim=3, k=1.5, iterations=50, seed=42)
    node_xyz = np.array([pos[v] for v in G.nodes()])
    edge_xyz = [(pos[u], pos[v]) for u, v in G.edges()]

    def create_node_trace(nodelist, color, symbol, size, name):
        """
        Build a node scatter trace.
        Color determines node type for hover text.
        """
        idx = [list(G.nodes()).index(n) for n in nodelist]
        return go.Scatter3d(
            x=node_xyz[idx, 0],
            y=node_xyz[idx, 1],
            z=node_xyz[idx, 2],
            mode="markers",
            marker=dict(
                symbol=symbol, size=size, color=color, line=dict(width=2, color="#444")
            ),
            hoverinfo="text",
            text=[
                f"{n} ({'Plant' if color == '#FF6B6B' else 'Dealer'})" for n in nodelist
            ],
            name=name,
        )

    # 3. Edge traces
    edge_traces = []
    for (u, v), (start, end) in zip(G.edges(), edge_xyz):
        x0, y0, z0 = start
        x1, y1, z1 = end
        edge_info = G.edges[u, v]
        hover_text = (
            f"<b>{u} → {v}</b><br>"
            f"Total Supply: {edge_info['total_supply']}<br>"
            f"Total Demand: {edge_info['total_demand']}"
        )
        custom = [
            [
                edge_info["total_supply"],
                edge_info["total_demand"],
                str(u),
                str(v),
            ],
            [
                edge_info["total_supply"],
                edge_info["total_demand"],
                str(u),
                str(v),
            ],
        ]
        edge_traces.append(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(color="rgba(100,100,100,0.7)", width=4),
                hoverinfo="text",
                text=[hover_text, hover_text],
                customdata=custom,
                hoverlabel=dict(bgcolor="white", font_size=12),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # 4. Combined figure
    fig = go.Figure(
        data=edge_traces
        + [
            create_node_trace(
                [n for n, attr in G.nodes(data=True) if attr["node_type"] == "plant"],
                "#FF6B6B",
                "square",
                16,
                "Plant",
            ),
            create_node_trace(
                [n for n, attr in G.nodes(data=True) if attr["node_type"] == "dealer"],
                "#4ECDC4",
                "circle",
                10,
                "Dealer",
            ),
        ]
    )

    fig.update_layout(
        title="Enhanced 3D Supply Chain Network Diagram",
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#888",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        ),
        hovermode="closest",
    )

    # 5. Save HTML and inject interaction JavaScript
    images_dir = _ensure_image_dir(output_file_loc)
    html_path = os.path.join(images_dir, "supply_chain_network_3d_enhanced.html")

    fig_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    post_script = """
    <div id='supply_chain_edge_info' style='position:fixed; right:20px; top:20px; z-index:9999;
         background:#ffffffcc; border:1px solid #ccc; padding:12px; border-radius:6px;
         box-shadow:0 4px 14px rgba(0,0,0,0.15); max-width:320px; font-family:Arial, Helvetica, sans-serif;'>
      <strong>Selected edge info</strong>
      <div id='edge_info_content' style='margin-top:8px; font-size:13px; color:#111;'>Click an edge to see details.</div>
    </div>
    <script>
    (function() {
        var gd = document.querySelector('[id^="plotly"]');
        if(!gd) return;
        var defaultColor = 'rgba(100,100,100,0.7)';
        var defaultWidth = 4;

        gd.on('plotly_click', function(data) {
            try {
                var pt = data.points[0];
                var traceIndex = pt.curveNumber;

                Plotly.restyle(gd, {'line.color': defaultColor, 'line.width': defaultWidth});
                Plotly.restyle(gd, {'line.color': 'rgba(255,0,0,0.95)', 'line.width': 8}, [traceIndex]);

                var custom = pt.customdata || (pt.data.customdata && pt.data.customdata[pt.pointIndex]);
                var total_supply = custom[0];
                var total_demand = custom[1];
                var from_node = custom[2];
                var to_node = custom[3];

                var html = '<div><b>' + from_node + ' → ' + to_node + '</b></div>' +
                           '<div style="margin-top:6px;">Total supply (plant across all periods): <b>' + total_supply + '</b></div>' +
                           '<div style="margin-top:4px;">Total demand (dealer for these SKUs): <b>' + total_demand + '</b></div>';
                document.getElementById('edge_info_content').innerHTML = html;
            } catch(e) {
                console.error('Error handling plotly_click', e);
            }
        });

        function resetEdgeHighlight() {
            Plotly.restyle(gd, {'line.color': defaultColor, 'line.width': defaultWidth});
            document.getElementById('edge_info_content').innerHTML = 'Click an edge to see details.';
        }

        gd.on('plotly_clickannotation', resetEdgeHighlight);
        gd.on('plotly_doubleclick', resetEdgeHighlight);
    })();
    </script>
    """

    full_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Supply Chain 3D Enhanced</title></head>"
        "<body style='margin:0;'>"
        f"{fig_div}{post_script}"
        "</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
