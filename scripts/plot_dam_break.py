"""
plot_dam_break.py
=================
Generates comparison plots for dam-break simulation runs.

Folder layout expected under INPUT_DIR:
  dam_break_sim_dam_break_nx_<N>/   ← vary spatial resolution
  dam_break_sim_dam_break_nt_<N>/   ← vary time resolution
  dam_break_sim_dam_break_flip_<N>/ ← vary flip parameter
  dam_break_sim_dam_break_base/     ← baseline run

Each folder must contain:
  <folder_name>.csv   – simulation output
  <folder_name>.json  – run parameters (must include "delta_t")

Output PNGs are saved to OUTPUT_DIR.
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
#  USER-CONFIGURABLE PARAMETERS
# ─────────────────────────────────────────────

INPUT_DIR = "../dam_break_apic_out"  # folder that contains all the sim sub-folders
OUTPUT_DIR = "plots"  # where PNG files are written

# Figure size (width, height) in inches
FIG_SIZE = (10, 5)

# DPI for saved figures
DPI = 150

# Whether to include the "base" run on every comparison plot
INCLUDE_BASE_IN_ALL = True

# Line style cycling for multiple runs
LINE_STYLES = ["-", "--", "-.", ":"]

# Colour cycle override (None → use matplotlib default)
COLORS = None  # e.g. ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# X-axis label
X_LABEL = "Time (s)"

# Cut the x-axis at this time value (seconds).  Set to None to show everything.
X_MAX = 0.6

# ── Per-category run filter ────────────────────────────────────────────────
# For each category, list the numeric values you want to keep on the plots.
# Set to None (or omit the key) to include ALL runs found for that category.
#
# Example: only plot nx=50 and nx=100  →  "nx": [50, 100]
#
FILTER = {
    "nx": None,  # e.g. [50, 100, 200]
    "nt": None,  # e.g. [1000, 2000, 5644]
    "flip": None,  # e.g. [0, 1, 2]
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

# Groups: regex captures the category ("nx"/"nt"/"flip"/"base") and the
# optional numeric value.
FOLDER_RE = re.compile(
    r"^dam_break_sim_dam_break_"
    r"(?P<category>nx|nt|flip|base)"
    r"(?:_(?P<value>\d+))?$"
)


def discover_runs(root):
    """
    Walk *root* and return a dict:
      { category: [ {name, csv_path, json_path, value}, … ] }
    where category ∈ {"nx", "nt", "flip", "base"}.
    Entries are sorted by numeric value (base has value=0).
    """
    runs = {"nx": [], "nt": [], "flip": [], "base": []}
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        m = FOLDER_RE.match(entry.name)
        if not m:
            continue
        cat = m.group("category")
        value = int(m.group("value")) if m.group("value") else 0
        csv_path = os.path.join(entry.path, entry.name + ".csv")
        json_path = os.path.join(entry.path, entry.name + ".json")
        if not os.path.isfile(csv_path) or not os.path.isfile(json_path):
            print(f"  [skip] {entry.name}: missing csv or json")
            continue
        runs[cat].append(
            {"name": entry.name, "csv": csv_path, "json": json_path, "value": value}
        )
    for cat in runs:
        runs[cat].sort(key=lambda r: r["value"])
        keep = FILTER.get(cat)
        if keep is not None:
            before = len(runs[cat])
            runs[cat] = [r for r in runs[cat] if r["value"] in keep]
            print(
                f"  [filter] {cat}: kept {len(runs[cat])}/{before} run(s) "
                f"matching values {keep}"
            )
    return runs


def load_run(run_info):
    """Return (DataFrame with 'time' column, delta_t).
    Rows beyond X_MAX are dropped early to keep memory low."""
    with open(run_info["json"]) as f:
        params = json.load(f)
    delta_t = params["delta_t"]

    df = pd.read_csv(run_info["csv"])
    df["time"] = df["step"] * delta_t
    if X_MAX is not None:
        df = df[df["time"] <= X_MAX]
    return df, delta_t


def apply_xlim(ax):
    """Set x-axis upper bound when X_MAX is configured."""
    if X_MAX is not None:
        ax.set_xlim(left=0, right=X_MAX)


def pressure_columns(df):
    """Return all column names that start with 'pressure_'."""
    return [c for c in df.columns if c.startswith("pressure_")]


def label_for(run_info):
    """Human-readable legend label."""
    cat = FOLDER_RE.match(run_info["name"]).group("category")
    val = run_info["value"]
    return f"base" if cat == "base" else f"{cat}={val}"


def _style(idx):
    ls = LINE_STYLES[idx % len(LINE_STYLES)]
    c = COLORS[idx % len(COLORS)] if COLORS else f"C{idx}"
    return dict(linestyle=ls, color=c)


# ─────────────────────────────────────────────
#  PLOT BUILDERS
# ─────────────────────────────────────────────


def plot_pressure(runs_in_group, base_runs, col_label, out_path, title):
    """
    One pressure-column comparison plot.

    *runs_in_group* – list of run-info dicts for the category being compared.
    *base_runs*     – list of run-info dicts for the base category (may be empty).
    *col_label*     – a display name for the pressure point, e.g. "pressure_266_3".
                      When nx changes the actual column name differs across runs;
                      we therefore match by *column index position* among pressure
                      columns.  col_label is used purely for the axis / title.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    all_runs = list(runs_in_group)
    if INCLUDE_BASE_IN_ALL:
        all_runs = list(base_runs) + all_runs

    for idx, run in enumerate(all_runs):
        df, _ = load_run(run)
        pcols = pressure_columns(df)

        # ── column matching ──────────────────────────────────────────────
        # col_label is the pressure column name from the *reference* run.
        # For runs with a different nx the column names change (different
        # node indices).  We fall back to matching by position (ordinal).
        if col_label in pcols:
            col = col_label
        else:
            # Determine the ordinal of col_label in the reference run's
            # pressure columns.  That ordinal is stored alongside col_label
            # in the outer loop (see plot_group); here we just pick the
            # same ordinal if available.
            #
            # This function receives the ordinal via the closure variable
            # `_col_ordinal` set by the caller (see below).
            ordinal = _col_ordinal  # noqa: F821  set by caller
            if ordinal < len(pcols):
                col = pcols[ordinal]
            else:
                print(
                    f"  [skip] {run['name']}: no pressure column at "
                    f"ordinal {ordinal}"
                )
                continue

        ax.plot(df["time"], df[col], label=label_for(run), linewidth=1.4, **_style(idx))

    ax.set_title(title)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(r"Pressure $[\text{Pa}]$")
    apply_xlim(ax)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.5, alpha=0.6)
    ax.grid(True, which="minor", linewidth=0.2, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_wave_front(runs_in_group, base_runs, out_path, title):
    """Wave-front position comparison plot."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    all_runs = list(runs_in_group)
    if INCLUDE_BASE_IN_ALL:
        all_runs = list(base_runs) + all_runs

    for idx, run in enumerate(all_runs):
        df, _ = load_run(run)
        if "wave_front" not in df.columns:
            print(f"  [skip] {run['name']}: no 'wave_front' column")
            continue
        ax.plot(
            df["time"],
            df["wave_front"],
            label=label_for(run),
            linewidth=1.4,
            **_style(idx),
        )

    ax.set_title(title)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(r"Wave front position $[\text{m}]$")
    apply_xlim(ax)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.5, alpha=0.6)
    ax.grid(True, which="minor", linewidth=0.2, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────

# Upper time bound used for shock-pressure search (seconds).
# Defaults to X_MAX when set, otherwise use this fallback.
SHOCK_PRESSURE_T_MAX = 0.6


def compute_stats(run_info):
    """
    Return a dict of key metrics for one run (uses the *full* CSV, not
    the X_MAX-clipped version, so that wave-front arrival is never missed
    even when X_MAX is shorter than the domain-crossing time).
    """
    with open(run_info["json"]) as f:
        params = json.load(f)
    delta_t = params["delta_t"]

    df = pd.read_csv(run_info["csv"])
    df["time"] = df["step"] * delta_t

    stats = {}

    # ── Wave-front arrival time ───────────────────────────────────────
    # "End of domain" = the last recorded wave_front value (assumed to be
    # the maximum, i.e. the front never retreats past the domain boundary).
    if "wave_front" in df.columns:
        end_position = df["wave_front"].iloc[-1]
        arrival_mask = df["wave_front"] >= end_position
        if arrival_mask.any():
            stats["wf_arrival_time"] = df.loc[arrival_mask, "time"].iloc[0]
            stats["wf_end_position"] = end_position
        else:
            stats["wf_arrival_time"] = None
            stats["wf_end_position"] = end_position
    else:
        stats["wf_arrival_time"] = None
        stats["wf_end_position"] = None

    # ── Average wave-front speed ──────────────────────────────────────
    # speed = total distance travelled / total time elapsed
    # Uses the full time range where wave_front data is available.
    if "wave_front" in df.columns and stats["wf_arrival_time"] is not None:
        t_start = df["time"].iloc[0]
        x_start = df["wave_front"].iloc[0]
        t_end = stats["wf_arrival_time"]
        x_end = stats["wf_end_position"]
        elapsed = t_end - t_start
        stats["wf_avg_speed"] = (x_end - x_start) / elapsed if elapsed > 0 else None
    else:
        stats["wf_avg_speed"] = None

    # ── Shock pressure (first pressure sensor, 0 ≤ t ≤ SHOCK_PRESSURE_T_MAX) ─
    pcols = pressure_columns(df)
    if pcols:
        first_sensor = pcols[0]
        second_sensor = pcols[1]
        t_limit = SHOCK_PRESSURE_T_MAX
        window = df[df["time"] <= t_limit]
        stats["shock_pressure"] = window[second_sensor].max()
        stats["shock_pressure_sensor"] = second_sensor
        stats["shock_pressure_t_max"] = t_limit
    else:
        stats["shock_pressure"] = None
        stats["shock_pressure_sensor"] = None
        stats["shock_pressure_t_max"] = None

    return stats


def print_stats_table(category, all_runs):
    """Pretty-print a stats table for every run in *all_runs*."""
    if not all_runs:
        return

    col_w = 12  # width of numeric columns

    header = (
        f"  {'Run':<35}  {'WF arrival (s)':>{col_w}}"
        f"  {'Avg speed':>{col_w}}  {'Shock P':>{col_w}}"
        f"  {'Sensor'}"
    )
    separator = "  " + "-" * (len(header) - 2)

    print(
        f"\n  [{category.upper()}] Statistics"
        f"  (shock P window: 0 – {SHOCK_PRESSURE_T_MAX} s)"
    )
    print(separator)
    print(header)
    print(separator)

    for run in all_runs:
        s = compute_stats(run)
        lbl = label_for(run)

        arrival = (
            f"{s['wf_arrival_time']:.4f}" if s["wf_arrival_time"] is not None else "n/a"
        )
        speed = (
            f"{s['wf_avg_speed']/np.sqrt(9.81 * 0.3):.4f}"
            if s["wf_avg_speed"] is not None
            else "n/a"
        )
        shock = (
            f"{s['shock_pressure']:.4f}" if s["shock_pressure"] is not None else "n/a"
        )
        sensor = s["shock_pressure_sensor"] or "n/a"

        print(
            f"  {lbl:<35}  {arrival:>{col_w}}"
            f"  {speed:>{col_w}}  {shock:>{col_w}}  {sensor}"
        )

    print(separator)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────


def plot_group(category, group_runs, base_runs, out_dir):
    """Generate all plots for one comparison category (nx / nt / flip)."""
    if not group_runs:
        print(f"[{category}] no runs found, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Use the *first* run in the group as the reference for column names.
    ref_df, _ = load_run(group_runs[0])
    pcols = pressure_columns(ref_df)

    # ── pressure plots ────────────────────────────────────────────────
    for ordinal, col in enumerate(pcols):
        # Make the ordinal available to plot_pressure via a module-level
        # variable (simple alternative to threading it through every call).
        global _col_ordinal
        _col_ordinal = ordinal

        out_path = os.path.join(out_dir, f"pressure_{col}.pdf")
        title = f"[{category}] {col}"
        plot_pressure(group_runs, base_runs, col, out_path, title)

    # ── wave-front plot ───────────────────────────────────────────────
    out_path = os.path.join(out_dir, "wave_front.pdf")
    title = f"[{category}] wave front"
    plot_wave_front(group_runs, base_runs, out_path, title)


def main():
    print(f"Scanning: {os.path.abspath(INPUT_DIR)}")
    runs = discover_runs(INPUT_DIR)

    for cat, found in runs.items():
        print(f"  {cat}: {len(found)} run(s)")

    base_runs = runs.get("base", [])

    for category in ("nx", "nt", "flip"):
        print(f"\n── {category.upper()} comparison ──")
        out_dir = os.path.join(OUTPUT_DIR, category)
        plot_group(category, runs[category], base_runs, out_dir)

    # ── Stats tables ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY STATISTICS")
    print("=" * 60)

    if base_runs:
        print_stats_table("base", base_runs)

    for category in ("nx", "nt", "flip"):
        all_runs = base_runs + runs[category] if INCLUDE_BASE_IN_ALL else runs[category]
        print_stats_table(category, all_runs)

    print("\nDone.")


if __name__ == "__main__":
    main()
