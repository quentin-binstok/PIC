"""
plot_sloshing_freq.py
=====================
Analyses sloshing simulation outputs from PIC/FLIP and APIC solvers.

Three kinds of output
---------------------
1. Frequency-vs-parameter comparison plots (PIC vs APIC), one PDF per
   signal type × parameter:
     plots_sloshing/<signal>/freq_vs_nx.pdf
     plots_sloshing/<signal>/freq_vs_nt.pdf
     plots_sloshing/<signal>/freq_vs_flip.pdf

2. Raw-signal + fitted-sine overlay plots, one PDF per selected run × signal:
     plots_sloshing/fit_checks/<run_name>_<signal>.pdf
   Configure which runs and signals you want in RAW_FIT_SELECTIONS (see below).

3. Power-spectrum (FFT) plots, one PDF per selected run × signal:
     plots_sloshing/fft/<run_name>_<signal>_fft.pdf
   Configure which runs and signals you want in FFT_SELECTIONS (see below).
   A vertical marker is drawn at the fitted frequency for easy reading.

Folder layout expected
----------------------
  <PIC_DIR>/sloshing_sim_sloshing_<param>_<value>/
            └── sloshing_sim_sloshing_<param>_<value>.{csv,json}
  <APIC_DIR>/sloshing_sim_apic_sloshing_<param>_<value>/
             └── sloshing_sim_apic_sloshing_<param>_<value>.{csv,json}

where <param> ∈ {nx, nt, flip, base}.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy.signal import periodogram
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  USER-CONFIGURABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Input / output directories
PIC_DIR = "../sloshing_pic_out"
APIC_DIR = "../sloshing_apic_out"
OUTPUT_DIR = "plots_sloshing"

# ── Fit model ─────────────────────────────────────────────────────────────────
# "simple" → A · sin(2π·f·t + φ) + C
# "damped" → (A + b·t) · sin(2π·f·t + φ) + C   (handles amplitude drift)
# "auto"   → try "damped" first; fall back to "simple" on failure
FIT_MODEL = "auto"

# Frequency search bounds [Hz] — adjust if your sloshing frequency is outside
F_BOUNDS = (0.05, 20.0)

# Skip this many rows (after step-0) before fitting, to avoid the initial
# transient where the wave hasn't established yet.
SKIP_ROWS = 0  # e.g. 200 to skip the first 200 data points

# ── Per-category run filter ───────────────────────────────────────────────────
# Set to None to include all runs, or provide a list of values to keep.
# For "flip" use float values, e.g. [0.0, 0.95, 1.0].
FILTER = {
    "nx": None,  # e.g. [50, 100, 200]
    "nt": None,  # e.g. [1000, 5000, 10000]
    "flip": None,  # e.g. [0.0, 0.95, 1.0]
}

# ── Raw data + fit overlay selection ─────────────────────────────────────────
# List of dicts; one output PDF is produced per entry × signal.
#
# Keys
# ----
# "solver"  : "pic" or "apic"
# "name"    : exact folder/run name (the subfolder name inside PIC_DIR or
#             APIC_DIR), e.g. "sloshing_sim_sloshing_nx_150"
#             or "sloshing_sim_apic_sloshing_base"
# "signals" : list of any combination of "pressure" and/or "depth"
#
# Example
# -------
# RAW_FIT_SELECTIONS = [
#     {
#         "solver":  "pic",
#         "name":    "sloshing_sim_sloshing_base",
#         "signals": ["pressure", "depth"],
#     },
#     {
#         "solver":  "pic",
#         "name":    "sloshing_sim_sloshing_nx_150",
#         "signals": ["pressure"],
#     },
#     {
#         "solver":  "apic",
#         "name":    "sloshing_sim_apic_sloshing_nx_150",
#         "signals": ["depth"],
#     },
# ]
RAW_FIT_SELECTIONS = [
    # ← add your entries here
    {
        "solver": "pic",
        "name": "sloshing_sim_sloshing_base",
        "signals": ["pressure", "depth"],
    },
]

# ── FFT (power-spectrum) plot selection ──────────────────────────────────────
# Same format as RAW_FIT_SELECTIONS.  A vertical marker is drawn at the
# frequency returned by the sinusoidal fit so you can read it off directly.
#
# Example
# -------
# FFT_SELECTIONS = [
#     {
#         "solver":  "pic",
#         "name":    "sloshing_sim_sloshing_base",
#         "signals": ["pressure", "depth"],
#     },
#     {
#         "solver":  "apic",
#         "name":    "sloshing_sim_apic_sloshing_nx_150",
#         "signals": ["pressure"],
#     },
# ]
FFT_SELECTIONS = [
    # ← add your entries here
    {
        "solver": "pic",
        "name": "sloshing_sim_sloshing_base",
        "signals": ["pressure", "depth"],
    },
]

# ── Figure aesthetics  (frequency-vs-param plots) ────────────────────────────
FIG_SIZE = (10, 7)  # (width, height) in inches
DPI = 150

FONTSIZE_LABEL = 28
FONTSIZE_TICK = 22
FONTSIZE_LEGEND = 28
FONTSIZE_TITLE = 0

COLOR_PIC = "#1f77b4"
COLOR_APIC = "#ff7f0e"
MARKER_PIC = "o"
MARKER_APIC = "s"
MARKERSIZE = 8
LINEWIDTH = 1
LINESTYLE = "--"  # set to "" to show markers only

GRID_MAJOR_ALPHA = 0.55
GRID_MINOR_ALPHA = 0.25

SHOW_TITLE = False  # whether to show a title on each plot

# ── Figure aesthetics  (raw data + fit overlay plots) ────────────────────────
RAW_FIG_SIZE = (12, 5)  # (width, height) in inches
RAW_DPI = 150

RAW_FONTSIZE_LABEL = 22
RAW_FONTSIZE_TICK = 18
RAW_FONTSIZE_LEGEND = 20
RAW_FONTSIZE_TITLE = 0

# Raw signal line
RAW_COLOR_DATA = "#444444"
RAW_LINEWIDTH_DATA = 0.8
RAW_ALPHA_DATA = 0.75
RAW_LABEL_DATA = "Simulation"

# Fitted sine line
RAW_COLOR_FIT = "#e63946"
RAW_LINEWIDTH_FIT = 2.0
RAW_LINESTYLE_FIT = "--"
RAW_LABEL_FIT = "Fitted sine"  # " (f = … Hz, <model>)" is appended

RAW_GRID_MAJOR_ALPHA = 0.50
RAW_GRID_MINOR_ALPHA = 0.20

# ── Figure aesthetics  (FFT / power-spectrum plots) ──────────────────────────
FFT_FIG_SIZE = (12, 5)  # (width, height) in inches
FFT_DPI = 150

FFT_FONTSIZE_LABEL = 22
FFT_FONTSIZE_TICK = 18
FFT_FONTSIZE_LEGEND = 20
FFT_FONTSIZE_TITLE = 0

# Whether to plot power on a log scale (useful when peaks are very narrow)
FFT_YLOG = False

# Frequency axis upper limit [Hz]; set to None to show the full Nyquist range
FFT_XLIM = 3.0  # e.g. 10.0

# Spectrum line
FFT_COLOR_SPECTRUM = "#2c6fad"
FFT_LINEWIDTH = 1.0
FFT_ALPHA = 0.85

# Vertical marker at the fitted frequency
FFT_COLOR_VLINE = "#e63946"
FFT_LINEWIDTH_VLINE = 1.8
FFT_LINESTYLE_VLINE = "--"
FFT_LABEL_VLINE = "Fitted frequency"  # " (f = … Hz)" is appended

FFT_GRID_MAJOR_ALPHA = 0.50
FFT_GRID_MINOR_ALPHA = 0.20

# ─────────────────────────────────────────────────────────────────────────────
#  FOLDER DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

PIC_RE = re.compile(
    r"^sloshing_sim_sloshing_"
    r"(?P<category>nx|nt|flip|base)"
    r"(?:_(?P<value>[\d.]+))?$"
)
APIC_RE = re.compile(
    r"^sloshing_sim_apic_sloshing_"
    r"(?P<category>nx|nt|flip|base)"
    r"(?:_(?P<value>[\d.]+))?$"
)


def discover_runs(root: str, pattern_re, solver_name: str) -> dict:
    """
    Walk *root* and return:
      { category: [ {name, csv, json, value, solver}, … ] }
    Entries are sorted by value ascending.
    """
    runs: dict = {"nx": [], "nt": [], "flip": [], "base": []}
    if not os.path.isdir(root):
        print(f"  [warn] directory not found: {os.path.abspath(root)}")
        return runs

    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        m = pattern_re.match(entry.name)
        if not m:
            continue

        cat = m.group("category")
        raw_val = m.group("value")
        value = float(raw_val) if raw_val is not None else 0.0

        csv_path = os.path.join(entry.path, entry.name + ".csv")
        json_path = os.path.join(entry.path, entry.name + ".json")
        if not os.path.isfile(csv_path) or not os.path.isfile(json_path):
            print(f"  [skip] {entry.name}: missing .csv or .json")
            continue

        runs[cat].append(
            {
                "name": entry.name,
                "csv": csv_path,
                "json": json_path,
                "value": value,
                "solver": solver_name,
            }
        )

    for cat, run_list in runs.items():
        run_list.sort(key=lambda r: r["value"])
        keep = FILTER.get(cat)
        if keep is not None:
            before = len(run_list)
            runs[cat] = [r for r in run_list if r["value"] in keep]
            print(
                f"  [filter] {solver_name}/{cat}: "
                f"{len(runs[cat])}/{before} run(s) kept"
            )
    return runs


def _all_runs_flat(runs: dict) -> list:
    """Flatten the category dict into a single list."""
    out = []
    for run_list in runs.values():
        out.extend(run_list)
    return out


def find_run_by_name(
    runs_pic: dict, runs_apic: dict, solver: str, name: str
) -> Optional[dict]:
    """
    Look up a run by solver tag and exact folder name.
    Returns None (with a warning) if not found.
    """
    pool = _all_runs_flat(runs_pic if solver == "pic" else runs_apic)
    for run in pool:
        if run["name"] == name:
            return run
    print(f"  [warn] run not found — solver={solver!r}  name={name!r}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def load_signals(run_info: dict):
    """
    Load the CSV for one run.

    Returns
    -------
    time     : 1-D ndarray   (seconds)
    pressure : 1-D ndarray   (Pa)  — first pressure column, or None
    depth    : 1-D ndarray   (m)   — first depth column,    or None
    """
    with open(run_info["json"]) as fh:
        params = json.load(fh)
    delta_t = params["delta_t"]

    df = pd.read_csv(run_info["csv"])

    # Drop the initial condition (step = 0) and any user-requested skips
    df = df[df["step"] > 0].reset_index(drop=True)
    if SKIP_ROWS > 0:
        df = df.iloc[SKIP_ROWS:].reset_index(drop=True)

    time = df["step"].values * delta_t
    pcols = [c for c in df.columns if c.startswith("pressure_")]
    dcols = [c for c in df.columns if c.startswith("depth_")]

    pressure = df[pcols[0]].values if pcols else None
    depth = df[dcols[0]].values if dcols else None

    return time, pressure, depth


# ─────────────────────────────────────────────────────────────────────────────
#  SINUSOIDAL FITTING
# ─────────────────────────────────────────────────────────────────────────────


def _sin_simple(t, A, f, phi, C):
    """A · sin(2π·f·t + φ) + C"""
    return A * np.sin(2 * np.pi * f * t + phi) + C


def _sin_damped(t, A, b, f, phi, C):
    """(A + b·t) · sin(2π·f·t + φ) + C"""
    return (A + b * t) * np.sin(2 * np.pi * f * t + phi) + C


def _fft_frequency_guess(time: np.ndarray, signal: np.ndarray) -> float:
    """Return a rough frequency estimate from the power spectrum."""
    dt = np.mean(np.diff(time))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, power = periodogram(signal - signal.mean(), fs=1.0 / dt)
    if len(freqs) > 1:
        idx = np.argmax(power[1:]) + 1
        f0 = freqs[idx]
        if F_BOUNDS[0] <= f0 <= F_BOUNDS[1]:
            return float(f0)
    return 0.5 * (F_BOUNDS[0] + F_BOUNDS[1])


def fit_signal(
    time: np.ndarray, signal: Optional[np.ndarray], run_name: str = ""
) -> dict:
    """
    Fit a sinusoidal model to *signal*.

    Returns a dict:
      "freq"         : fitted frequency [Hz], or None
      "model"        : "damped" | "simple" | "fft_fallback" | None
      "fitted_curve" : ndarray matching *time*, or None
      "params"       : raw popt (model-dependent), or None
    """
    result = {"freq": None, "model": None, "fitted_curve": None, "params": None}
    if signal is None or len(signal) < 20:
        return result

    dc = signal.mean()
    sig = signal - dc
    amp0 = np.std(sig) * np.sqrt(2)
    f0 = _fft_frequency_guess(time, sig)
    f_lo, f_hi = F_BOUNDS

    def try_damped():
        p0 = [amp0, 0.0, f0, 0.0, 0.0]
        lo = [-np.inf, -np.inf, f_lo, -2 * np.pi, -np.inf]
        hi = [np.inf, np.inf, f_hi, 2 * np.pi, np.inf]
        try:
            popt, _ = curve_fit(
                _sin_damped, time, sig, p0=p0, bounds=(lo, hi), maxfev=100_000
            )
            freq = abs(float(popt[2]))
            curve = _sin_damped(time, *popt) + dc
            return freq, popt, curve, "damped"
        except Exception as exc:
            print(f"    [damped fit fail] {run_name}: {exc}")
            return None

    def try_simple():
        p0 = [amp0, f0, 0.0, 0.0]
        lo = [-np.inf, f_lo, -2 * np.pi, -np.inf]
        hi = [np.inf, f_hi, 2 * np.pi, np.inf]
        try:
            popt, _ = curve_fit(
                _sin_simple, time, sig, p0=p0, bounds=(lo, hi), maxfev=100_000
            )
            freq = abs(float(popt[1]))
            curve = _sin_simple(time, *popt) + dc
            return freq, popt, curve, "simple"
        except Exception as exc:
            print(f"    [simple fit fail] {run_name}: {exc}")
            return None

    attempt = None
    if FIT_MODEL == "simple":
        attempt = try_simple()
    elif FIT_MODEL == "damped":
        attempt = try_damped()
    else:  # "auto"
        attempt = try_damped()
        if attempt is None:
            attempt = try_simple()

    if attempt is None:
        print(f"    [fallback to FFT] {run_name}")
        result["freq"] = f0
        result["model"] = "fft_fallback"
        return result

    freq, popt, curve, model = attempt
    result["freq"] = freq
    result["model"] = model
    result["params"] = popt
    result["fitted_curve"] = curve
    return result


def fit_frequency(time, signal, run_name="") -> Optional[float]:
    """Convenience wrapper — returns only the frequency."""
    return fit_signal(time, signal, run_name)["freq"]


# ─────────────────────────────────────────────────────────────────────────────
#  COLLECT FREQUENCIES  (for summary plots)
# ─────────────────────────────────────────────────────────────────────────────


def collect_frequencies(
    runs_pic: dict, runs_apic: dict, category: str, signal_type: str
) -> dict:
    """Return {"pic": [(value, freq), …], "apic": [(value, freq), …]}."""
    results = {"pic": [], "apic": []}
    pairs = [
        ("pic", runs_pic.get(category, [])),
        ("apic", runs_apic.get(category, [])),
    ]
    for solver_key, run_list in pairs:
        for run in run_list:
            time, pressure, depth = load_signals(run)
            sig = pressure if signal_type == "pressure" else depth
            freq = fit_frequency(time, sig, run["name"])
            if freq is not None:
                results[solver_key].append((run["value"], freq))
                print(
                    f"    {solver_key.upper():4s}  {run['name']:50s}  "
                    f"f = {freq:.4f} Hz"
                )
            else:
                print(f"    [skip] {run['name']}: fit returned None")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED AXIS STYLING
# ─────────────────────────────────────────────────────────────────────────────


def _style_axes(
    ax,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    fontsize_label: int = FONTSIZE_LABEL,
    fontsize_tick: int = FONTSIZE_TICK,
    fontsize_title: int = FONTSIZE_TITLE,
    grid_major_alpha: float = GRID_MAJOR_ALPHA,
    grid_minor_alpha: float = GRID_MINOR_ALPHA,
):
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_tick)
    ax.tick_params(axis="both", which="minor", labelsize=fontsize_tick)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.6, alpha=grid_major_alpha)
    ax.grid(True, which="minor", linewidth=0.2, alpha=grid_minor_alpha)
    if title and SHOW_TITLE:
        ax.set_title(title, fontsize=fontsize_title)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1 — frequency vs. parameter (summary)
# ─────────────────────────────────────────────────────────────────────────────


def plot_freq_vs_param(
    runs_pic: dict,
    runs_apic: dict,
    category: str,
    signal_type: str,
    xlabel: str,
    out_path: str,
    title: str,
):
    """Produce one frequency-vs-parameter comparison plot (PIC vs APIC)."""
    print(f"\n  ── {signal_type.upper()} / {category.upper()} ──")
    data = collect_frequencies(runs_pic, runs_apic, category, signal_type)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    plot_specs = [
        ("pic", COLOR_PIC, MARKER_PIC, "PIC/FLIP"),
        ("apic", COLOR_APIC, MARKER_APIC, "APIC"),
    ]
    any_plotted = False
    for solver_key, color, marker, label in plot_specs:
        pts = data[solver_key]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts, key=lambda p: p[0]))
        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            label=label,
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            linestyle=LINESTYLE,
        )
        any_plotted = True

    _style_axes(ax, xlabel, ylabel=r"Natural frequency $f\;[\mathrm{Hz}]$", title=title)
    if any_plotted:
        ax.legend(fontsize=FONTSIZE_LEGEND, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2 — raw signal + fitted sine overlay
# ─────────────────────────────────────────────────────────────────────────────

_YLABELS = {
    "pressure": r"Pressure $[\mathrm{Pa}]$",
    "depth": r"Water depth $[\mathrm{m}]$",
}


def plot_raw_fit(run_info: dict, signal_type: str, out_path: str):
    """
    Plot the raw signal alongside the fitted sine for one run + signal.
    Saves to *out_path*.
    """
    time, pressure, depth = load_signals(run_info)
    sig = pressure if signal_type == "pressure" else depth

    if sig is None:
        print(f"  [skip] {run_info['name']}: no {signal_type} column found")
        return

    fit = fit_signal(time, sig, run_info["name"])
    freq = fit["freq"]

    fig, ax = plt.subplots(figsize=RAW_FIG_SIZE)

    # ── raw data ──────────────────────────────────────────────────────────────
    ax.plot(
        time,
        sig,
        color=RAW_COLOR_DATA,
        linewidth=RAW_LINEWIDTH_DATA,
        alpha=RAW_ALPHA_DATA,
        label=RAW_LABEL_DATA,
    )

    # ── fitted curve (or FFT annotation) ─────────────────────────────────────
    if fit["fitted_curve"] is not None and freq is not None:
        fit_label = f"{RAW_LABEL_FIT} " f"(f = {freq:.4f} Hz, {fit['model']})"
        ax.plot(
            time,
            fit["fitted_curve"],
            color=RAW_COLOR_FIT,
            linewidth=RAW_LINEWIDTH_FIT,
            linestyle=RAW_LINESTYLE_FIT,
            label=fit_label,
        )
    elif freq is not None:
        # FFT fallback — no curve available, just show the estimate
        ax.axhline(
            np.nan,
            color=RAW_COLOR_FIT,
            linestyle=RAW_LINESTYLE_FIT,
            label=f"FFT estimate: f = {freq:.4f} Hz",
        )

    solver_tag = run_info["solver"].upper()
    _style_axes(
        ax,
        xlabel=r"Time $[\mathrm{s}]$",
        ylabel=_YLABELS.get(signal_type, signal_type),
        title=f"{run_info['name']}  [{solver_tag}]  —  {signal_type}",
        fontsize_label=RAW_FONTSIZE_LABEL,
        fontsize_tick=RAW_FONTSIZE_TICK,
        fontsize_title=RAW_FONTSIZE_TITLE,
        grid_major_alpha=RAW_GRID_MAJOR_ALPHA,
        grid_minor_alpha=RAW_GRID_MINOR_ALPHA,
    )

    ax.legend(fontsize=RAW_FONTSIZE_LEGEND, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=RAW_DPI)
    plt.close(fig)
    print(f"  saved → {out_path}")


def run_raw_fit_plots(runs_pic: dict, runs_apic: dict, out_dir: str):
    """Iterate over RAW_FIT_SELECTIONS and produce one PDF per entry × signal."""
    if not RAW_FIT_SELECTIONS:
        print("  (RAW_FIT_SELECTIONS is empty — skipping fit-check plots)")
        return

    os.makedirs(out_dir, exist_ok=True)

    for sel in RAW_FIT_SELECTIONS:
        solver = sel.get("solver", "pic")
        name = sel.get("name", "")
        signals = sel.get("signals", ["pressure", "depth"])

        run = find_run_by_name(runs_pic, runs_apic, solver, name)
        if run is None:
            continue

        for sig_type in signals:
            safe_name = name.replace("/", "_")
            filename = f"{safe_name}_{sig_type}.pdf"
            out_path = os.path.join(out_dir, filename)
            print(f"\n  ── fit-check: {name}  [{solver.upper()}]  {sig_type} ──")
            plot_raw_fit(run, sig_type, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 3 — FFT / power spectrum
# ─────────────────────────────────────────────────────────────────────────────


def compute_fft(time: np.ndarray, signal: np.ndarray):
    """
    Compute a one-sided power spectral density via Welch's periodogram.

    Returns
    -------
    freqs : 1-D ndarray   frequency bins [Hz]
    power : 1-D ndarray   power at each bin (same units as signal²/Hz)
    """
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, power = periodogram(signal - signal.mean(), fs=fs)
    # Drop the DC bin (index 0)
    return freqs[1:], power[1:]


def plot_fft(run_info: dict, signal_type: str, out_path: str):
    """
    Plot the power spectrum for one run + signal, with a vertical marker at
    the fitted frequency.  Saves to *out_path*.
    """
    time, pressure, depth = load_signals(run_info)
    sig = pressure if signal_type == "pressure" else depth

    if sig is None:
        print(f"  [skip] {run_info['name']}: no {signal_type} column found")
        return

    # ── spectrum ──────────────────────────────────────────────────────────────
    freqs, power = compute_fft(time, sig)

    # ── fitted frequency (for the marker) ────────────────────────────────────
    fit = fit_signal(time, sig, run_info["name"])
    freq = fit["freq"]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FFT_FIG_SIZE)

    ax.plot(
        freqs,
        power,
        color=FFT_COLOR_SPECTRUM,
        linewidth=FFT_LINEWIDTH,
        alpha=FFT_ALPHA,
        label="Power spectrum",
    )

    if freq is not None:
        vline_label = f"{FFT_LABEL_VLINE} (f = {freq:.4f} Hz)"
        ax.axvline(
            freq,
            color=FFT_COLOR_VLINE,
            linewidth=FFT_LINEWIDTH_VLINE,
            linestyle=FFT_LINESTYLE_VLINE,
            label=vline_label,
        )

    if FFT_YLOG:
        ax.set_yscale("log")

    if FFT_XLIM is not None:
        ax.set_xlim(left=0, right=FFT_XLIM)
    else:
        ax.set_xlim(left=0)

    # Y-axis label depends on signal units
    power_ylabel = {
        "pressure": r"PSD $[\mathrm{Pa}^2/\mathrm{Hz}]$",
        "depth": r"PSD $[\mathrm{m}^2/\mathrm{Hz}]$",
    }.get(signal_type, r"PSD")

    solver_tag = run_info["solver"].upper()
    _style_axes(
        ax,
        xlabel=r"Frequency $[\mathrm{Hz}]$",
        ylabel=power_ylabel,
        title=f"{run_info['name']}  [{solver_tag}]  —  {signal_type}  (FFT)",
        fontsize_label=FFT_FONTSIZE_LABEL,
        fontsize_tick=FFT_FONTSIZE_TICK,
        fontsize_title=FFT_FONTSIZE_TITLE,
        grid_major_alpha=FFT_GRID_MAJOR_ALPHA,
        grid_minor_alpha=FFT_GRID_MINOR_ALPHA,
    )

    ax.legend(fontsize=FFT_FONTSIZE_LEGEND, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=FFT_DPI)
    plt.close(fig)
    print(f"  saved → {out_path}")


def run_fft_plots(runs_pic: dict, runs_apic: dict, out_dir: str):
    """Iterate over FFT_SELECTIONS and produce one PDF per entry × signal."""
    if not FFT_SELECTIONS:
        print("  (FFT_SELECTIONS is empty — skipping FFT plots)")
        return

    os.makedirs(out_dir, exist_ok=True)

    for sel in FFT_SELECTIONS:
        solver = sel.get("solver", "pic")
        name = sel.get("name", "")
        signals = sel.get("signals", ["pressure", "depth"])

        run = find_run_by_name(runs_pic, runs_apic, solver, name)
        if run is None:
            continue

        for sig_type in signals:
            safe_name = name.replace("/", "_")
            filename = f"{safe_name}_{sig_type}_fft.pdf"
            out_path = os.path.join(out_dir, filename)
            print(f"\n  ── FFT: {name}  [{solver.upper()}]  {sig_type} ──")
            plot_fft(run, sig_type, out_path)


# ─────────────────────────────────────────────────────────────────────────────

# (category_key, x-axis label for frequency-vs-param plots)
CATEGORIES = [
    ("nx", r"Number of cells along $x$"),
    ("nt", r"Number of time steps"),
    ("flip", r"FLIP blending ratio"),
]

# (signal_type_key, human description)
SIGNAL_TYPES = [
    ("pressure", "Pressure (first sensor)"),
    ("depth", "Water depth"),
]


def main():
    print(f"Scanning PIC  → {os.path.abspath(PIC_DIR)}")
    print(f"Scanning APIC → {os.path.abspath(APIC_DIR)}")

    runs_pic = discover_runs(PIC_DIR, PIC_RE, "pic")
    runs_apic = discover_runs(APIC_DIR, APIC_RE, "apic")

    print("\nRuns discovered:")
    for cat in ("nx", "nt", "flip", "base"):
        n_pic = len(runs_pic.get(cat, []))
        n_apic = len(runs_apic.get(cat, []))
        print(f"  {cat:6s}: {n_pic:3d} PIC   {n_apic:3d} APIC")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Frequency-vs-parameter summary plots ───────────────────────────────
    for sig_key, sig_desc in SIGNAL_TYPES:
        sig_dir = os.path.join(OUTPUT_DIR, sig_key)
        os.makedirs(sig_dir, exist_ok=True)
        print(f"\n{'═'*60}")
        print(f"  Signal: {sig_desc}")
        print(f"{'═'*60}")
        for cat, xlabel in CATEGORIES:
            out_path = os.path.join(sig_dir, f"freq_vs_{cat}.pdf")
            title = f"Natural frequency vs. {cat}  [{sig_desc}]"
            plot_freq_vs_param(
                runs_pic, runs_apic, cat, sig_key, xlabel, out_path, title
            )

    # ── 2. Raw data + fit overlay plots ──────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Raw signal + fit-check plots")
    print(f"{'═'*60}")
    fit_check_dir = os.path.join(OUTPUT_DIR, "fit_checks")
    run_raw_fit_plots(runs_pic, runs_apic, fit_check_dir)

    # ── 3. FFT / power-spectrum plots ────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  FFT / power-spectrum plots")
    print(f"{'═'*60}")
    fft_dir = os.path.join(OUTPUT_DIR, "fft")
    run_fft_plots(runs_pic, runs_apic, fft_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
