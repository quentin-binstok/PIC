"""
Cylindrical in an uniform flow — validation of the method.
 
Subcommands:
    run         — build JSON configs and launch all simulations
    plot        — plot velocity at one probe point over time
    slice       — plot velocity profile along y at multiple timesteps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from pandas import read_csv
import json
import argparse
import pathlib
import copy
import subprocess
import os
 
 

NY_LIST = [100, 150, 200]
NT_LIST = [250, 500, 1000, 2000, 4000]
ROOT_SIM = pathlib.Path("von_karman_sim/SL")
ROOT_OUT = pathlib.Path("Von_Karman/SL")
SOLVER = "SL"

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
 
TICK_SIZE   = 22   # axis tick label size
LABEL_SIZE  = 28   # x/y axis label size
LEGEND_SIZE = 18   # legend text size
TITLE_SIZE  = 20   # figure title size
LINE_WIDTH  = 2.8  # default line width for simulation curves
FIT_WIDTH   = 3.2  # line width for fit / reference curves
 
def apply_style(ax):
    """Apply consistent tick and grid style to an Axes."""
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, width=1.4, length=6)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
 
 
# ---------------------------------------------------------------------------
# Qualitative color palette (standard, distinguishable colors)
# ---------------------------------------------------------------------------
 
# A hand-picked palette that reads well on white and in print:
# blue, orange, green, red, purple  (one per resolution)
_PALETTE = [
    "#1f77b4",   # muted blue
    "#ff7f0e",   # safety orange
    "#2ca02c",   # cooked asparagus green
    "#d62728",   # brick red
    "#9467bd",   # muted purple
]
 
def _resolution_styles(nx_list):
    """
    Return a dict {nx: {"color": ..., "ls": ..., "marker": ...}}
    Each resolution gets a distinct color from the qualitative palette.
    """
    nx_sorted = sorted(nx_list)
    styles = {}
    for i, nx in enumerate(nx_sorted):
        styles[nx] = {
            "color":  _PALETTE[i % len(_PALETTE)],
            "alpha":  1.0,
        }
    return styles
 
 
 
# ---------------------------------------------------------------------------
# Shared helper: load one run, return everything needed for plotting
# ---------------------------------------------------------------------------
 
def load_run(json_path: pathlib.Path, csv_path: pathlib.Path,
             probe_ix: int = None, probe_iy: int = None) -> dict:
 
    with open(json_path) as f:
        data = json.load(f)
 
    dt = data["delta_t"]
    nt = data["nt"]
    dx = data["space_steps"]
    nx = data["grid"][0]
    ny = data["grid"][1]
 
    bc_inflow = next(bc for bc in data["bc"] if "speed_x" in bc)
    U0 = bc_inflow["speed_x"]
    V0 = bc_inflow.get("speed_y", 0.0)
 
    if probe_ix is None:
        probe_ix = data["metrics"][0]["idx"][0]
    if probe_iy is None:
        probe_iy = data["metrics"][0]["idx"][1]
 
    x_probe = probe_ix * dx
    y_probe = probe_iy * dx
 
    # skip t=0
    times = np.linspace(0.0, nt * dt, nt, dtype=float)[1:]
 
    df    = read_csv(csv_path)
    col_u = f"vx_{probe_ix}_{probe_iy}"
    col_v = f"vy_{probe_ix}_{probe_iy}"
 
    missing = [c for c in (col_u, col_v) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns {missing} not found in {csv_path}.\n"
            f"Available: {list(df.columns)}"
        )
 
    sim_u = df[col_u].values[1:nt]
    sim_v = df[col_v].values[1:nt]
 
    return dict(
        nx=nx, ny=ny, dx=dx, dt=dt, nt=nt,
        probe_ix=probe_ix, probe_iy=probe_iy,
        x_probe=x_probe, y_probe=y_probe,
        times=times,
        sim_u=sim_u, sim_v=sim_v,
        ana_u0=U0, ana_v0=V0,
    )
 
 
# ---------------------------------------------------------------------------
# Simulation folder builder
# ---------------------------------------------------------------------------
 
def build_folder(args):
    temp = ROOT_SIM
    temp.mkdir(exist_ok=True)
    ROOT_OUT.mkdir(exist_ok=True)
 
    with open(args.input) as f:
        base = json.load(f)
    
    base_T = 7.5
    base_L = 1.0
    CFL = 0.2
 
    def slice_timesteps(nt):
        return [1, nt // 3, 2 * nt // 3, nt - 2]

 
    for ny in NY_LIST:
        nx    = 1.5*ny  # keep aspect ratio 4:1
        dx    = base_L / ny
        dt    = CFL * dx  # CFL condition for stability
        nt    = int(base_T / dt)
 
        work = copy.deepcopy(base)
        work["solver"]      = "pic"
        work["grid"]        = [nx, ny]
        work["space_steps"] = dx
        work["delta_t"]     = dt
        work["nt"]          = nt
        work["dir"]         = str(ROOT_OUT / f"ny_{ny}")
        work["sampling_rate"] = nt//50  # ~100 samples per run
        work["flip"]    = 0.95

        work["ic_cylinders"][0]["center"] = [(int)(nx/8), (int)(ny/2)]
        work["ic_cylinders"][0]["radius"] = (int)(ny/20)


        work["metrics"][0]["idx"] = [3*nx // 4, ny // 2]
        work["metrics"][1]["idx"] = [3*nx // 4, ny // 2]
        work["metrics"][2]["idx"] = [3*nx // 4, ny // 2]
        work["metrics"][3]["A_ref"] = 2 * (ny/20) * dx


 
        """ work["slice_x_csv"] = {
            "enabled": True,
            "field": "vx",
            "type": "vertical",
            "i": 5 * nx // 6,
            "j_start": 1,
            "j_end": ny - 1,
            "timesteps": slice_timesteps(nt),
            "file": "vx_slice.csv",
        }

        work["slice_y_csv"] = {
            "enabled": True,
            "field": "vy",
            "type": "vertical",
            "i": 5 * nx // 6,
            "j_start": 1,
            "j_end": ny - 1,
            "timesteps": slice_timesteps(nt),
            "file": "vy_slice.csv",
        } """
 
        with open(temp / f"von_karman_ny_{ny}.json", "w") as f:
            json.dump(work, f, indent=2)
 
    return temp
 
 
def launch_sims(args, folder):
    for json_file in sorted(folder.glob("von_karman_ny_*.json")):
        sbinary = os.path.abspath(args.binary)
        subprocess.run([sbinary, str(json_file)], check=True)

def dominant_frequency(signal, dt):
    """
    Return dominant frequency (Hz) of a 1D signal using FFT.
    """
    signal = np.asarray(signal)
    signal = signal - np.mean(signal)

    n = len(signal)
    freqs = np.fft.rfftfreq(n, dt)
    fft_vals = np.abs(np.fft.rfft(signal))

    # ignore zero frequency
    fft_vals[0] = 0.0

    idx = np.argmax(fft_vals)
    return freqs[idx]
 
# ---------------------------------------------------------------------------
# plot subcommand — velocity at probe point over time
# ---------------------------------------------------------------------------
 
def _load_vk_run(ny: int, col: str):
    """
    Load a single column from the von Kármán metrics CSV for resolution ny.
    Returns (times, signal, dt) or None if files are missing.
    """
    json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
    csv_path  = ROOT_OUT / f"ny_{ny}" / f"metrics_SL_{ny}.csv"
 
    if not json_path.exists() or not csv_path.exists():
        print(f"ny={ny}: missing files, skipping.")
        return None
 
    with open(json_path) as f:
        data = json.load(f)
 
    dt = data["delta_t"]
    nt = data["nt"]
 
    df = read_csv(csv_path)
    if col not in df.columns:
        print(f"ny={ny}: column '{col}' not found, skipping.")
        return None
 
    signal = df[col].values[:nt].astype(float)
    times  = np.linspace(0.0, nt * dt, nt, dtype=float)
 
    return times, signal, dt, nt
 
 
def _transient_cut(nt: int, fraction: float = 1/2) -> int:
    """Index at which the transient is considered over."""
    return int(fraction * nt)
 
 
# ---------------------------------------------------------------------------
# cmd_plot_vx
# ---------------------------------------------------------------------------
 
def cmd_plot_vx(args):
    """
    Von Kármán — plot vx at the probe point over time for all resolutions,
    and extract the dominant frequency of the oscillation.
    """
    styles = _resolution_styles(NY_LIST)
 
    fig, ax = plt.subplots(figsize=(9, 6))
 
    ny_vals  = []
    freq_vals = []
 
    for ny in NY_LIST:
        json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
        csv_path  = ROOT_OUT / f"ny_{ny}" / f"metrics_APIC_VK.csv"
 
        if not json_path.exists() or not csv_path.exists():
            print(f"ny={ny}: missing files, skipping.")
            continue
 
        with open(json_path) as f:
            data = json.load(f)
 
        dt = data["delta_t"]
        nt = data["nt"]
        ix, iy = data["metrics"][0]["idx"]
 
        col = f"vx_{ix}_{iy}"
        df  = read_csv(csv_path)
        if col not in df.columns:
            print(f"ny={ny}: column '{col}' not found, skipping.")
            continue
 
        signal = df[col].values[:nt].astype(float)
        cut    = _transient_cut(nt)
        times  = np.linspace(cut * dt, nt * dt, nt - cut, dtype=float)
        signal = signal[cut:]
 
        freq = dominant_frequency(signal, dt)
        ny_vals.append(ny)
        freq_vals.append(freq)
        print(f"ny={ny:4d} | f(vx) = {freq:.4f} Hz")
 
        color = styles[ny]["color"]
        ax.plot(times, signal, lw=LINE_WIDTH, color=color,
                alpha=styles[ny].get("alpha", 1.0), label=f"ny={ny}")
 
    ax.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$v_x$ (m/s)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig.tight_layout()
 
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_OUT / f"vk_vx_{SOLVER}.pdf", format="pdf")
    print(f"Saved: {ROOT_OUT / f'vk_vx_{SOLVER}.pdf'}")
 
    _plot_frequency_convergence(ny_vals, freq_vals, ylabel=r"$f_{v_x}$ (Hz)",
                                 out_name=f"vk_freq_vx_{SOLVER}.pdf")
    plt.show()
 
 
# ---------------------------------------------------------------------------
# cmd_plot_vy
# ---------------------------------------------------------------------------
 
def cmd_plot_vy(args):
    """
    Von Kármán — plot vy at the probe point over time for all resolutions,
    and extract the dominant frequency of the oscillation.
    """
    styles = _resolution_styles(NY_LIST)
 
    fig, ax = plt.subplots(figsize=(9, 6))
 
    ny_vals   = []
    freq_vals = []
 
    for ny in NY_LIST:
        json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
        csv_path  = ROOT_OUT / f"ny_{ny}" / f"metrics_APIC_VK.csv"
 
        if not json_path.exists() or not csv_path.exists():
            print(f"ny={ny}: missing files, skipping.")
            continue
 
        with open(json_path) as f:
            data = json.load(f)
 
        dt = data["delta_t"]
        nt = data["nt"]
        ix, iy = data["metrics"][0]["idx"]
 
        col = f"vy_{ix}_{iy}"
        df  = read_csv(csv_path)
        if col not in df.columns:
            print(f"ny={ny}: column '{col}' not found, skipping.")
            continue
 
        signal = df[col].values[:nt].astype(float)
        cut    = _transient_cut(nt)
        times  = np.linspace(cut * dt, nt * dt, nt - cut, dtype=float)
        signal = signal[cut:]
 
        freq = dominant_frequency(signal, dt)
        ny_vals.append(ny)
        freq_vals.append(freq)
        print(f"ny={ny:4d} | f(vy) = {freq:.4f} Hz")
 
        color = styles[ny]["color"]
        ax.plot(times, signal, lw=LINE_WIDTH, color=color,
                alpha=styles[ny].get("alpha", 1.0), label=f"ny={ny}")
 
    ax.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$v_y$ (m/s)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig.tight_layout()
 
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_OUT / f"vk_vy_{SOLVER}.pdf", format="pdf")
    print(f"Saved: {ROOT_OUT / f'vk_vy_{SOLVER}.pdf'}")
 
    _plot_frequency_convergence(ny_vals, freq_vals, ylabel=r"$f_{v_y}$ (Hz)",
                                 out_name=f"vk_freq_vy_{SOLVER}.pdf")
    plt.show()
 
 
# ---------------------------------------------------------------------------
# cmd_plot_p
# ---------------------------------------------------------------------------
 
def cmd_plot_p(args):
    """
    Von Kármán — plot pressure p at the probe point over time for all
    resolutions, and extract the dominant frequency of the oscillation.
    """
    styles = _resolution_styles(NY_LIST)
 
    fig, ax = plt.subplots(figsize=(9, 6))
 
    ny_vals   = []
    freq_vals = []
 
    for ny in NY_LIST:
        json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
        csv_path  = ROOT_OUT / f"ny_{ny}" / f"metrics_APIC_VK.csv"
 
        if not json_path.exists() or not csv_path.exists():
            print(f"ny={ny}: missing files, skipping.")
            continue
 
        with open(json_path) as f:
            data = json.load(f)
 
        dt = data["delta_t"]
        nt = data["nt"]
        ix, iy = data["metrics"][0]["idx"]
 
        col = f"pressure_{ix}_{iy}"
        df  = read_csv(csv_path)
        if col not in df.columns:
            print(f"ny={ny}: column '{col}' not found, skipping.")
            continue
 
        signal = df[col].values[:nt].astype(float)
        cut    = _transient_cut(nt)
        times  = np.linspace(cut * dt, nt * dt, nt - cut, dtype=float)
        signal = signal[cut:]
 
        freq = dominant_frequency(signal, dt)
        ny_vals.append(ny)
        freq_vals.append(freq)
        print(f"ny={ny:4d} | f(p)  = {freq:.4f} Hz")
 
        color = styles[ny]["color"]
        ax.plot(times, signal, lw=LINE_WIDTH, color=color,
                alpha=styles[ny].get("alpha", 1.0), label=f"ny={ny}")
 
    ax.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$p$ (Pa)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig.tight_layout()
 
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_OUT / f"vk_p_{SOLVER}.pdf", format="pdf")
    print(f"Saved: {ROOT_OUT / f'vk_p_{SOLVER}.pdf'}")
 
    _plot_frequency_convergence(ny_vals, freq_vals, ylabel=r"$f_p$ (Hz)",
                                 out_name=f"vk_freq_p_{SOLVER}.pdf")
    plt.show()
 
 
# ---------------------------------------------------------------------------
# cmd_plot_cl
# ---------------------------------------------------------------------------
 
def cmd_plot_cl(args):
    """
    Von Kármán — plot the lift coefficient Cl over time for all resolutions,
    and extract the dominant (shedding) frequency.
    """
    styles = _resolution_styles(NY_LIST)
 
    fig_1, ax = plt.subplots(figsize=(9, 6))
    fig_2, ax2 = plt.subplots(figsize=(9, 6))
 
    ny_vals   = []
    freq_vals = []
 
    for ny in NY_LIST:
        json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
        csv_path  = ROOT_OUT / f"ny_{ny}" / f"metrics_APIC_VK.csv"
 
        if not json_path.exists() or not csv_path.exists():
            print(f"ny={ny}: missing files, skipping.")
            continue
 
        with open(json_path) as f:
            data = json.load(f)
 
        dt = data["delta_t"]
        nt = data["nt"]
 
        df = read_csv(csv_path)
        if "Cl" not in df.columns:
            print(f"ny={ny}: column 'Cl' not found, skipping.")
            continue

        Cl = df["Cl"]
        T = np.linspace(0.0, nt * dt, nt, dtype=float)
        ax2.plot(T, Cl, lw=LINE_WIDTH, color=styles[ny]["color"],
                 alpha=styles[ny].get("alpha", 1.0), label=f"ny={ny}")

        signal = df["Cl"].values[:nt].astype(float)
        cut    = _transient_cut(nt)
        times  = np.linspace(cut * dt, nt * dt, nt - cut, dtype=float)
        signal = signal[cut:]
 
        freq = dominant_frequency(signal, dt)
        ny_vals.append(ny)
        freq_vals.append(freq)
        print(f"ny={ny:4d} | f(Cl) = {freq:.4f} Hz")
 
        color = styles[ny]["color"]
        ax.plot(times, signal, lw=LINE_WIDTH, color=color,
                alpha=styles[ny].get("alpha", 1.0), label=f"ny={ny}")
 
    ax.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$C_l$", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig_1.tight_layout()

    ax2.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax2.set_ylabel(r"$C_l$", fontsize=LABEL_SIZE)
    ax2.legend(fontsize=LEGEND_SIZE)
    apply_style(ax2)
    fig_2.tight_layout()
 
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    fig_1.savefig(ROOT_OUT / f"vk_cl_time_{SOLVER}.pdf", format="pdf")
    print(f"Saved: {ROOT_OUT / f'vk_cl_time_{SOLVER}.pdf'}")
    fig_2.savefig(ROOT_OUT / f"vk_cl_{SOLVER}.pdf", format="pdf")
    print(f"Saved: {ROOT_OUT / f'vk_cl_{SOLVER}.pdf'}")
 
    _plot_frequency_convergence(ny_vals, freq_vals, ylabel=r"$f_{C_l}$ (Hz)",
                                 out_name=f"vk_freq_cl_{SOLVER}.pdf")
    plt.show()
 
# ---------------------------------------------------------------------------
# shared helper — frequency convergence vs ny
# ---------------------------------------------------------------------------
 
def _plot_frequency_convergence(ny_vals, freq_vals, ylabel: str, out_name: str):
    """
    Plot dominant frequency vs resolution (ny) and save to ROOT_OUT.
    Called internally by every cmd_plot_* function.
    """
    if len(ny_vals) < 2:
        return
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    ax.plot(ny_vals, freq_vals, "o-", lw=FIT_WIDTH, ms=7,
            color=_PALETTE[0], label=ylabel)
 
    ax.set_xlabel(r"Resolution $n_y$", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig.tight_layout()
 
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_OUT / out_name, format="pdf")
    print(f"Saved: {ROOT_OUT / out_name}")
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
 
    parser = argparse.ArgumentParser(description="Taylor–Green vortex validation")
    sub = parser.add_subparsers(dest="mode", required=True)

    # -------------------------
    # run
    # -------------------------
    run_p = sub.add_parser("run")
    run_p.add_argument("binary", type=pathlib.Path)
    run_p.add_argument("-i", "--input", required=True, type=pathlib.Path)

    # -------------------------
    # plotting commands
    # -------------------------
    sub.add_parser("plot_vx", help="Von Kármán vx time history + frequency")
    sub.add_parser("plot_vy", help="Von Kármán vy time history + frequency")
    sub.add_parser("plot_p",  help="Von Kármán pressure time history + frequency")

    # ✅ plot_cl with debug flag
    plot_cl_p = sub.add_parser("plot_cl", help="Von Kármán Cl time history + shedding frequency")
    plot_cl_p.add_argument(
        "--debug_fft",
        action="store_true",
        help="Show FFT spectrum for each run"
    )

    args = parser.parse_args()
 
    if args.mode == "run":
        folder = build_folder(args)
        launch_sims(args, folder)
    elif args.mode == "plot_vx":  cmd_plot_vx(args)
    elif args.mode == "plot_vy":  cmd_plot_vy(args)
    elif args.mode == "plot_p":   cmd_plot_p(args)
    elif args.mode == "plot_cl":  cmd_plot_cl(args)
    
 
 
if __name__ == "__main__":
    main()
 