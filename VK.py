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
import matplotlib.colors as mcolors
from pandas import read_csv
import json
import argparse
import pathlib
import copy
import subprocess
 
ROOT_SIM = pathlib.Path("von_karman_sim/SL")
ROOT_OUT = pathlib.Path("Von_Karman/SL")
NY_LIST = [30, 40, 50, 80, 100, 150]
 
 
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
    temp_folder = pathlib.Path("von_karman_sim/SL")
    temp_folder.mkdir(parents=True, exist_ok=True)
    pathlib.Path("Von_Karman/SL").mkdir(parents=True, exist_ok=True)
 
    with open(args.input, "r") as f:
        base_data = json.load(f)
 
    with open(temp_folder / "von_karman_base.json", "w") as f:
        json.dump(base_data, f, indent=2)
 
    base_dx  = base_data["space_steps"]
    base_ny  = base_data["grid"][1]
    base_nt  = base_data["nt"]
    base_dt  = base_data["delta_t"]
    base_cfl = 4*base_dt / base_dx
    base_T   = base_nt * base_dt
 
    # timesteps at which the slice is saved: spread over the simulation
    # use 4 snapshots: 0%, 33%, 66%, 100% of total steps
    def slice_timesteps(nt):
        return [1, nt // 3, 2 * nt // 3, nt - 2]
 
    for ny in NY_LIST:
        ratio = ny / base_ny
        dx    = base_dx / ratio
        dt    = base_cfl * dx
        nt    = int(round(base_T / dt))
        nx   = 2 * ny
 
        run_dir = pathlib.Path("Von_Karman/SL") / f"ny_{ny}"
 
        work = copy.deepcopy(base_data)
        work["grid"]        = [nx, ny]
        work["space_steps"] = dx
        work["delta_t"]     = dt
        work["nt"]          = nt
        work["dir"]         = str(run_dir)

        work["ic_cylinders"][0]["center"] = [nx/5, ny/2]
        work["ic_cylinders"][0]["radius"] = ny/20


        work["metrics"][0]["idx"] = [3*nx // 4, ny // 2]
        work["metrics"][1]["idx"] = [3*nx // 4, ny // 2]
        work["metrics"][2]["idx"] = [3*nx // 4, ny // 2]
        work["metrics"][3]["A_ref"] = 2 * (ny/20) * dx


 
        work["slice_x_csv"] = {
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
        }
 
        with open(temp_folder / f"von_karman_ny_{ny}.json", "w") as f:
            json.dump(work, f, indent=2)
 
    return temp_folder
 
 
def launch_sims(args, folder):
    json_files = sorted(    
        folder.glob("von_karman_ny_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for json_file in json_files:
        print(f"\nRunning {json_file.name} ...")
        subprocess.run([str(args.binary), str(json_file)], check=True)

import numpy as np

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


def _resolution_styles(
    nx_list,
    cmap_name="Blues",
    alpha=1.0,
    t_min=0.2,   # skip near-white colors
    t_max=0.9,
):
    """
    Color varies with resolution, avoiding invisible light colors.
    Alpha is fixed.
    """

    nx_sorted = sorted(nx_list)
    n = len(nx_sorted)

    cmap = plt.get_cmap(cmap_name)

    styles = {}
    for i, nx in enumerate(nx_sorted):
        t = i / (n - 1) if n > 1 else 0.5
        t = t_min + t * (t_max - t_min)
        styles[nx] = {
            "color": cmap(t),
            "alpha": alpha,
        }

    return styles
 
 
# ---------------------------------------------------------------------------
# plot subcommand — velocity at probe point over time
# ---------------------------------------------------------------------------
 
def cmd_plot(args):
    r = load_run(args.input, pathlib.Path(args.csv), args.ix, args.iy)
 
    print(f"\n{'='*57}")
    print(f"  Probe      : grid ({r['probe_ix']}, {r['probe_iy']})  →  "
          f"(x={r['x_probe']:.4f}, y={r['y_probe']:.4f})")
    print(f"  Analytical : u0 = {r['ana_u0']:+.6f},  v0 = {r['ana_v0']:+.6f}")
    print(f"  Sim u      : min={r['sim_u'].min():.6f}  max={r['sim_u'].max():.6f}")
    print(f"  Sim v      : min={r['sim_v'].min():.6f}  max={r['sim_v'].max():.6f}")
    print(f"{'='*57}\n")
 
    times = r["times"]
 
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"Uniform flow — velocity at probe\n"
        f"Probe ({r['probe_ix']}, {r['probe_iy']})  "
        f"($x$={r['x_probe']:.3f}, $y$={r['y_probe']:.3f})",
        fontsize=12,
    )
 
    for ax, sim, ana0, lbl, color in [
        (axes[0], r["sim_u"], r["ana_u0"], "u", "steelblue"),
        (axes[1], r["sim_v"], r["ana_v0"], "v", "tomato"),
    ]:
        ax.axhline(ana0, color="gray", lw=1.2, ls=":", label="Analytical")
        ax.plot(times, sim, color=color, lw=1.5, label="Simulation")
        ax.set_ylabel(f"${lbl}$ velocity")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    axes[1].set_xlabel("Time (s)", fontsize=20)
    plt.tight_layout()
    plt.show()
 
 
# ---------------------------------------------------------------------------
# slice subcommand — velocity profile along y at multiple timesteps
# ---------------------------------------------------------------------------
 
def plot_slice(csv_path: pathlib.Path, json_path: pathlib.Path, every: int = 1):
    """
    Plot vertical velocity slices:
        x-axis: velocity
        y-axis: physical position
        one curve per stored timestep
    """
    with open(json_path) as f:
        data = json.load(f)
 
    dx        = data["space_steps"]
    dt        = data["delta_t"]
    slice_cfg = data["slice_csv"]
    timesteps = slice_cfg["timesteps"]   # exact step numbers for each row
 
    bc_inflow = next(bc for bc in data["bc"] if "speed_x" in bc)
    U0        = bc_inflow["speed_x"]
 
    df = read_csv(csv_path)
 
    # columns: vx_<i>_<j> — extract j indices
    slice_cols = [c for c in df.columns if c.startswith("vx_")]
    if not slice_cols:
        raise ValueError("No slice columns (vx_*) found in CSV.")
 
    j_indices  = [int(c.split("_")[-1]) for c in slice_cols]
    y_coords   = np.array(j_indices) * dx
 
    order      = np.argsort(y_coords)
    y_coords   = y_coords[order]
    slice_cols = [slice_cols[i] for i in order]
 
    fig, ax = plt.subplots(figsize=(6, 8))
 
    for idx, row in df.iterrows():
        if idx % every != 0:
            continue
        step       = timesteps[idx]        # actual step number from JSON
        time       = step * dt
        vx_profile = row[slice_cols].values.astype(float)
        ax.plot(vx_profile, y_coords, lw=1.5,
                label=f"t = {time:.3f}s  (step {step})")
 
    # analytical reference: vertical line at U0
    ax.axvline(U0, color="black", lw=1.2, ls=":",
               label=f"Analytical $U_0$ = {U0}")
 
    ax.set_xlabel(r"$v_x$", fontsize=20)
    ax.set_ylabel(r"$y$", fontsize=20)
    ax.set_title(
        f"Uniform flow — $v_x$ slice at $i$ = {slice_cfg['i']}  "
        f"($x$ = {slice_cfg['i'] * dx:.3f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def cmd_compare(args):
    """
    Compare vx and vy vertical slices across all resolutions (uniform flow).
    """

    styles = _resolution_styles(NY_LIST, cmap_name="Blues")

    fig_x, ax_x = plt.subplots(figsize=(6, 7))
    fig_y, ax_y = plt.subplots(figsize=(6, 7))

    y_ref = None

    for ny in NY_LIST:
        json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
        run_dir   = ROOT_OUT / f"ny_{ny}"

        vx_csv = run_dir / "vx_slice.csv"
        vy_csv = run_dir / "vy_slice.csv"

        if not json_path.exists() or not vx_csv.exists() or not vy_csv.exists():
            print(f"ny={ny}: missing files, skipping.")
            continue

        with open(json_path) as f:
            data = json.load(f)

        bc_inflow = next(bc for bc in data["bc"] if "speed_x" in bc)
        U0 = bc_inflow["speed_x"]
        V0 = bc_inflow.get("speed_y", 0.0)

        dx        = data["space_steps"]
        slice_x  = data["slice_x_csv"]
        slice_y  = data["slice_y_csv"]

        sid = 1  # same slice index for all resolutions

        # ---- vx slice ------------------------------------------------------
        df_x = read_csv(vx_csv)
        vx_cols = sorted(
            [c for c in df_x.columns if c.startswith("vx_")],
            key=lambda c: int(c.split("_")[-1]),
        )

        j_idx = np.array([int(c.split("_")[-1]) for c in vx_cols])
        y     = (j_idx + 0.5) * dx

        vx_num = df_x.iloc[sid][vx_cols].values.astype(float)

        ax_x.plot(
            vx_num, y,
            lw=1.5,
            color=styles[ny]["color"],
            alpha=styles[ny]["alpha"],
            label=f"ny={ny}",
        )

        # ---- vy slice ------------------------------------------------------
        df_y = read_csv(vy_csv)
        vy_cols = sorted(
            [c for c in df_y.columns if c.startswith("vy_")],
            key=lambda c: int(c.split("_")[-1]),
        )

        vy_num = df_y.iloc[sid][vy_cols].values.astype(float)

        ax_y.plot(
            vy_num, y,
            lw=1.5,
            color=styles[ny]["color"],
            alpha=styles[ny]["alpha"],
            label=f"ny={ny}",
        )

        if y_ref is None:
            y_ref = y

    # ---- analytical references --------------------------------------------
    ax_x.axvline(1.0, color="black", lw=2.5, ls="--", zorder = 10, label=f"$U_0 = 1.0$")
    ax_y.axvline(0.0, color="black", lw=2.5, ls="--", zorder = 10, label=f"$V_0 = 0.0$")

    for ax, comp in [(ax_x, "x"), (ax_y, "y")]:
        ax.set_xlabel(rf"$v_{comp}$ (m/s)", fontsize=24)
        ax.set_ylabel(r"$y$ (m)", fontsize=24)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    """ fig_x.suptitle("Uniform flow — $v_x$ slice convergence", fontsize=14)
    fig_y.suptitle("Uniform flow — $v_y$ slice convergence", fontsize=14) """

    fig_x.tight_layout()
    fig_y.tight_layout()
    plt.show()

def cmd_convergence(args):
    """
    Plot Cl and vy at probe point over time for all resolutions
    and show dominant frequency convergence with respect to ny.
    """

    styles = _resolution_styles(NY_LIST, cmap_name="Blues")

    fig_x, ax_x = plt.subplots(figsize=(9, 7))
    fig_y, ax_y = plt.subplots(figsize=(9, 7))

    # ---- storage for frequency convergence -----------------------------
    ny_vals = []
    freq_cl_vals = []
    freq_vy_vals = []

    V0_ref = None

    for ny in NY_LIST:
        json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
        csv_path  = ROOT_OUT / f"ny_{ny}" / f"metrics_SL_{ny}.csv"

        if not json_path.exists() or not csv_path.exists():
            print(f"ny={ny}: missing files, skipping.")
            continue

        # ---- load simulation metadata -----------------------------------
        with open(json_path) as f:
            data = json.load(f)

        bc_inflow = next(bc for bc in data["bc"] if "speed_x" in bc)
        V0 = bc_inflow.get("speed_y", 0.0)

        if V0_ref is None:
            V0_ref = V0

        dt = data["delta_t"]
        nt = data["nt"]
        ix, iy = data["metrics"][0]["idx"]

        # remove transient
        cl_start = int(2 * nt / 3)

        times = np.linspace(cl_start * dt, nt * dt, nt - cl_start)

        # ---- load time series -------------------------------------------
        df = read_csv(csv_path)

        cl = df["Cl"].values[cl_start:nt]
        v  = df[f"vy_{ix}_{iy}"].values[cl_start:nt]

        # ---- dominant frequency extraction ------------------------------
        f_cl = dominant_frequency(cl, dt)
        f_vy = dominant_frequency(v, dt)

        ny_vals.append(ny)
        freq_cl_vals.append(f_cl)
        freq_vy_vals.append(f_vy)

        print(f"ny={ny:4d} | f_Cl = {f_cl:.4f} Hz | f_vy = {f_vy:.4f} Hz")

        # ---- time signals -----------------------------------------------
        ax_x.plot(
            times, cl,
            lw=1,
            color=styles[ny]["color"],
            alpha=styles[ny]["alpha"],
            label=f"ny={ny}",
        )

        ax_y.plot(
            times, v,
            lw=1,
            color=styles[ny]["color"],
            alpha=styles[ny]["alpha"],
            label=f"ny={ny}",
        )

    # ---- reference -----------------------------------------------------
    ax_y.axhline(
        V0_ref, color="black", lw=2.5, ls="--",
        label=r"$V_0$"
    )

    # ---- formatting: Cl time history ----------------------------------
    ax_x.set_xlabel("Time (s)", fontsize=24)
    ax_x.set_ylabel("Pressure base coefficient", fontsize=24)
    ax_x.tick_params(labelsize=14)
    ax_x.legend(fontsize=12)
    ax_x.grid(True, alpha=0.3)

    # ---- formatting: vy time history ----------------------------------
    ax_y.set_xlabel("Time (s)", fontsize=24)
    ax_y.set_ylabel(r"$v_y$ (m/s)", fontsize=24)
    ax_y.tick_params(labelsize=14)
    ax_y.legend(fontsize=12)
    ax_y.grid(True, alpha=0.3)

    fig_x.tight_layout()
    fig_y.tight_layout()

    # ==== FREQUENCY CONVERGENCE PLOT ===================================
    fig_f, ax_f = plt.subplots(figsize=(9, 7))

    ax_f.plot(
        ny_vals, freq_cl_vals,
        "o-", lw=2, ms=7,
        label=r"$f_{Cl}$"
    )

    ax_f.plot(
        ny_vals, freq_vy_vals,
        "s--", lw=2, ms=7,
        label=r"$f_{v_y}$"
    )

    ax_f.set_xlabel(r"Resolution $n_y$", fontsize=24)
    ax_f.set_ylabel("Dominant frequency (Hz)", fontsize=24)
    ax_f.tick_params(labelsize=14)
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(fontsize=14)

    fig_f.tight_layout()

    plt.show()

def cmd_error(args):
    """
    Plot mean absolute error |vx - U0| averaged over the full slice
    and all stored timesteps, vs dx — one point per resolution.
    """
    with open(args.input) as f:
        base_data = json.load(f)
    bc_inflow = next(bc for bc in base_data["bc"] if "speed_x" in bc)
    U0 = bc_inflow["speed_x"]

    dx_arr  = []
    err_arr = []

    for ny in NY_LIST:
        json_path = ROOT_SIM / f"von_karman_ny_{ny}.json"
        csv_path  = ROOT_OUT / f"ny_{ny}" / "vx_slice.csv"

        if not json_path.exists() or not csv_path.exists():
            print(f"ny={ny}: missing files, skipping.")
            continue

        with open(json_path) as f:
            data = json.load(f)

        dx = data["space_steps"]
        df = read_csv(csv_path)

        slice_cols = sorted(
            [c for c in df.columns if c.startswith("vx_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        if not slice_cols:
            print(f"ny={ny}: no slice columns, skipping.")
            continue

        # mean |vx - U0| over all rows and all slice points
        vx_all = df[slice_cols].values.astype(float)
        error  = np.mean(np.abs(vx_all - U0))

        dx_arr.append(dx)
        err_arr.append(error)
        print(f"ny={ny:>4}  dx={dx:.5f}  mean_error={error:.4e}")

    if len(dx_arr) < 2:
        print("Not enough runs to plot.")
        return

    dx_arr  = np.array(dx_arr)
    err_arr = np.array(err_arr)

    # log-log fit to get convergence order
    coeffs = np.polyfit(np.log(dx_arr), np.log(err_arr), 1)
    order  = coeffs[0]
    c_fit  = np.exp(coeffs[1])
    print(f"\n  Convergence order : p = {order:.3f}  (error ~ dx^p)")

    dx_fine  = np.logspace(np.log10(dx_arr.min()), np.log10(dx_arr.max()), 100)
    err_fine = c_fit * dx_fine ** order

    # reference slopes
    mid_err = np.median(err_arr)
    mid_dx  = np.median(dx_arr)
    ref1    = mid_err * (dx_fine / mid_dx) ** 1
    ref2    = mid_err * (dx_fine / mid_dx) ** 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(dx_arr, err_arr, "ko-", lw=1.5, markersize=6,
              label=r"Mean $|v_x - U_0|$")
    ax.loglog(dx_fine, err_fine, "-", color="black", lw=1, alpha=0.4,
              label=rf"fit: $C \cdot \Delta x^{{{order:.2f}}}$")
    ax.loglog(dx_fine, ref1, ":", color="gray", lw=1.2, label="slope = 1")
    ax.loglog(dx_fine, ref2, "--", color="gray", lw=1.2, label="slope = 2")

    ax.set_xlabel(r"Grid spacing $\Delta x$", fontsize=12)
    ax.set_ylabel(r"Mean $|v_x - U_0|$", fontsize=12)
    ax.set_title("Uniform flow — error convergence", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Uniform flow: validation of numerical method"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
 
    # run
    run_p = subparsers.add_parser("run", help="Build configs and launch simulations")
    run_p.add_argument("binary", type=pathlib.Path, help="Simulation binary")
    run_p.add_argument("-i", "--input",  required=True, type=pathlib.Path,
                       help="Base JSON config")
    run_p.add_argument("-o", "--output", required=True, type=pathlib.Path,
                       help="Output directory")
 
    # plot
    plot_p = subparsers.add_parser("plot",
        help="Plot velocity at probe point over time")
    plot_p.add_argument("--csv", required=True, type=str,
                        help="Path to simulation CSV output")
    plot_p.add_argument("-i", "--input", required=True, type=pathlib.Path,
                        help="JSON config used for that run")
    plot_p.add_argument("--ix", required=False, type=int, default=None,
                        help="Grid index ix of the probe point (default: nx//4)")
    plot_p.add_argument("--iy", required=False, type=int, default=None,
                        help="Grid index iy of the probe point (default: ny//4)")
 
    # slice
    slice_p = subparsers.add_parser("slice",
        help="Plot velocity profile along y at multiple timesteps")
    slice_p.add_argument("--csv", required=True, type=pathlib.Path,
                         help="Slice CSV file (vx_slice.csv)")
    slice_p.add_argument("-i", "--input", required=True, type=pathlib.Path,
                         help="JSON config used for that run")
    slice_p.add_argument("--every", type=int, default=1,
                         help="Plot every Nth stored timestep (default: 1)")
    
    compare_p = subparsers.add_parser("compare",
        help="Plot last vx slice for all resolutions on one figure")
    compare_p.add_argument("-i", "--input", required=True, type=pathlib.Path,
                        help="Base JSON config")
    
    conv_p = subparsers.add_parser("convergence",
        help="Plot vx at probe point over time for all resolutions")
    conv_p.add_argument("-i", "--input", required=True, type=pathlib.Path,
                        help="Base JSON config")
    
    error_p = subparsers.add_parser("error",
        help="Plot mean slice error over time for all resolutions")
    error_p.add_argument("-i", "--input", required=True, type=pathlib.Path,
                          help="Base JSON config")
 
    args = parser.parse_args()
 
    if args.mode == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        folder = build_folder(args)
        launch_sims(args, folder)
 
    elif args.mode == "plot":
        cmd_plot(args)
 
    elif args.mode == "slice":
        plot_slice(args.csv, args.input, args.every)

    elif args.mode == "compare":
        cmd_compare(args)
 
    elif args.mode == "convergence":
        cmd_convergence(args)

    elif args.mode == "error":
        cmd_error(args)
 
 
if __name__ == "__main__":
    main()
 