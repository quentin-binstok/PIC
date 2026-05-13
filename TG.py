"""
Taylor–Green vortex — validation of the method.

Analytical solution in [0, π] × [0, π]:

    u(x,y,t) =  sin(x) cos(y) exp(-2 ν t)
    v(x,y,t) = -cos(x) sin(y) exp(-2 ν t)

Subcommands:
    run         — build JSON configs and launch all simulations
    plot        — plot velocity at one probe point over time
    slice       — plot velocity profile along y at multiple timesteps
    compare     — compare last slice across resolutions
    convergence — probe evolution for all resolutions
    error       — mean slice error vs dx
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import json
import argparse
import pathlib
import copy
import subprocess


NX_LIST = [25, 50, 100, 200, 400]


# ---------------------------------------------------------------------------
# Analytical Taylor–Green solution
# ---------------------------------------------------------------------------

def tg_u(x, y, t, nu):
    return np.sin(x) * np.cos(y) * np.exp(-2.0 * nu * t)

def tg_v(x, y, t, nu):
    return -np.cos(x) * np.sin(y) * np.exp(-2.0 * nu * t)


# ---------------------------------------------------------------------------
# Shared helper: load one run
# ---------------------------------------------------------------------------

def load_run(json_path, csv_path, probe_ix=None, probe_iy=None):

    with open(json_path) as f:
        data = json.load(f)

    dx = data["space_steps"]
    dt = data["delta_t"]
    nt = data["nt"]
    nx = data["grid"][0]
    nu = data.get("viscosity", 0.0)

    if probe_ix is None:
        probe_ix = nx // 4
    if probe_iy is None:
        probe_iy = nx // 4

    x = probe_ix * dx
    y = probe_iy * dx

    times = np.linspace(0.0, nt * dt, nt, dtype=float)[1:]

    df    = read_csv(csv_path)
    col_u = f"vx_{probe_ix}_{probe_iy}"
    col_v = f"vy_{probe_ix}_{probe_iy}"

    sim_u = df[col_u].values[1:nt]
    sim_v = df[col_v].values[1:nt]

    ana_u = tg_u(x, y, times, nu)
    ana_v = tg_v(x, y, times, nu)

    return dict(
        nx=nx, dx=dx, dt=dt, nt=nt, nu=nu,
        probe_ix=probe_ix, probe_iy=probe_iy,
        x=x, y=y, times=times,
        sim_u=sim_u, sim_v=sim_v,
        ana_u=ana_u, ana_v=ana_v,
    )


# ---------------------------------------------------------------------------
# Folder builder (same logic as uniform flow)
# ---------------------------------------------------------------------------

def build_folder(args):

    temp = pathlib.Path("taylor_green_sim")
    temp.mkdir(exist_ok=True)
    pathlib.Path("Taylor_Green").mkdir(exist_ok=True)

    with open(args.input) as f:
        base = json.load(f)

    base_dx  = base["space_steps"]
    base_dt  = base["delta_t"]
    base_nt  = base["nt"]
    base_T   = base_dt * base_nt
    base_cfl = base_dt / base_dx

    def slice_timesteps(nt):
        return [1, nt // 3, 2 * nt // 3, nt - 2]

    for nx in NX_LIST:
        ratio = nx / base["grid"][0]
        dx    = base_dx / ratio
        dt    = base_cfl * dx
        nt    = int(round(base_T / dt))

        work = copy.deepcopy(base)
        work["grid"]        = [nx, nx]
        work["space_steps"] = dx
        work["delta_t"]     = dt
        work["nt"]          = nt
        work["dir"]         = f"Taylor_Green/nx_{nx}"

        work["metrics"][0]["idx"] = [nx // 4, nx // 4]
        work["metrics"][1]["idx"] = [nx // 4, nx // 4]

        work["slice_x_csv"] = {
            "enabled": True,
            "field": "vx",
            "type": "vertical",
            "i": 2* nx // 3,
            "j_start": 1,
            "j_end": nx - 1,
            "timesteps": slice_timesteps(nt),
            "file": "vx_slice.csv",
        }

        work["slice_y_csv"] = {
            "enabled": True,
            "field": "vy",
            "type": "vertical",
            "i": 2* nx // 3,
            "j_start": 1,
            "j_end": nx - 1,
            "timesteps": slice_timesteps(nt),
            "file": "vy_slice.csv",
        }

        with open(temp / f"tgv_nx_{nx}.json", "w") as f:
            json.dump(work, f, indent=2)

    return temp


def launch_sims(args, folder):
    for json_file in sorted(folder.glob("tgv_nx_*.json")):
        subprocess.run([str(args.binary), str(json_file)], check=True)


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
# Plot: probe evolution
# ---------------------------------------------------------------------------

def cmd_plot(args):
    r = load_run(args.input, args.csv, args.ix, args.iy)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ax, sim, ana, lbl in [
        (axes[0], r["sim_u"], r["ana_u"], "u"),
        (axes[1], r["sim_v"], r["ana_v"], "v"),
    ]:
        ax.plot(r["times"], sim, lw=1.5, label="Simulation")
        ax.plot(r["times"], ana, "k--", lw=2, label="Analytical")
        ax.set_ylabel(lbl)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[1].set_xlabel("Time")
    fig.suptitle(
        f"Taylor–Green vortex — probe "
        f"(x={r['x']:.3f}, y={r['y']:.3f}), ν={r['nu']}"
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Slice plot
# ---------------------------------------------------------------------------

def plot_slice(csv_path, json_path):

    with open(json_path) as f:
        data = json.load(f)

    dx = data["space_steps"]
    dt = data["delta_t"]
    nu = data.get("viscosity", 0.0)
    slice_cfg = data["slice_csv"]
    timesteps = slice_cfg["timesteps"]

    df = read_csv(csv_path)

    slice_cols = sorted(
        [c for c in df.columns if c.startswith("vx_")],
        key=lambda c: int(c.split("_")[-1]),
    )

    j_idx  = [int(c.split("_")[-1]) for c in slice_cols]
    y      = np.array(j_idx) * dx
    x      = slice_cfg["i"] * dx

    fig, ax = plt.subplots(figsize=(6, 8))

    for k, row in df.iterrows():
        t = timesteps[k] * dt
        ax.plot(
            row[slice_cols], y, lw=1.5,
            label=f"t={t:.3f}"
        )

        ax.plot(
            tg_u(x, y, t, nu), y,
            "k:", lw=2
        )

    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Taylor–Green vortex — vertical slice")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def cmd_convergence(args):
    """
    Taylor–Green vortex:
    Plot vx and vy at a probe point over time for all resolutions,
    using TWO SEPARATE figures.
    """

    styles = _resolution_styles(NX_LIST, cmap_name="Blues") 

    fig_x, ax_x = plt.subplots(figsize=(9, 7))
    fig_y, ax_y = plt.subplots(figsize=(9, 7))

    # Will store analytical reference once
    ana_u_ref = None
    ana_v_ref = None
    times_ref = None

    for nx in NX_LIST:
        json_path = pathlib.Path("taylor_green_sim") / f"tgv_nx_{nx}.json"
        csv_path  = pathlib.Path("Taylor_Green") / f"nx_{nx}" / "metrics_SL_GL.csv"

        if not json_path.exists() or not csv_path.exists():
            print(f"nx={nx}: missing files, skipping.")
            continue

        # --- load run config ------------------------------------------------
        with open(json_path) as f:
            data = json.load(f)

        dx = data["space_steps"]
        dt = data["delta_t"]
        nt = data["nt"]
        nu = data.get("viscosity", 0.0)

        ix, iy = data["metrics"][0]["idx"]

        # cell-centered coordinates (CRITICAL)
        x = (ix + 0.5) * dx
        y = (iy + 0.5) * dx

        times = np.linspace(0.0, nt * dt, nt, dtype=float)[1:]

        df = read_csv(csv_path)

        sim_u = df[f"vx_{ix}_{iy}"].values[1:nt]
        sim_v = df[f"vy_{ix}_{iy}"].values[1:nt]

        ax_x.plot(
            times, sim_u,
            lw=1.5,
            color=styles[nx]["color"],
            alpha=styles[nx]["alpha"],
            label=f"nx={nx}",
        )

        ax_y.plot(
            times, sim_v,
            lw=1.5,
            color=styles[nx]["color"],
            alpha=styles[nx]["alpha"],
            label=f"nx={nx}",
        )

        # store analytical reference once (same physics for all nx)
        if ana_u_ref is None:
            ana_u_ref = tg_u(x, y, times, 0)
            ana_v_ref = tg_v(x, y, times, 0)
            times_ref = times

    # --- analytical references --------------------------------------------
    ax_x.plot(
        times_ref, ana_u_ref,
        color="black", lw=2.5, ls="--", zorder =10,
        label="Inviscid solution",
    )

    ax_y.plot(
        times_ref, ana_v_ref,
        color="black", lw=2.5, ls="--", zorder =10,
        label="Inviscid solution",
    )

    # --- formatting --------------------------------------------------------
    ax_x.set_xlabel("Time (s)", fontsize=24)
    ax_x.set_ylabel(r"$v_x$ (m/s)", fontsize=24)
    """ ax_x.set_title("Taylor–Green vortex — $v_x$ convergence", fontsize=14) """
    ax_x.tick_params(labelsize=14)
    ax_x.legend(fontsize=12)
    ax_x.grid(True, alpha=0.3)

    ax_y.set_xlabel("Time (s)", fontsize=24)
    ax_y.set_ylabel(r"$v_y$ (m/s)", fontsize=24)
    """ ax_y.set_title("Taylor–Green vortex — $v_y$ convergence", fontsize=14) """
    ax_y.tick_params(labelsize=14)
    ax_y.legend(fontsize=12)
    ax_y.grid(True, alpha=0.3)

    fig_x.tight_layout()
    fig_y.tight_layout()
    plt.show()

def cmd_compare(args):
    """
    Compare vx and vy vertical slices across all resolutions,
    using separate vx_slice.csv and vy_slice.csv files.
    """

    styles = _resolution_styles(NX_LIST, cmap_name="Blues")

    with open(args.input) as f:
        base = json.load(f)

    nu = base.get("viscosity", 0.0)

    fig_x, ax_x = plt.subplots(figsize=(6, 7))
    fig_y, ax_y = plt.subplots(figsize=(6, 7))

    # Will store reference coordinates for analytical curve
    x_ref = y_ref = t_ref = None

    for nx in NX_LIST:
        json_path = pathlib.Path("taylor_green_sim") / f"tgv_nx_{nx}.json"
        dir_path  = pathlib.Path("Taylor_Green") / f"nx_{nx}"

        vx_csv = dir_path / "vx_slice.csv"
        vy_csv = dir_path / "vy_slice.csv"

        if not json_path.exists() or not vx_csv.exists() or not vy_csv.exists():
            print(f"nx={nx}: missing files, skipping.")
            continue

        with open(json_path) as f:
            data = json.load(f)

        dx = data["space_steps"]
        dt = data["delta_t"]

        slice_x_cfg = data["slice_x_csv"]
        slice_y_cfg = data["slice_y_csv"]

        # same slice index for all resolutions (last stored)
        sid  = -1
        step = slice_x_cfg["timesteps"][sid]
        t    = step * dt

        i = slice_x_cfg["i"]
        x = (i + 0.5) * dx

        # --- vx slice -------------------------------------------------------
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
            color=styles[nx]["color"],
            alpha=styles[nx]["alpha"],
            label=f"nx={nx}",
        )

        # --- vy slice -------------------------------------------------------
        df_y = read_csv(vy_csv)
        vy_cols = sorted(
            [c for c in df_y.columns if c.startswith("vy_")],
            key=lambda c: int(c.split("_")[-1]),
        )

        vy_num = df_y.iloc[sid][vy_cols].values.astype(float)

        ax_y.plot(
            vy_num, y,
            lw=1.5,
            color=styles[nx]["color"],
            alpha=styles[nx]["alpha"],
            label=f"nx={nx}",
        )

        # store reference coordinates once (from first valid run)
        if x_ref is None:
            x_ref = x
            y_ref = y
            t_ref = t

    # --- analytical references --------------------------------------------
    if x_ref is not None:
        vx_ana = tg_u(x_ref, y_ref, t_ref, nu)
        vy_ana = tg_v(x_ref, y_ref, t_ref, nu)

        ax_x.plot(
            vx_ana, y_ref,
            color="black", lw=2.5, ls="--", zorder = 10,
            label="Inviscid solution",
        )

        ax_y.plot(
            vy_ana, y_ref,
            color="black", lw=2.5, ls="--", zorder = 10,
            label="Inviscid solution",
        )

    # --- formatting --------------------------------------------------------
    for ax, comp in [(ax_x, "x"), (ax_y, "y")]:
        ax.set_xlabel(rf"$v_{comp}$ (m/s)", fontsize=24)
        ax.set_ylabel(r"$y$ (m)", fontsize=24)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    """ fig_x.suptitle("Taylor–Green vortex — $v_x$ slice convergence", fontsize=14)
    fig_y.suptitle("Taylor–Green vortex — $v_y$ slice convergence", fontsize=14) """

    fig_x.tight_layout()
    fig_y.tight_layout()
    plt.show()



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Taylor–Green vortex validation"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("binary", type=pathlib.Path)
    run_p.add_argument("-i", "--input", required=True, type=pathlib.Path)

    plot_p = sub.add_parser("plot")
    plot_p.add_argument("--csv", required=True, type=pathlib.Path)
    plot_p.add_argument("-i", "--input", required=True, type=pathlib.Path)
    plot_p.add_argument("--ix", type=int)
    plot_p.add_argument("--iy", type=int)

    slice_p = sub.add_parser("slice")
    slice_p.add_argument("--csv", required=True, type=pathlib.Path)
    slice_p.add_argument("-i", "--input", required=True, type=pathlib.Path)

    
    conv_p = sub.add_parser("convergence",
        help="Probe velocity over time for all resolutions")
    conv_p.add_argument("-i", "--input", required=True, type=pathlib.Path)

    compare_p = sub.add_parser("compare",
        help="Compare vx slice across all resolutions")
    compare_p.add_argument("-i", "--input", required=True, type=pathlib.Path)


    err_p = sub.add_parser("error")
    err_p.add_argument("-i", "--input", required=True, type=pathlib.Path)


    args = parser.parse_args()

    if args.mode == "run":
        folder = build_folder(args)
        launch_sims(args, folder)

    elif args.mode == "plot":
        cmd_plot(args)

    elif args.mode == "slice":
        plot_slice(args.csv, args.input)

    elif args.mode == "convergence":
        cmd_convergence(args)

    elif args.mode == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()