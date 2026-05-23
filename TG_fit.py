"""
Taylor-Green vortex — numerical dissipation analysis.
 
The flow is theoretically inviscid (ν=0), so the analytical solution is:
    u(x, y, t) =  sin(x) cos(y)      (constant in time)
    v(x, y, t) = -cos(x) sin(y)      (constant in time)
 
Any decay observed in the simulation is purely numerical dissipation.
We fit an effective viscosity ν_num by matching the simulated decay to:
    u_sim(t) ≈ u0 · exp(-2 · ν_num · t)
    v_sim(t) ≈ v0 · exp(-2 · ν_num · t)
 
giving a quantitative measure of the scheme's numerical dissipation.
 
Subcommands:
    run         — build JSON configs and launch all simulations
    plot        — plot velocity traces + decay fit for one run
    convergence — overlay all runs + plot ν_num vs dx (log-log convergence order)
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
 
 

NX_LIST = [50, 100, 150, 200, 250]
NT_LIST = [250, 500, 1000, 2000, 4000]
ROOT_SIM = pathlib.Path("taylor_green_sim/PIC_dt")
ROOT_OUT = pathlib.Path("Taylor_Green/PIC_dt")
SOLVER = "PIC"

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
# Analytical Taylor–Green solution
# ---------------------------------------------------------------------------
 
def tg_u(x, y, t, nu):
    return np.sin(x) * np.cos(y) * np.exp(-2.0 * nu * t)
 
def tg_v(x, y, t, nu):
    return -np.cos(x) * np.sin(y) * np.exp(-2.0 * nu * t)
 
# ---------------------------------------------------------------------------
# Decay model for curve fitting
# ---------------------------------------------------------------------------
 
def decaying_exponential(t: np.ndarray, u0: float, nu_num: float) -> np.ndarray:
    """Model: u0 * exp(-2 * nu_num * t)"""
    return u0 * np.exp(-2.0 * nu_num * t)
 
def fit_numerical_viscosity(
    times: np.ndarray,
    signal: np.ndarray,
    u0_guess: float,
) -> tuple[float, float, float, float]:
    """
    Fit the decay curve and return (nu_num, nu_std, u0_fit, u0_std).
    t=0 is assumed already excluded from times and signal.
    """
    t_fit = times
    s_fit = signal
 
    sign0 = np.sign(u0_guess)
    valid = (sign0 * s_fit) > 0
    if valid.sum() < 3:
        raise RuntimeError("Not enough valid points for log-linear pre-fit.")
 
    log_s    = np.log(np.abs(s_fit[valid]))
    coeffs   = np.polyfit(t_fit[valid], log_s, 1)   # slope = -2*nu_num
    nu_guess = -coeffs[0] / 2.0
    u0_lin   = np.exp(coeffs[1])
 
    p0 = [np.sign(u0_guess) * u0_lin, max(nu_guess, 1e-10)]
 
    try:
        popt, pcov = curve_fit(
            decaying_exponential, t_fit, s_fit,
            p0=p0,
            bounds=([-np.inf, 0.0], [np.inf, np.inf]),
            maxfev=10_000,
        )
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        popt = np.array(p0)
        perr = np.array([np.nan, np.nan])
 
    u0_fit, nu_num = popt
    u0_std, nu_std = perr
    return nu_num, nu_std, u0_fit, u0_std
 
 
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
 
    times = np.linspace(0.0, nt * dt, nt, dtype=float)
 
    df    = read_csv(csv_path)
    col_u = f"vx_{probe_ix}_{probe_iy}"
    col_v = f"vy_{probe_ix}_{probe_iy}"

    energy = df["energy"].values
    sim_u = df[col_u].values
    sim_v = df[col_v].values
 
    ana_u = tg_u(x, y, times, nu)
    ana_v = tg_v(x, y, times, nu)
 
    return dict(
        nx=nx, dx=dx, dt=dt, nt=nt, nu=nu,
        probe_ix=probe_ix, probe_iy=probe_iy,
        x=x, y=y, times=times,
        sim_u=sim_u, sim_v=sim_v,
        ana_u=ana_u, ana_v=ana_v,
        energy=energy,
    )
 
 
# ---------------------------------------------------------------------------
# Folder builder
# ---------------------------------------------------------------------------
 
def build_folder(args):
 
    temp = ROOT_SIM
    temp.mkdir(exist_ok=True)
    ROOT_OUT.mkdir(exist_ok=True)
 
    with open(args.input) as f:
        base = json.load(f)
    
    base_T = 20.0
    base_L = np.pi
    CFL = 0.5
 
    def slice_timesteps(nt):
        return [1, nt // 3, 2 * nt // 3, nt - 2]
 
    for nx in NX_LIST:
        """ nx    = 100 """
        dx    = base_L / nx
        dt    = CFL * dx  # CFL condition for stability
        nt    = int(base_T / dt)
 
        work = copy.deepcopy(base)
        work["solver"]      = "pic"
        work["grid"]        = [nx, nx]
        work["space_steps"] = dx
        work["delta_t"]     = dt
        work["nt"]          = nt
        work["dir"]         = str(ROOT_OUT / f"nx_{nx}")
        work["sampling_rate"] = nt//25  # ~100 samples per run
        work["flip"]    = 0.0

 
        work["metrics"][0]["idx"] = [nx // 4, nx // 4]
        work["metrics"][1]["idx"] = [nx // 4, nx // 4]
        work["metrics"][2]["idx"] = [nx // 4, nx // 4]

        work["taylor_green"][0]["wavenumber"] = np.pi / base_L
 
        """ work["slice_x_csv"] = {
            "enabled": True,
            "field": "vx",
            "type": "vertical",
            "i": 2 * nx // 3,
            "j_start": 1,
            "j_end": nx - 1,
            "timesteps": slice_timesteps(nt),
            "file": "vx_slice.csv",
        }
 
        work["slice_y_csv"] = {
            "enabled": True,
            "field": "vy",
            "type": "vertical",
            "i": 2 * nx // 3,
            "j_start": 1,
            "j_end": nx - 1,
            "timesteps": slice_timesteps(nt),
            "file": "vy_slice.csv",
        } """
 
        with open(temp / f"tgv_nx_{nx}.json", "w") as f:
            json.dump(work, f, indent=2)
 
    return temp
 
 
def launch_sims(args, folder):
    for json_file in sorted(folder.glob("tgv_nx_*.json")):
        sbinary = os.path.abspath(args.binary)
        subprocess.run([sbinary, str(json_file)], check=True)
 
 
# ---------------------------------------------------------------------------
# Slice plot
# ---------------------------------------------------------------------------
 
def cmd_plot(args):
    """
    Taylor–Green vortex:
    Plot vx(t) and vy(t) for ALL resolutions on the SAME figures,
    including exponential decay fits and extracted numerical viscosity.
    """
 
    styles = _resolution_styles(NX_LIST)
 
    fig_u, ax_u = plt.subplots(figsize=(9, 6))
    fig_v, ax_v = plt.subplots(figsize=(9, 6))
 
    ana_u_ref = None
    ana_v_ref = None
    times_ref = None
 
    for nx in NX_LIST:
 
        json_path = ROOT_SIM / f"tgv_nx_{nx}.json"
        csv_path  = ROOT_OUT / f"nx_{nx}" / "metrics_SL_GL.csv"
 
        if not json_path.exists() or not csv_path.exists():
            print(f"nt={nx}: missing files, skipping.")
            continue
 
        run = load_run(json_path, csv_path)
 
        with open(json_path) as f:
            data = json.load(f)
 
        dx = data["space_steps"]
        ix, iy = data["metrics"][0]["idx"]
        x = (ix + 0.5) * dx
        y = (iy + 0.5) * dx
 
        times = run["times"][1:]
        sim_u = run["sim_u"][1:]
        sim_v = run["sim_v"][1:]

        k = data["taylor_green"][0]["wavenumber"]
 
        """ cut = int(0.1 * len(times))
        times = times[cut:]
        sim_u = sim_u[cut:]
        sim_v = sim_v[cut:] """
 
        nu_u, _, u0_u, _ = fit_numerical_viscosity(times, sim_u, u0_guess=sim_u[0])
        nu_v, _, u0_v, _ = fit_numerical_viscosity(times, sim_v, u0_guess=sim_v[0])
 
        fit_u = decaying_exponential(times, u0_u, nu_u)
        fit_v = decaying_exponential(times, u0_v, nu_v)
 
        color = styles[nx]["color"]
 
        ax_u.plot(times, sim_u, color=color, lw=LINE_WIDTH, label=rf"$n_x={nx}$")
        ax_u.plot(times, fit_u, color=color, lw=FIT_WIDTH, ls="--")
 
        ax_v.plot(times, sim_v, color=color, lw=LINE_WIDTH, label=rf"$n_x={nx}$")
        ax_v.plot(times, fit_v, color=color, lw=FIT_WIDTH, ls="--")
 
        ana_u_ref = tg_u(k*x, k*y, times, 0)
        ana_v_ref = tg_v(k*x, k*y, times, 0)
        times_ref = times
 
        print(f"nt={nx:4d} | nu_num(vx)={nu_u:.4e} | nu_num(vy)={nu_v:.4e}")
 
    for ax, ana_ref, ylabel, title in [
        (ax_u, ana_u_ref, r"$v_x$", "Taylor–Green vortex — $v_x(t)$ numerical decay"),
        (ax_v, ana_v_ref, r"$v_y$", "Taylor–Green vortex — $v_y(t)$ numerical decay"),
    ]:
        ax.plot(times_ref, ana_ref, color="black", lw=FIT_WIDTH, ls="--",
                zorder=10, label="Inviscid solution")
        ax.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        apply_style(ax)
 
    fig_u.tight_layout()
    fig_v.tight_layout()
    
    """ # ✅ Ensure output directory exists
    ROOT_OUT.mkdir(parents=True, exist_ok=True)

    # ✅ Save figures
    fig_u.savefig(ROOT_OUT / f"conv_vx_{SOLVER}.pdf", format="pdf")
    fig_v.savefig(ROOT_OUT / f"conv_vy_{SOLVER}.pdf", format="pdf")

    print(f"Saved: {ROOT_OUT / f'conv_vx_{SOLVER}.pdf'}")
    print(f"Saved: {ROOT_OUT / f'conv_vy_{SOLVER}.pdf'}") """

    plt.show()
 
def cmd_convergence(args):
    """
    Taylor–Green vortex:
    Plot vx and vy at a probe point over time for all resolutions.
    """
 
    styles = _resolution_styles(NX_LIST)
 
    fig_x, ax_x = plt.subplots(figsize=(9, 7))
    fig_y, ax_y = plt.subplots(figsize=(9, 7))
 
    for nx in NX_LIST:
        json_path = ROOT_SIM / f"tgv_nx_{nx}.json"
        csv_path  = ROOT_OUT / f"nx_{nx}" / "metrics_SL_GL.csv"
 
        if not json_path.exists() or not csv_path.exists():
            print(f"nx={nx}: missing files, skipping.")
            continue
 
        with open(json_path) as f:
            data = json.load(f)
 
        dx = data["space_steps"]
        dt = data["delta_t"]
        nt = data["nt"]
 
        ix, iy = data["metrics"][0]["idx"]
        x = (ix+0.5) * dx
        y = (iy+0.5) * dx

        k = data["taylor_green"][0]["wavenumber"]
 
        times = np.linspace(0.0, nt * dt, nt, dtype=float)[1:]
 
        df = read_csv(csv_path)
        sim_u = df[f"vx_{ix}_{iy}"].values[1:nt]
        sim_v = df[f"vy_{ix}_{iy}"].values[1:nt]
 
        color = styles[nx]["color"]
 
        ax_x.plot(times, sim_u, lw=LINE_WIDTH, color=color, label=f"nx={nx}")
        ax_y.plot(times, sim_v, lw=LINE_WIDTH, color=color, label=f"nx={nx}")
 
        ana_u_ref = tg_u(k*x, k*y, times, 0)
        ana_v_ref = tg_v(k*x, k*y, times, 0)
        times_ref = times
 
    for ax, ana_ref, ylabel in [
        (ax_x, ana_u_ref, r"$v_x$ (m/s)"),
        (ax_y, ana_v_ref, r"$v_y$ (m/s)"),
    ]:
        ax.plot(times_ref, ana_ref, color="black", lw=FIT_WIDTH, ls="--",
                zorder=10, label="Inviscid solution")
        ax.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        apply_style(ax)
 
    fig_x.tight_layout()
    fig_y.tight_layout()

    
# ✅ Ensure output directory exists
    ROOT_OUT.mkdir(parents=True, exist_ok=True)

    # ✅ Save figures
    fig_x.savefig(ROOT_OUT / f"vx_{SOLVER}.pdf", format="pdf")
    fig_y.savefig(ROOT_OUT / f"vy_{SOLVER}.pdf", format="pdf")

    print(f"Saved: {ROOT_OUT / f'vx_{SOLVER}.pdf'}")
    print(f"Saved: {ROOT_OUT / f'vy_{SOLVER}.pdf'}")

    plt.show()
 
def cmd_compare(args):
    """
    Compare vx and vy vertical slices across all resolutions.
    """
 
    styles = _resolution_styles(NT_LIST)
 
    fig_x, ax_x = plt.subplots(figsize=(6, 7))
    fig_y, ax_y = plt.subplots(figsize=(6, 7))
 
    x_ref = y_ref = t_ref = None
 
    for nx in NX_LIST:
        json_path = ROOT_SIM / f"tgv_nx_{nx}.json"
        dir_path  = ROOT_OUT / f"nx_{nx}"
        vx_csv    = dir_path / "vx_slice.csv"
        vy_csv    = dir_path / "vy_slice.csv"
 
        if not json_path.exists() or not vx_csv.exists() or not vy_csv.exists():
            print(f"nx={nx}: missing files, skipping.")
            continue
 
        with open(json_path) as f:
            data = json.load(f)
 
        dx  = data["space_steps"]
        dt  = data["delta_t"]
        slice_x_cfg = data["slice_x_csv"]
 
        sid  = -1
        step = slice_x_cfg["timesteps"][sid]
        t    = step * dt
        i    = slice_x_cfg["i"]
        x    = (i + 0.5) * dx
 
        df_x    = read_csv(vx_csv)
        vx_cols = sorted([c for c in df_x.columns if c.startswith("vx_")],
                         key=lambda c: int(c.split("_")[-1]))
        j_idx   = np.array([int(c.split("_")[-1]) for c in vx_cols])
        y       = (j_idx + 0.5) * dx
        vx_num  = df_x.iloc[sid][vx_cols].values.astype(float)
 
        df_y    = read_csv(vy_csv)
        vy_cols = sorted([c for c in df_y.columns if c.startswith("vy_")],
                         key=lambda c: int(c.split("_")[-1]))
        vy_num  = df_y.iloc[sid][vy_cols].values.astype(float)
 
        color = styles[nx]["color"]
        ax_x.plot(vx_num, y, lw=LINE_WIDTH, color=color, label=f"nx={nx}")
        ax_y.plot(vy_num, y, lw=LINE_WIDTH, color=color, label=f"nx={nx}")
 
        if x_ref is None:
            x_ref, y_ref, t_ref = x, y, t
 
    if x_ref is not None:
        vx_ana = tg_u(x_ref, y_ref, t_ref, 0)
        vy_ana = tg_v(x_ref, y_ref, t_ref, 0)
        ax_x.plot(vx_ana, y_ref, color="black", lw=FIT_WIDTH, ls="--",
                  zorder=10, label="Inviscid solution")
        ax_y.plot(vy_ana, y_ref, color="black", lw=FIT_WIDTH, ls="--",
                  zorder=10, label="Inviscid solution")
 
    for ax, comp in [(ax_x, "x"), (ax_y, "y")]:
        ax.set_xlabel(rf"$v_{comp}$ (m/s)", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"$y$ (m)", fontsize=LABEL_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        apply_style(ax)
 
    fig_x.tight_layout()
    fig_y.tight_layout()
    plt.show()
 
def cmd_viscosity_dx(args):
    """
    Taylor–Green vortex:
    Plot extracted numerical viscosity versus grid spacing dx.
    """

    dx_vals   = []
    nu_u_vals = []
    nu_v_vals = []
    nu_e_vals = []

    for nx in NX_LIST:
        json_path = ROOT_SIM / f"tgv_nx_{nx}.json"
        csv_path  = ROOT_OUT / f"nx_{nx}" / "metrics_SL_GL.csv"

        if not json_path.exists() or not csv_path.exists():
            print(f"nx={nx}: missing files, skipping.")
            continue

        run = load_run(json_path, csv_path)

        times = run["times"]
        sim_u = run["sim_u"]
        sim_v = run["sim_v"]

        nu_u, _, _, _ = fit_numerical_viscosity(times, sim_u, u0_guess=sim_u[0])
        nu_v, _, _, _ = fit_numerical_viscosity(times, sim_v, u0_guess=sim_v[0])

        # --- energy fit ---
        times_e = run["times"][1:]
        energy  = run["energy"][1:]
        E_0     = energy[0]
        cut     = int(0.1 * len(times_e))
        times_e = times_e[cut:]
        energy  = energy[cut:] / E_0
        nu_e, _, _, _ = fit_numerical_viscosity(times_e, energy, u0_guess=energy[0])

        dx = run["dx"]

        dx_vals.append(dx)
        nu_u_vals.append(abs(nu_u))
        nu_v_vals.append(abs(nu_v))
        nu_e_vals.append(abs(nu_e) / 2)

        print(f"nx={nx:4d} | dx={dx:.4e} | nu_x={nu_u:.4e} | nu_y={nu_v:.4e} | nu_e={nu_e/2:.4e}")

    dx_vals   = np.array(dx_vals)
    nu_u_vals = np.array(nu_u_vals)
    nu_v_vals = np.array(nu_v_vals)
    nu_e_vals = np.array(nu_e_vals)

    p_u = np.polyfit(np.log(dx_vals), np.log(nu_u_vals), 1)[0]
    p_v = np.polyfit(np.log(dx_vals), np.log(nu_v_vals), 1)[0]
    p_e = np.polyfit(np.log(dx_vals), np.log(nu_e_vals), 1)[0]

    print(f"\nObserved convergence orders:")
    print(f"  nu_x ~ dx^{p_u:.3f}")
    print(f"  nu_y ~ dx^{p_v:.3f}")
    print(f"  nu_e ~ dx^{p_e:.3f}")

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.loglog(dx_vals, nu_u_vals, "o-",  lw=FIT_WIDTH, ms=8,
              color=_PALETTE[0],
              label=rf"$\nu_x \sim \Delta x^{{{p_u:.2f}}}$")
    ax.loglog(dx_vals, nu_v_vals, "s--", lw=FIT_WIDTH, ms=8,
              color=_PALETTE[2],
              label=rf"$\nu_y \sim \Delta x^{{{p_v:.2f}}}$")
    ax.loglog(dx_vals, nu_e_vals, "^:",  lw=FIT_WIDTH, ms=8,
              color=_PALETTE[1],
              label=rf"$\nu_e \sim \Delta x^{{{p_e:.2f}}}$")

    dx_ref = np.linspace(dx_vals.min(), dx_vals.max(), 200)
    ref1   = nu_u_vals[0] * (dx_ref / dx_vals[0]) ** 1
    ref2   = nu_u_vals[0] * (dx_ref / dx_vals[0]) ** 2

    ax.loglog(dx_ref, ref1, ":", color="gray", lw=1.8, label="order 1")
    ax.loglog(dx_ref, ref2, "--", color="gray", lw=1.8, label="order 2")

    ax.set_xlabel(r"$\Delta x$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\nu_{\mathrm{num}}$", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig.tight_layout()

    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_OUT / f"visc_num_dx_{SOLVER}.pdf", format="pdf")
    print(f"Saved: {ROOT_OUT / f'visc_num_dx_{SOLVER}.pdf'}")
    plt.show()


def cmd_viscosity_dt(args):
    """
    Taylor–Green vortex:
    Plot extracted numerical viscosity versus time step dt.
    """

    dt_vals   = []
    nu_u_vals = []
    nu_v_vals = []
    nu_e_vals = []

    for nt in NT_LIST:
        if nt == 250 or nt == 500:
            continue  # skip the coarsest run which has too few points for a reliable fit
        json_path = ROOT_SIM / f"tgv_nt_{nt}.json"
        csv_path  = ROOT_OUT / f"nt_{nt}" / "metrics_SL_GL.csv"

        if not json_path.exists() or not csv_path.exists():
            print(f"nt={nt}: missing files, skipping.")
            continue

        run = load_run(json_path, csv_path)

        times = run["times"]
        sim_u = run["sim_u"]
        sim_v = run["sim_v"]

        nu_u, _, _, _ = fit_numerical_viscosity(times, sim_u, u0_guess=sim_u[0])
        nu_v, _, _, _ = fit_numerical_viscosity(times, sim_v, u0_guess=sim_v[0])

        # --- energy fit ---
        times_e = run["times"][1:]
        energy  = run["energy"][1:]
        E_0     = energy[0]
        cut     = int(0.1 * len(times_e))
        times_e = times_e[cut:]
        energy  = energy[cut:] / E_0
        nu_e, _, _, _ = fit_numerical_viscosity(times_e, energy, u0_guess=energy[0])

        dt = run["dt"]

        dt_vals.append(dt)
        nu_u_vals.append(abs(nu_u))
        nu_v_vals.append(abs(nu_v))
        nu_e_vals.append(abs(nu_e) / 2)

        print(f"nt={nt:4d} | dt={dt:.4e} | nu_x={nu_u:.4e} | nu_y={nu_v:.4e} | nu_e={nu_e/2:.4e}")

    dt_vals   = np.array(dt_vals)
    nu_u_vals = np.array(nu_u_vals)
    nu_v_vals = np.array(nu_v_vals)
    nu_e_vals = np.array(nu_e_vals)

    p_u = np.polyfit(np.log(dt_vals), np.log(nu_u_vals), 1)[0]
    p_v = np.polyfit(np.log(dt_vals), np.log(nu_v_vals), 1)[0]
    p_e = np.polyfit(np.log(dt_vals), np.log(nu_e_vals), 1)[0]

    print(f"\nObserved convergence orders:")
    print(f"  nu_x ~ dt^{p_u:.3f}")
    print(f"  nu_y ~ dt^{p_v:.3f}")
    print(f"  nu_e ~ dt^{p_e:.3f}")

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.loglog(dt_vals, nu_u_vals, "o-",  lw=FIT_WIDTH, ms=8,
              color=_PALETTE[0],
              label=rf"$\nu_x \sim \Delta t^{{{p_u:.2f}}}$")
    ax.loglog(dt_vals, nu_v_vals, "s--", lw=FIT_WIDTH, ms=8,
              color=_PALETTE[2],
              label=rf"$\nu_y \sim \Delta t^{{{p_v:.2f}}}$")
    ax.loglog(dt_vals, nu_e_vals, "^:",  lw=FIT_WIDTH, ms=8,
              color=_PALETTE[1],
              label=rf"$\nu_e \sim \Delta t^{{{p_e:.2f}}}$")

    dt_ref = np.linspace(dt_vals.min(), dt_vals.max(), 200)
    ref_1  = nu_u_vals[0] * (dt_ref / dt_vals[0]) ** -1
    ref1   = nu_u_vals[0] * (dt_ref / dt_vals[0]) ** 1
    ref2   = nu_u_vals[0] * (dt_ref / dt_vals[0]) ** 2

    ax.loglog(dt_ref, ref_1, ":", color="gray", lw=1.8, label="order -1")
    ax.loglog(dt_ref, ref1,  ":", color="gray", lw=1.8, label="order 1")
    ax.loglog(dt_ref, ref2,  "--", color="gray", lw=1.8, label="order 2")

    ax.set_xlabel(r"$\Delta t$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\nu_{\mathrm{num}}$", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig.tight_layout()

    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_OUT / f"visc_num_dt_{SOLVER}.pdf", format="pdf")
    print(f"Saved: {ROOT_OUT / f'visc_num_dt_{SOLVER}.pdf'}")
    plt.show()

def cmd_energy(args):
    """
    Taylor–Green vortex:
    Plot vx(t) and vy(t) for ALL resolutions on the SAME figures,
    including exponential decay fits and extracted numerical viscosity.
    """

    styles = _resolution_styles(NX_LIST)

    fig, ax = plt.subplots(figsize=(9, 7))

    for nx in NX_LIST:

        json_path = ROOT_SIM / f"tgv_nx_{nx}.json"
        csv_path  = ROOT_OUT / f"nx_{nx}" / "metrics_SL_GL.csv"

        if not json_path.exists() or not csv_path.exists():
            print(f"nx={nx}: missing files, skipping.")
            continue

        run = load_run(json_path, csv_path)

        with open(json_path) as f:
            data = json.load(f)

        times  = run["times"][1:]
        energy = run["energy"][1:]

        E_0 = energy[0]

        cut = 0.1 * len(times)
        times = times[int(cut):]
        energy = energy[int(cut):]

        energy = energy / E_0

        nu_e, _, u0_e, _ = fit_numerical_viscosity(times, energy, u0_guess=energy[0])
        fit_e = decaying_exponential(times, u0_e, nu_e)

        color = styles[nx]["color"]

        ax.plot(times, energy, color=color, lw=LINE_WIDTH, label=rf"$n_x={nx}$")
        ax.plot(times, fit_e,  color=color, lw=FIT_WIDTH,  ls="--")

        print(f"nx={nx:4d} | nu_num(energy)={nu_e/2:.4e}")

    ax.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel(r"$E(t) / E(0)$", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    apply_style(ax)
    fig.tight_layout()

    # ✅ Ensure output directory exists
    ROOT_OUT.mkdir(parents=True, exist_ok=True)

    # ✅ Save figures
    fig.savefig(ROOT_OUT / f"energy_{SOLVER}.pdf", format="pdf")

    print(f"Saved: {ROOT_OUT / f'energy_{SOLVER}.pdf'}")
    plt.show()
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
 
    parser = argparse.ArgumentParser(description="Taylor–Green vortex validation")
    sub = parser.add_subparsers(dest="mode", required=True)
 
    run_p = sub.add_parser("run")
    run_p.add_argument("binary", type=pathlib.Path)
    run_p.add_argument("-i", "--input", required=True, type=pathlib.Path)
 
    sub.add_parser("plot",
        help="Plot vx/vy decay with exponential fit for all resolutions")
 
    slice_p = sub.add_parser("slice")
    slice_p.add_argument("--csv", required=True, type=pathlib.Path)
    slice_p.add_argument("-i", "--input", required=True, type=pathlib.Path)
 
    conv_p = sub.add_parser("convergence",
        help="Probe velocity over time for all resolutions")
 
    compare_p = sub.add_parser("compare",
        help="Compare vx slice across all resolutions")
 
    viscosity_dx_p = sub.add_parser("viscosity_dx",
        help="Plot extracted numerical viscosity vs dx convergence")

    viscosity_dt_p = sub.add_parser("viscosity_dt",
        help="Plot extracted numerical viscosity vs dt convergence")
    
    energy_p = sub.add_parser("energy",
        help="Plot kinetic energy decay with exponential fit for all resolutions")

    args = parser.parse_args()
 
    if args.mode == "run":
        folder = build_folder(args)
        launch_sims(args, folder)
    elif args.mode == "plot":
        cmd_plot(args)
    elif args.mode == "convergence":
        cmd_convergence(args)
    elif args.mode == "compare":
        cmd_compare(args)
    elif args.mode == "viscosity_dx":
        cmd_viscosity_dx(args)
    elif args.mode == "viscosity_dt":
        cmd_viscosity_dt(args)
    elif args.mode == "energy":
        cmd_energy(args)
 
if __name__ == "__main__":
    main()