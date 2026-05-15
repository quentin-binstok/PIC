import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class Infos:
    def __init__(
        self,
        xlabel,
        ylabel,
        xlog: bool = False,
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlog = xlog
        self.nb_plots = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 7))

    def plot(self, xvals, yvals, label: Optional[str] = None):
        if label:
            self.ax.plot(xvals, yvals, "o", label=label, linewidth=1)
        else:
            self.ax.plot(xvals, yvals, "o", linewidth=1)

        self.nb_plots += 1

    def make_plot(self, name="plot"):
        self.ax.tick_params(axis="both", which="major", labelsize=18)
        self.ax.tick_params(axis="both", which="minor", labelsize=18)

        self.ax.set_xlabel(self.xlabel, fontsize=26)
        self.ax.set_ylabel(self.ylabel, fontsize=26)

        if self.xlog:
            self.ax.set_xscale("log")

        if self.nb_plots > 1:
            plt.legend(fontsize=26)
        plt.tight_layout()

        plt.savefig(name + ".pdf")


def main():
    wave_front_fct_nx = Infos(
        r"Number of cells along the $x$ axis",
        r"Dimensionless wavefront speed",
        # xlog=True,
    )

    wave_front_fct_nx.plot(
        np.array([53, 196, 268, 339, 482, 625, 768, 911, 1054, 1197]),
        [
            0.7989,
            1.3541,
            1.3825,
            1.4021,
            1.4089,
            1.4145,
            1.4163,
            1.4124,
            1.4143,
            1.4214,
        ],
        "PIC/FLIP",
    )
    wave_front_fct_nx.plot(
        np.array([196, 268, 482, 625, 768, 911, 1054, 1197]),
        [1.4243, 1.4310, 1.3934, 1.4330, 1.3970, 1.4315, 1.4003, 1.4321],
        "APIC",
    )
    wave_front_fct_nx.make_plot("wave_front_fct_nx")

    wave_front_fct_nt = Infos(
        r"Number of time steps",
        r"Dimensionless wavefront speed",
    )

    wave_front_fct_nt.plot(
        [1000, 2199, 3015, 5665, 7765, 10000, 12462, 17082, 20000],
        [1.3728, 1.4075, 1.4038, 1.4009, 1.3927, 1.3825, 1.3723, 1.3523, 1.3336],
        "PIC/FLIP",
    )

    wave_front_fct_nt.plot(
        [1000, 2199, 3015, 5665, 7765, 10000, 12462, 17082, 20000],
        [1.3858, 1.4356, 1.4310, 1.4298, 1.4294, 1.4310, 1.4302, 1.3499, 1.4352],
        "APIC",
    )

    wave_front_fct_nt.make_plot("wave_front_fct_nt")

    wave_front_exp = np.array(
        [1.56, 1.34, 1.48, 1.69, 1.54, 1.7, 1.74, 1.21, 1.14, 1.3]
    )

    print(
        f"Wavefront speed, exp: mean = {np.mean(wave_front_exp)}, std = {np.std(wave_front_exp)}, range = [{np.mean(wave_front_exp) - 2 * np.std(wave_front_exp)} ; {np.mean(wave_front_exp) + 2 * np.std(wave_front_exp)}]"
    )

    impact_pressure_nx = Infos(
        r"Number of cells along the $x$ axis", r"Impact pressure $[\text{Pa}]$"
    )

    impact_pressure_nx.plot(
        [196, 268, 339, 482, 625, 768, 911, 1054, 1197],
        [
            15447.3311,
            6545.8726,
            5897.7573,
            5741.9360,
            5909.6714,
            5590.2705,
            5581.9321,
            5679.9824,
            5488.0151,
        ],
        "PIC/FLIP",
    )

    impact_pressure_nx.plot(
        [196, 268, 482, 625, 768, 911, 1054, 1197],
        [
            27765.4414,
            8138.8047,
            6109.1196,
            5949.3008,
            31752.0469,
            5538.7793,
            23828.0137,
            5446.9736,
        ],
        "APIC",
    )

    impact_pressure_nx.make_plot("impact_pressure_nx")

    impact_pressure_nt = Infos(
        r"Number of time steps", r"Impact pressure $[\text{Pa}]$"
    )

    impact_pressure_nt.plot(
        [1000, 2199, 3015, 5665, 7765, 10000, 12462, 17082, 20000],
        [
            6433.3218,
            6296.0591,
            6288.9053,
            6313.9966,
            6496.1699,
            6545.8726,
            11955.0918,
            33047.9297,
            44203.6953,
        ],
        "PIC/FLIP",
    )

    impact_pressure_nt.plot(
        [1000, 2199, 3015, 5665, 7765, 10000, 12462, 17082, 20000],
        [
            6011.9014,
            22164.4297,
            6710.4897,
            7679.5391,
            7672.9126,
            8138.8047,
            8683.1699,
            10437.3545,
            11383.0586,
        ],
        "APIC",
    )

    impact_pressure_nt.make_plot("impact_pressure_nt")


main()
