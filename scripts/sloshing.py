import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import json
import argparse
import pathlib
import copy


def build_folder(args):
    """
    Builds the folder of json to launch
    """

    temp_folder = "sloshing_sim"
    pathlib.Path.mkdir(temp_folder)
    temp_folder = pathlib.Path(temp_folder)

    with open(args.input, "r") as f_json:
        base_data = json.load(f_json)

    with open(temp_folder / f"sloshing_base.json", "w") as file:
        json.dump(base_data, file)

    base_dx = base_data["space_steps"]
    base_nx = base_data["grid"][0]
    base_ny = base_data["grid"][1]
    base_nt = base_data["nt"]
    base_dt = base_data["delta_t"]
    base_flip = base_data["flip"]
    base_height = base_data["special"]["height"]
    base_amp = base_data["special"]["amplitude"]

    nx_arr = np.linspace(base_nx / 2, base_nx * 10, 10, dtype=int)
    for nx in nx_arr:
        nx = int(nx)
        ratio = nx / base_nx
        ny = int(base_ny * ratio)
        work_data = copy.deepcopy(base_data)
        work_data["grid"] = [nx, ny]
        work_data["ic_cell"][0]["br"] = [nx - 2, ny - 2]

        dx = base_dx / ratio
        work_data["space_steps"] = dx
        work_data["metrics"][1]["idx"] = [1, int(0.03 / dx) + 1]
        work_data["metrics"][2]["idx"] = [
            1,
            int(0.07 / dx) + 1,
        ]
        work_data["metrics"][3]["idx"] = [
            1,
            int((0.11 / dx)) + 1,
        ]
        work_data["metrics"][4]["idx"] = [
            1,
            int(0.15 / dx) + 1,
        ]
        work_data["metrics"][5]["idx"] = [
            1,
            int(0.19 / dx) + 1,
        ]

        work_data["special"]["height"] = int(base_height * ratio)
        work_data["special"]["amplitude"] = int(base_amp * ratio)

        with open(temp_folder / f"sloshing_nx_{nx}.json", "w") as file:
            json.dump(work_data, file)

    nt_arr = np.linspace(base_nt / 2, base_nt * 10, 10, dtype=int)
    for nt in nt_arr:
        nt = int(nt)
        work_data = copy.deepcopy(base_data)
        work_data["delta_t"] = base_nt * base_dt / nt
        work_data["nt"] = nt
        with open(temp_folder / f"sloshing_nt_{nt}.json", "w") as file:
            json.dump(work_data, file)

    flip_arr = [0.0, 0.95, 0.98, 1.0]
    for flip in flip_arr:
        if flip == base_flip:
            continue
        work_data = copy.deepcopy(base_data)
        work_data["flip"] = flip
        with open(temp_folder / f"sloshing_flip_{flip}.json", "w") as file:
            json.dump(work_data, file)

    return temp_folder


def launch_sims(args, folder):
    from tests import run_batch

    run_batch(folder, args.output, args.binary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", help="The binary file", type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, help="The output directory")
    parser.add_argument(
        "-i",
        "--input",
        help="The base sloshing file",
        required=True,
        type=pathlib.Path,
    )

    args = parser.parse_args()

    pathlib.Path.mkdir(args.output)
    folder = build_folder(args)
    launch_sims(args, folder)

    # with open(args.input) as f_json:
    #     data = json.load(f_json)
    #     dt = data["delta_t"]
    #     nt = data["nt"]
    #     dx = data["space_steps"]
    #     idx = 0
    #     for i in data["metrics"]:
    #         if i["header"] == "depth":
    #             idx = i["idx"]
    #             break

    #     header_depth = "depth_" + str(idx)
    #     times = np.linspace(0, nt * dt, nt, dtype=float)
    #     print(nt)
    #     print(dt)
    #     print(times)

    #     dat = read_csv(args.csv)

    #     plt.plot(times, dat[header_depth], label="sim")
    #     plt.plot(
    #         times,
    #         ritter_h(idx * dx, 1000 * dx, 1000 * dx, times),
    #         label="ritter",
    #     )
    #     plt.legend()
    #     plt.show()


main()
