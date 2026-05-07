#!/usr/bin/env python3

"""
This script is to handle systematic testing.
It supports:
    - creating a database of reference results
    - testing code against a given database
    - generic batch launching of tests
"""

import argparse
import logging
import pathlib
import subprocess
import json
from pandas import read_csv

# ================
# Utilities
# ================

# Setting up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="test.py.log",
    encoding="utf-8",
    format="%(asctime)s\t%(levelname)s\t%(message)s",
    level=logging.DEBUG,
)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


# Arguments
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "binary", help="The binary file used to run tests", type=pathlib.Path
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="The output directory where files will be stored",
        default="test_results",
    )
    parser.add_argument("--log", help="The log file", default="tests.log")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["batch", "compare"],
        help="The mode in which to run the script",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="The folder containing the test files",
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument(
        "-c",
        "--compare",
        help="The folder containing the test results to compare with",
        type=pathlib.Path,
    )

    args = parser.parse_args()
    return args


# ================
# Functions
# ================


def run_batch(input_dir, output_dir, binary: pathlib.Path):
    logger.info("Starting batch")
    for root, dirs, files in pathlib.Path.walk(input_dir):
        for file in files:
            abs_dir = root / file
            with open(abs_dir, "r") as f:
                data = json.load(f)
                filename = abs_dir.with_suffix("")
                filename = (
                    str(filename)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(".", "")
                    .lstrip("_")
                )
                logger.info(f"\tStarting {filename}")

                data["dir"] = output_dir + "/" + filename
                data["metrics_file"] = filename + ".csv"
                json_file_name = filename + ".json"
                with open(json_file_name, "w") as json_file:
                    json.dump(data, json_file)

                subprocess.run(
                    [str(binary.resolve()), json_file_name], capture_output=True
                )
                pathlib.Path(json_file_name)._delete()
                logger.info(f"\tEnded {filename}")


def run_compare(input_dir, output_dir):
    logging.info("Starting compare")
    for root, dirs, files in pathlib.Path.walk(input_dir):
        for file in files:
            abs_dir = root / file
            with open(abs_dir, "r") as f:
                data = json.load(f)
                filename = abs_dir.with_suffix("")
                filename = (
                    str(filename)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(".", "")
                    .lstrip("_")
                )
                logger.info(f"\tStarting {filename}")

                data["dir"] = output_dir + "/" + filename
                data["metrics_file"] = filename + ".csv"
                metrics_file = pathlib.Path(filename + ".csv")
                json_file_name = filename + ".json"
                with open(json_file_name, "w") as json_file:
                    json.dump(data, json_file)

                subprocess.run(
                    [str(args.binary.resolve()), json_file_name], capture_output=True
                )
                pathlib.Path(json_file_name)._delete()

                df = read_csv(
                    args.output_dir + "/" + filename + "/" + str(metrics_file)
                )
                df_ref = read_csv(
                    str(args.compare) + "/" + filename + "/" + str(metrics_file)
                )
                for key in df.keys():
                    test_val = 0.0
                    ref_val = 0.0
                    diff_val = 0.0

                    for i in range(len(df[key])):
                        test_val += abs(df[key][i])
                        if len(df[key]) == len(df_ref[key]):
                            diff_val += abs(df[key][i] - df_ref[key][i])

                    for i in range(len(df_ref[key])):
                        ref_val += abs(df_ref[key][i])

                    diff_val /= len(df[key])
                    if len(df[key]) != len(df_ref[key]):
                        diff_val = abs(test_val - ref_val)

                    test_val /= len(df[key])
                    ref_val /= len(df_ref[key])

                    logger.info(
                        f"\t\tKEY {key}\tREF {ref_val}\tTEST {test_val}\tDIFF {diff_val}"
                    )

                logger.info(f"\tEnded {filename}")


# ================
# Main logic
# ================


def main():
    args = get_args()

    # logging.basicConfig(
    #     filename=args.log,
    #     encoding="utf-8",
    #     format="%(asctime)s\t%(levelname)s\t%(message)s",
    #     level=logging.DEBUG,
    # )

    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # logger.addHandler(ch)

    logger.info(f"===========================")
    logger.info(f"Running in {args.mode} mode")
    logger.info(f"Input dir: {args.input}")
    logger.info(f"Output dir: {args.output_dir}")
    if args.mode == "compare":
        logger.info(f"Compare dir: {args.compare}")

    if pathlib.Path.exists(args.output_dir):
        logger.error("Output directory already exists.")
        exit(1)

    pathlib.Path.mkdir(args.output_dir)

    if args.mode == "compare" and (
        (args.compare == None or not pathlib.Path.exists(args.compare))
    ):
        logger.error("Compare directory does not exist")
        exit(1)

    # Main part
    if args.mode == "batch":
        run_batch(args.input, args.output_dir)
    elif args.mode == "compare":
        run_compare(args.input, args.ouput_dir)


if __name__ == "__main__":
    main()
