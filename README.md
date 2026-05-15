# MATH0471 - Particle-in-Cell methods for fluid mechanics

This repository contains the code for the class _Multiphysics Integrated Computational Project_ given at the University of Liège. The documentation is available at [Table of contents](docs/toc.md), while instructions to build and run the code are given below.

The main focus of this project is to implement several particle-in-cell methods, and to evaluate their results from a physical standpoint, namely a Couette flow, a von Karman vorted street, a dam break, and other free surface flow cases.

The methods that will be implemented are:

- a semi-Lagrangian method,
- the vanilla PIC method,
- the FLIP method,
- and the APIC method.

A more advanced topic will be explored, which will be adding thermic effects to the simulations.

## Building

To build, just run `./make.sh`. If you need to clean up the directory, run `./make.sh clean`. If you want to make a clean build, `./make.sh clean_build` is available.

## Running

Once you have obtained a binary file, run `./solver.out <json file>` to run the simulation.

Dam break and sloshing tests were realised using some Python automation. Scripts used for this are given in the `scripts` folder. Running the `dam_break.py` and `sloshing.py` scripts will launch the simulations if provided with `tests/free_surface/dam_break.json` and `tests/free_surface/sloshing.json` as input files, respectively. Similar simulation files exist for the APIC case. The other scripts are used to make graphs, except `tests.py`, which provides helper functions to launch batches of tests.
