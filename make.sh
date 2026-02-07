#!/usr/bin/env bash

executable="solver.out"
build_dir="build"
data_dir="data"

rm -r $data_dir/*

if [[ $1 == "clean" || $1 == "clean_build" ]]; then
    rm -r $build_dir 
    rm $executable
fi

if [[ $1 == "clean" ]]; then
    exit
fi

mkdir -p $build_dir 
cd $build_dir 
cmake ..
make -j$(nproc)
cp $executable ..