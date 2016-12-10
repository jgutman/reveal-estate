#!/bin/bash
#PBS -l mem=16GB
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=20
#PBS -M jg3862@nyu.edu
#PBS -m ae

source $HOME/miniconda3/bin/activate capstone
cd $SCRATCH/reveal-estate
python3 regression_loop.py --model rf --data data/merged/bronx_brooklyn_manhattan_queens_statenisland_2003_2016.csv --iters 50 > rf_all_50.out
