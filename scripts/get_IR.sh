#!/bin/bash

START=$1
STOP=$2
SCRIPTS="/home/lk/software/MATS-L2-processing/scripts/"

for chn in IR1 IR2 IR3 IR4; do
    python $SCRIPTS/get_data.py --channel $chn --start_time $START --stop_time $STOP --ncdf_out ${chn}.nc;
done

cp IR1.nc IRc.nc
python $SCRIPTS/superpose.py IRc.nc IR2.nc IR3.nc IR4.nc
# rm IR1.nc IR2.nc IR3.nc IR4.nc

