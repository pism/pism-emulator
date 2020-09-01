#!/bin/bash
#
# This script merges individual files into one CSV file and should be only used for legacy
# reasons. Not needed with the most current commit.

rcp=$1
kernel=$2

mkdir -p loo/${kernel}

for year in {2009..2100}; do
    cat gp-loo/${kernel}/gp_kernel_${kernel}_rcp_${rcp}_${year}_loo_*.csv | sort -r | uniq > loo/${kernel}/gp_kernel_${kernel}_rcp_${rcp}_${year}_loo.csv
done
