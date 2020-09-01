#!bin/bash

NEOF=5

anom_file=../data/speeds/cdo_speeds_anomalies.nc
eval_file=eval.nc
eof_file=eof.nc

# Calculate EOFs (eigenvectors) and coefficients (eigenvalues)
cdo eof,$NEOF $anom_file $eval_file $eof_file
cdo eofcoeff $eof_file $anom_file eofcoeff_

# Calculate the % variance explained by an EOF
# Code snippet from
# https://code.mpimet.mpg.de/boards/1/topics/5236
cdo fldsum -timvar $anom_file totvar1
for ((i=1; i<=$NEOF; ++i)) ; do
  cdo -seltimestep,${i} $eval_file totvar2
  MODE=`cdo output -mulc,100 -div totvar2 totvar1 | sed  "s- --g"`
  echo "Variance explained by EOF MODE ${i}: $MODE" 
done
