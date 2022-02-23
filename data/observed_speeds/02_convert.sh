#!/bin/bash


xmin=-638000
ymin=-3349600
xmax=864700
ymax=-657600
GRID=1800


rm -rf greenland_vel_mosaic250_*_v1_g${GRID}m*.nc 
for var in vx vy; do
    gdalwarp -overwrite  -r average -s_srs EPSG:3413 -t_srs EPSG:3413 -te $xmin $ymin $xmax $ymax -tr $GRID $GRID greenland_vel_mosaic250_${var}_v1.tif greenland_vel_mosaic250_${var}_v1_g${GRID}m_i.nc
    cdo setattribute,$var@units="m year-'" -chname,Band1,$var greenland_vel_mosaic250_${var}_v1_g${GRID}m_i.nc greenland_vel_mosaic250_${var}_v1_g${GRID}m.nc
done
cdo expr,"velsurf_mag=sqrt(vx^2+vy^2)" -merge greenland_vel_mosaic250_*_v1_g${GRID}m.nc greenland_vel_mosaic250_v1_g${GRID}m.nc
