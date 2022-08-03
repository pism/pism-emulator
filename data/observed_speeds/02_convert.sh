#!/bin/bash


xmin=-678650
ymin=-3371600
xmax=905350
ymax=-635600


for GRID in 9000 1800; do
    rm -rf greenland_vel_mosaic250_*_v1_g${GRID}m*.nc 
    for var in vx vy ex ey; do
        gdalwarp -overwrite  -r average -s_srs EPSG:3413 -t_srs EPSG:3413 -te $xmin $ymin $xmax $ymax -tr $GRID $GRID greenland_vel_mosaic250_${var}_v1.tif greenland_vel_mosaic250_${var}_v1_g${GRID}m_i.nc
        cdo setattribute,$var@units="m year-'" -chname,Band1,$var greenland_vel_mosaic250_${var}_v1_g${GRID}m_i.nc greenland_vel_mosaic250_${var}_v1_g${GRID}m.nc
    done
    cdo expr,"velsurf_mag=sqrt(vx^2+vy^2); velsurf_mag_error=sqrt(ex^2+ey^2);" -merge greenland_vel_mosaic250_*_v1_g${GRID}m.nc greenland_vel_mosaic250_v1_g${GRID}m.nc
done
