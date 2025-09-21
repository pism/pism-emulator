#!/bin/bash

mkdir -p emulator
wget -r -np -nH -nc --cut-dirs=4 -R "index.html*" -P emulator \
  https://arcticdata.io/data/10.18739/A2KW57K4R/emulators/
mkdir -p speeds_v2
wget -r -np -nH -nc --cut-dirs=4 -R "index.html*" -P speeds_v2 \
  https://arcticdata.io/data/10.18739/A2KW57K4R/speeds_v2/
