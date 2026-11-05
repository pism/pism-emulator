#!/usr/bin/env bash
# Run sample-posterior-speed on the 1800 m Greenland calibration used in README.md.
#
# usage: run_sample.sh CODE_DIR OUT_DIR BURN SAMPLES CHAINS [extra sampler args]
#   CODE_DIR  checkout whose pism_emulator is used (e.g. this repo, or a worktree of main)
#   OUT_DIR   output directory; the run happens there so its ./_preds chain files
#             are kept per run
# PYTHON selects the interpreter (default: python).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
code=$(cd "$1" && pwd); out=$2; burn=$3; samples=$4; chains=$5; shift 5
mkdir -p "$out"; cd "$out"
PYTHONPATH=$code "$PYTHON" -W ignore -c "
import sys, glob
from pism_emulator.mcmc.sample_speed import main
import pism_emulator; print('code from', pism_emulator.__file__, flush=True)
R = '$R'
files = sorted(glob.glob(R + '/data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc'))
main(['--emulator','DNN','--emulator-dir','$PWD','--accelerator','cpu',
      '--samples-file', R + '/data/samples/velocity_calibration_samples_100.csv',
      '--target-file', R + '/legacy/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc',
      '--y-transform','log10','--y-lim','1','1e5',
      '--burn','$burn','--samples','$samples','--chains','$chains', *sys.argv[1:],
      *files, R + '/legacy/emulators_dnn/emulator/emulator_0.ckpt'])
" "$@"
