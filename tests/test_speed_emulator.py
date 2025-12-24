# Copyright (C) 2015 Andy Aschwanden
#
# This file is part of pism-emulator.
#
# PISM-EMULATOR is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-EMULATOR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Test speed emulator and sampler workflow.
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """
    Return the repository root directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the repository root (one level above ``tests/``).
    """
    return Path(__file__).resolve().parents[1]


def _require_inputs(repo: Path) -> tuple[Path, Path, list[Path]]:
    """
    Locate input files required by the speed-emulator tests.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root directory (as returned by :func:`_repo_root`).

    Returns
    -------
    samples_file : pathlib.Path
        Path to the samples CSV used for training.
    target_file : pathlib.Path
        Path to the observed/target velocity dataset.
    training_files : list[pathlib.Path]
        List of training files discovered under the legacy training directory.
    """

    samples_file = repo / "data/samples/velocity_calibration_samples_100.csv"
    target_file = repo / "legacy/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc"
    training_files = sorted(
        Path(p)
        for p in glob(str(repo / "legacy/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc"))
    )

    missing = [p for p in (samples_file, target_file) if not p.exists()]
    if missing:
        pytest.skip(f"Missing required input file(s): {[str(p) for p in missing]}")
    if not training_files:
        pytest.skip(
            "No training files matched legacy/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc"
        )

    return samples_file, target_file, training_files


@pytest.mark.integration
def test_speed_emulator_pipeline_via_mains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Smoke test: train then evaluate using the provided scripts' ``main()``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory used as an output location.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to temporarily set environment variables and isolate
        global process state during the test.
    """
    repo = _repo_root()
    samples_file, target_file, training_files = _require_inputs(repo)

    # Import inside the test so we can skip cleanly if optional deps aren't available.
    # pylint: disable=import-outside-toplevel
    try:
        from pism_emulator.emulators.speed.train import main as train_main
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Could not import train entrypoint: {exc}")

    try:
        from pism_emulator.sampler.mala_speed import main as sample_main
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Could not import mala_speed entrypoint: {exc}")

    try:
        from pism_emulator.emulators.speed.evaluate import main as evaluate_main
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Could not import evaluate entrypoint: {exc}")
    # pylint: enable=import-outside-toplevel

    out_dir = tmp_path / "emulator_test_log10"
    ckpts: list[Path] = []

    for model_index in (0, 1):
        train_argv = [
            "train-emulator",
            "--emulator",
            "DNN",
            "--max-epochs",
            "1",
            "--emulator-dir",
            str(out_dir),
            "--model-index",
            str(model_index),
            "--y-transform",
            "log10",
            "--y-lim",
            "1",
            "1e5",
            "--samples-file",
            str(samples_file),
            "--target-file",
            str(target_file),
            *[str(p) for p in training_files],
        ]
        monkeypatch.setattr(sys, "argv", train_argv)
        train_main()

        ckpt_matches = sorted(out_dir.glob(f"emulator/*_{model_index}.ckpt"))
        assert ckpt_matches, f"No ckpt produced for model_index={model_index}"
        ckpt = ckpt_matches[0]
        ckpts.append(ckpt)

        sample_argv = [
            "sample-posterior-speed",
            "--emulator-dir",
            str(out_dir),
            "--model-index",
            str(model_index),
            "--samples-file",
            str(samples_file),
            "--target-file",
            str(target_file),
            "--y-transform",
            "log10",
            "--y-lim",
            "1",
            "1e5",
            "--chains",
            "1",
            "--burn",
            "5",
            "--samples",
            "10",
            *[str(p) for p in training_files],
            str(ckpt),
        ]

        monkeypatch.setattr(sys, "argv", sample_argv)
        sample_main()
        posterior_nc = out_dir / "posterior" / f"X_posterior_model_{model_index}.nc"
        assert posterior_nc.exists(), f"Missing posterior NetCDF: {posterior_nc}"

    eval_argv = [
        "evaluate-emulator",
        "--emulator-dir",
        str(out_dir),
        "--samples-file",
        str(samples_file),
        "--target-file",
        str(target_file),
        "--training-files",
        *[str(p) for p in training_files],
        "--y-transform",
        "log10",
        "--y-lim",
        "1",
        "1e5",
        *[str(p) for p in ckpts],
    ]
    monkeypatch.setattr(sys, "argv", eval_argv)
    evaluate_main()

    pdf = out_dir / "train" / "speed_emulator_train.pdf"
    assert pdf.exists(), f"Missing evaluation PDF: {pdf}"
