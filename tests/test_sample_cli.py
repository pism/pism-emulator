"""
End-to-end test of a ``sample-posterior-*`` command on synthetic data.
"""

from pathlib import Path

import pytest
import xarray as xr

from pism_emulator.mcmc.sample_pdd_verify import main


@pytest.mark.slow
def test_pdd_verify_writes_posterior_with_three_chains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Check that three chains (which do not divide the prior draws) are written.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory (pytest fixture).
    monkeypatch : pytest.MonkeyPatch
        Used to run in ``tmp_path``, where the chains are staged in ``_preds``.
    """
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "posterior"
    main(
        [
            "--accelerator",
            "cpu",
            "--chains",
            "3",
            "--burn",
            "5",
            "--samples",
            "10",
            "--result-dir",
            str(out),
        ]
    )
    post = xr.open_dataset(out / "X_posterior_model.nc", group="posterior")
    assert post.sizes["chain"] == 3 and post.sizes["draw"] == 10
    prior = xr.open_dataset(out / "X_posterior_model.nc", group="prior")
    assert prior.sizes["chain"] == 3
