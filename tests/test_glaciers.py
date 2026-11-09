"""Tests for training and evaluating one emulator on several glaciers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from pism_emulator.emulators.speed import evaluate_glaciers, train
from pism_emulator.emulators.speed.evaluate_glaciers import weighted_stats
from pism_emulator.glaciers import (
    GLACIER_INDEX_FILE,
    GlacierIndex,
    MultiGlacierDataset,
    discover_glaciers,
)

TEMPLATE = "{root}/obs/{{rgi_id}}/obs_{{rgi_id}}.nc"
PARAMS = pd.DataFrame(
    {"uq": range(6), "a": np.linspace(0.1, 0.6, 6), "b": np.linspace(1.0, 3.5, 6)}
)


def _speed(member: int, glacier: int) -> float:
    """
    Model speed of a member on a glacier, unique per pair.

    Parameters
    ----------
    member : int
        Member id.
    glacier : int
        Glacier number.

    Returns
    -------
    float
        Speed in m/yr.
    """
    return 10.0 * (member + 1) + 100.0 * glacier


def _make_glacier(
    root: Path,
    rgi_id: str,
    number: int,
    shape: tuple[int, int],
    dx: float,
    members: list[int],
    params: pd.DataFrame = PARAMS,
) -> int:
    """
    Write one synthetic glacier: ensemble output, uq.csv and observations.

    The ice covers the lower-left half of the grid; the observed speed is also
    defined off the ice.

    Parameters
    ----------
    root : pathlib.Path
        Base directory.
    rgi_id : str
        Glacier id.
    number : int
        Glacier number, used to make speeds unique.
    shape : tuple[int, int]
        Model grid ``(ny, nx)``.
    dx : float
        Model grid spacing.
    members : list[int]
        Members with output on this glacier.
    params : pandas.DataFrame, optional
        Ensemble parameters written to ``uq.csv``.

    Returns
    -------
    int
        Number of ice cells on the model grid.
    """
    ny, nx = shape
    x = dx / 2 + dx * np.arange(nx) + 1e6 * number
    y = dx / 2 + dx * np.arange(ny)
    out = root / "calib" / rgi_id / "output"
    (out / "state").mkdir(parents=True)
    params.to_csv(out / "uq.csv", index=False)
    for m in members:
        v = np.full((1, ny, nx), _speed(m, number), dtype=np.float32)
        xr.Dataset(
            {"velsurf_mag": (("time", "y", "x"), v)},
            coords={"time": [0.0], "y": y, "x": x},
        ).to_netcdf(out / "state" / f"state_g500m_{rgi_id}_id_0_uq_{m}_0001-01-01.nc")

    ice = np.zeros((ny, nx), dtype=np.int8)
    ice[: ny // 2 + 1, : nx // 2 + 1] = 1
    obs = root / "obs" / rgi_id
    obs.mkdir(parents=True)
    xr.Dataset(
        {
            "v": (("y", "x"), np.full((ny, nx), 50.0 + number, dtype=np.float32)),
            "vx": (("y", "x"), np.full((ny, nx), 3.0, dtype=np.float32)),
            "vy": (("y", "x"), np.full((ny, nx), 4.0, dtype=np.float32)),
            "vx_error": (("y", "x"), np.full((ny, nx), 1.0, dtype=np.float32)),
            "vy_error": (("y", "x"), np.full((ny, nx), 2.0, dtype=np.float32)),
            "landice": (("y", "x"), ice),
        },
        coords={"y": y, "x": x},
    ).to_netcdf(obs / f"obs_{rgi_id}.nc")
    return int(ice.sum())


@pytest.fixture(name="glaciers")
def fixture_glaciers(tmp_path: Path) -> dict:
    """
    Three glaciers: A lacks member 2, B lacks member 4 (and uses 1 km cells), C has no output.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    dict
        Paths and expected ice cell counts.
    """
    n_a = _make_glacier(tmp_path, "RGI-A", 0, (4, 6), 500.0, [0, 1, 3, 4, 5])
    n_b = _make_glacier(tmp_path, "RGI-B", 1, (5, 3), 1000.0, [0, 1, 2, 3, 5])
    _make_glacier(tmp_path, "RGI-C", 2, (3, 3), 500.0, [])
    return {
        "calib": tmp_path / "calib",
        "template": TEMPLATE.format(root=tmp_path),
        "n_a": n_a,
        "n_b": n_b,
    }


def _dataset(glaciers: dict) -> MultiGlacierDataset:
    specs = discover_glaciers(glaciers["calib"], glaciers["template"])
    return MultiGlacierDataset(specs, y_lim=(1.0, 1e5), parallel=False)


def test_discover_skips_glaciers_without_output(glaciers: dict) -> None:
    """
    Skip glaciers without state files; raise for a requested missing glacier.

    Parameters
    ----------
    glaciers : dict
        Synthetic glaciers.
    """
    specs = discover_glaciers(glaciers["calib"], glaciers["template"])
    assert [s.rgi_id for s in specs] == ["RGI-A", "RGI-B"]
    assert len(specs[0].training_files) == 5
    only_b = discover_glaciers(
        glaciers["calib"], glaciers["template"], glaciers=["RGI-B"]
    )
    assert [s.rgi_id for s in only_b] == ["RGI-B"]
    with pytest.raises(FileNotFoundError):
        discover_glaciers(glaciers["calib"], glaciers["template"], glaciers=["RGI-X"])


def test_members_matched_and_dropped_everywhere(glaciers: dict) -> None:
    """
    Drop members missing on any glacier and line rows up by member id.

    Parameters
    ----------
    glaciers : dict
        Synthetic glaciers.
    """
    ds = _dataset(glaciers)
    n_a, n_b = glaciers["n_a"], glaciers["n_b"]
    assert ds.samples.ids == [0, 1, 3, 5]
    assert ds.samples.Y.shape == (4, n_a + n_b)
    assert ds.index.n_nodes == n_a + n_b
    speed = 10 ** ds.samples.Y.numpy()
    parts = ds.index.split(speed)
    for row, m in enumerate(ds.samples.ids):
        np.testing.assert_allclose(parts["RGI-A"][row], _speed(m, 0), rtol=1e-5)
        np.testing.assert_allclose(parts["RGI-B"][row], _speed(m, 1), rtol=1e-5)
    # parameters of the kept members, normalized over the kept members
    raw = PARAMS.set_index("uq").loc[[0, 1, 3, 5]].to_numpy(np.float32)
    np.testing.assert_allclose(ds.samples.X_mean.numpy(), raw.mean(0), rtol=1e-6)
    table = ds.member_table()
    assert not table.loc[2, "RGI-A"]
    assert not table.loc[4, "RGI-B"]
    assert table["used"].tolist() == [True, True, False, True, False, True]


def test_area_weights_and_derived_error(glaciers: dict) -> None:
    """
    Weight nodes by cell area and derive the speed error from vx/vy errors.

    Parameters
    ----------
    glaciers : dict
        Synthetic glaciers.
    """
    ds = _dataset(glaciers)
    w = ds.index.split(ds.samples.normed_area)
    assert np.isclose(ds.samples.normed_area.sum(), 1.0)
    np.testing.assert_allclose(w["RGI-B"][0] / w["RGI-A"][0], 4.0, rtol=1e-5)
    # sqrt((3*1)^2 + (4*2)^2) / 5
    assert ds.Y_target_error is not None
    np.testing.assert_allclose(
        ds.Y_target_error.numpy(), np.sqrt(73.0) / 5.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        10 ** ds.index.split(ds.Y_target)["RGI-B"], 51.0, rtol=1e-5
    )


def test_index_roundtrip_and_maps(glaciers: dict, tmp_path: Path) -> None:
    """
    The index survives save/load and maps nodes back onto each grid.

    Parameters
    ----------
    glaciers : dict
        Synthetic glaciers.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = _dataset(glaciers)
    ds.index.save(tmp_path / "index.nc")
    back = GlacierIndex.load(tmp_path / "index.nc")
    assert back.same_layout(ds.index)
    assert back.member_ids == [0, 1, 3, 5]
    assert back["RGI-A"].available_ids == [0, 1, 3, 4, 5]
    np.testing.assert_array_equal(back["RGI-B"].x, ds.index["RGI-B"].x)
    m = back.to_2d(np.arange(back.n_nodes, dtype=float), "RGI-B")
    assert m.shape == (5, 3)
    assert m.count() == glaciers["n_b"]
    assert m.compressed().min() == glaciers["n_a"]  # B's nodes follow A's
    assert m.mask[-1, -1]  # ice-free corner


def test_parameter_mismatch_raises(tmp_path: Path) -> None:
    """
    Require glaciers to agree on the parameters of a member.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    _make_glacier(tmp_path, "RGI-A", 0, (4, 4), 500.0, [0, 1, 2])
    other = PARAMS.copy()
    other.loc[1, "a"] += 1.0
    _make_glacier(tmp_path, "RGI-B", 1, (4, 4), 500.0, [0, 1, 2], params=other)
    specs = discover_glaciers(tmp_path / "calib", TEMPLATE.format(root=tmp_path))
    with pytest.raises(ValueError, match="parameters differ"):
        MultiGlacierDataset(specs, parallel=False)


def test_train_and_evaluate_cli(glaciers: dict, tmp_path: Path, monkeypatch) -> None:
    """
    Run pism-train-emulator --glacier-dir, then pism-evaluate-emulator-glaciers.

    Parameters
    ----------
    glaciers : dict
        Synthetic glaciers.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture.
    """
    monkeypatch.chdir(tmp_path)
    emu = tmp_path / "emu"
    for model_index in (0, 1):
        argv = [
            "pism-train-emulator",
            "--emulator-dir", str(emu),
            "--glacier-dir", str(glaciers["calib"]),
            "--target-template", glaciers["template"],
            "--target-var", "v",
            "--y-lim", "1", "1e5",
            "--max-epochs", "2",
            "--accelerator", "cpu",
            "--num-workers", "0",
            "--model-index", str(model_index),
        ]  # fmt: skip
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
    assert (emu / GLACIER_INDEX_FILE).is_file()
    assert (emu / "eigenglaciers" / "eigenglaciers_0_RGI-B.pdf").is_file()
    assert pd.read_csv(emu / "members.csv")["used"].sum() == 4

    table = evaluate_glaciers.run(["--emulator-dir", str(emu), "--n-plot-members", "2"])
    out = emu / "evaluation"
    for name in ("stats_by_glacier.csv", "stats_overall.csv", "stats_by_member.csv"):
        assert (out / name).is_file()
    assert (out / "maps_RGI-A.pdf").is_file() and (out / "scatter.png").is_file()
    assert set(table.emulator) == {"emulator_0", "emulator_1", "ensemble_mean"}
    assert set(table.glacier) == {"all", "RGI-A", "RGI-B"}
    by_member = pd.read_csv(out / "stats_by_member.csv")
    assert sorted(by_member.member.unique()) == [0, 1, 3, 5]
    assert torch.isfinite(torch.tensor(table[["rmse", "bias"]].to_numpy())).all()


def test_weighted_stats_known_values() -> None:
    """Area-weighted statistics on a hand-checked example."""
    true = np.array([[1.0, 3.0]])
    pred = np.array([[2.0, 3.0]])
    st = weighted_stats(pred, true, np.array([3.0, 1.0]))
    assert st["bias"] == pytest.approx(0.75)
    assert st["rmse"] == pytest.approx(np.sqrt(0.75))
    assert st["mae"] == pytest.approx(0.75)
    assert st["median_rel_err"] == pytest.approx(0.5)
