"""
Tests for the eigenglacier basis and the data loaders in ``pism_emulator.datamodules``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pism_emulator.datamodules import LegacyPISMDataModule, PISMDataModule

N_SAMPLES, N_GRID = 40, 25


def _data(seed: int = 0, rank: int = 4) -> tuple[torch.Tensor, ...]:
    """
    Make responses with a decaying spectrum plus noise, and Dirichlet weights.

    Parameters
    ----------
    seed : int, optional
        Seed of the generator for F and X.
    rank : int, optional
        Number of dominant modes (at most 4).

    Returns
    -------
    tuple of torch.Tensor
        ``(X, F, omegas, omegas_0)``.
    """
    # Everything is seeded and in float64, so the comparisons below do not depend
    # on the global RNG state or on float32 round-off of the BLAS in use.
    g = torch.Generator().manual_seed(seed)
    kw = {"generator": g, "dtype": torch.float64}
    scales = torch.tensor([10.0, 4.0, 2.0, 1.0], dtype=torch.float64)[:rank]
    F = (torch.randn(N_SAMPLES, rank, **kw) * scales) @ torch.randn(rank, N_GRID, **kw)
    F = F + 0.01 * torch.randn(N_SAMPLES, N_GRID, **kw)
    X = torch.randn(N_SAMPLES, 3, **kw)
    omegas = torch.from_numpy(
        np.random.default_rng(seed).dirichlet(np.ones(N_SAMPLES))
    ).unsqueeze(1)
    omegas_0 = torch.full((N_SAMPLES, 1), 1.0 / N_SAMPLES, dtype=torch.float64)
    return X, F, omegas, omegas_0


def _weighted_cov(F: torch.Tensor, omegas: torch.Tensor) -> torch.Tensor:
    F_bar = F - (F * omegas).sum(0)
    return F_bar.T @ torch.diag(omegas.squeeze()) @ F_bar


def test_lowrank_and_eigh_agree() -> None:
    """Low-rank SVD and eigh give the same leading eigenvalues and subspace."""
    X, F, om, om0 = _data()
    q = 4
    torch.manual_seed(0)
    V1, _, _, l1 = PISMDataModule(X, F, om, om0).get_eigenglaciers(
        q=q, svd_lowrank=True
    )
    V2, _, _, l2 = PISMDataModule(X, F, om, om0).get_eigenglaciers(
        q=q, svd_lowrank=False
    )
    assert V1.shape == V2.shape == (N_GRID, q)
    torch.testing.assert_close(l1, l2, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(V1 @ V1.T, V2 @ V2.T, rtol=1e-3, atol=1e-3)


def test_full_basis_reconstructs_weighted_covariance() -> None:
    """With all modes, V_hat V_hat^T is the weighted covariance F_bar^T W F_bar."""
    X, F, om, om0 = _data()
    V_hat, F_bar, F_mean, _ = PISMDataModule(X, F, om, om0).get_eigenglaciers(
        q=N_GRID, svd_lowrank=False
    )
    torch.testing.assert_close(F_mean, (F * om).sum(0))
    torch.testing.assert_close(F_bar, F - F_mean)
    torch.testing.assert_close(
        V_hat @ V_hat.T, _weighted_cov(F, om), rtol=1e-3, atol=1e-2
    )
    assert torch.isfinite(V_hat).all()


@pytest.mark.parametrize("svd_lowrank", [True, False])
def test_cutoff_keeps_enough_modes(svd_lowrank: bool) -> None:
    """
    Check that the basis keeps the fewest modes whose variance reaches cutoff.

    Parameters
    ----------
    svd_lowrank : bool
        Use the low-rank SVD.
    """
    X, F, om, om0 = _data()
    torch.manual_seed(0)
    V_hat, _, _, lam = PISMDataModule(X, F, om, om0).get_eigenglaciers(
        cutoff=0.5, svd_lowrank=svd_lowrank
    )
    frac = torch.cumsum(lam / lam.sum(), 0)
    assert frac[0] > 0.5  # first mode alone explains more than half
    assert V_hat.shape[1] == 1
    V_hat, _, _, lam = PISMDataModule(X, F, om, om0).get_eigenglaciers(
        cutoff=0.999, svd_lowrank=svd_lowrank
    )
    k = V_hat.shape[1]
    frac = torch.cumsum(lam / lam.sum(), 0)
    assert frac[k - 1] >= 0.999 and (k == 1 or frac[k - 2] < 0.999)


def test_cache_is_keyed_by_inputs(tmp_path: Path) -> None:
    """
    Check that a cache file is only reused for the same F, omegas and settings.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory (pytest fixture).
    """
    X, F, om, om0 = _data()
    _, _, om_b, _ = _data(seed=1)
    cache = tmp_path / "eig.pt"
    torch.manual_seed(0)
    a = PISMDataModule(X, F, om, om0)
    Va, *_ = a.get_eigenglaciers(q=4, cache_path=cache)
    b = PISMDataModule(X, F, om_b, om0)
    assert a.eig_cache_key(q=4) != b.eig_cache_key(q=4)
    assert a.eig_cache_key(q=4) != a.eig_cache_key(q=5)
    Vb, _, Fb_mean, _ = b.get_eigenglaciers(q=4, cache_path=cache)
    torch.testing.assert_close(Fb_mean, (F * om_b).sum(0))
    assert not torch.allclose(Vb @ Vb.T, Va @ Va.T)
    # a fresh module with the same inputs as b loads b's file from disk
    c = PISMDataModule(X, F, om_b, om0)
    Vc, *_ = c.get_eigenglaciers(q=4, cache_path=cache)
    torch.testing.assert_close(Vc, Vb)
    assert torch.load(cache)["key"] == c.eig_cache_key(q=4)


def test_prepare_data_without_arguments_keeps_basis() -> None:
    """Check that Lightning's argument-less prepare_data() keeps the basis."""
    X, F, om, om0 = _data()
    dm = PISMDataModule(X, F, om, om0)
    dm.prepare_data(q=6, svd_lowrank=False)
    V_hat = dm.eig.V_hat
    assert V_hat is not None and V_hat.shape[1] == 6
    dm.prepare_data()  # what Trainer.fit(..., datamodule=dm) does
    assert dm.eig.V_hat is V_hat


def test_loaders_work_without_worker_processes() -> None:
    """Check that num_workers=0 (the default, needed on MPS) builds working loaders."""
    X, F, om, om0 = _data()
    dm = PISMDataModule(X, F, om, om0, batch_size=16, num_workers=0)
    dm.prepare_data(q=4)
    dm.setup("fit")
    x, f, _, _ = next(iter(dm.train_dataloader()))
    assert x.shape == (16, 3) and f.shape == (16, N_GRID)
    assert len(list(dm.val_dataloader())) == 3


def test_legacy_datamodule_eigh_path() -> None:
    """Check that LegacyPISMDataModule's eigh path returns the q largest modes."""
    X, F, om, om0 = _data()
    dm = LegacyPISMDataModule(X, F, om, om0)
    V_hat, _, _, lam = dm.get_eigenglaciers(q=4, svd_lowrank=False)
    assert V_hat.shape == (N_GRID, 4)
    assert torch.all(lam[:-1] >= lam[1:])
    ref = torch.linalg.eigvalsh(_weighted_cov(F, om)).flip(0)[:4]
    torch.testing.assert_close(lam, ref, rtol=1e-4, atol=1e-4)
