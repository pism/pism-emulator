# Copyright (C) 2026 Andy Aschwanden
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
Evaluate a multi-glacier emulator ensemble, for all glaciers and per glacier.

The emulators are compared with the PISM ensemble they were trained on (there is
no held-out split: training uses bootstrap weights over all members). Statistics
are area-weighted, in physical units (m/yr) and in the transformed (training)
space, for every emulator and for the ensemble mean of the emulators.

Outputs (in ``--out-dir``, default ``EMULATOR_DIR/evaluation``):

* ``stats_by_glacier.csv``: one row per emulator, glacier (or ``all``) and space
* ``stats_overall.csv``: the ``all`` rows of the above
* ``stats_by_member.csv``: ensemble-mean RMSE and bias per member and glacier
* ``speed_emulator_<RGI id>.pdf``: PISM, emulator and relative error for a few
  members, with their parameters and fit
* ``scatter.png``: emulator vs PISM, coloured by glacier
"""

from __future__ import annotations

from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
import pylab as plt
import torch
from cmglaciology import cm as cmg

from pism_emulator.datasets import YTransformName, inverse_y_transform_np
from pism_emulator.emulators.speed.train import EMULATORS
from pism_emulator.glaciers import (
    DEFAULT_STATE_GLOB,
    GLACIER_INDEX_FILE,
    GlacierIndex,
    MultiGlacierDataset,
    discover_glaciers,
)
from pism_emulator.utils import param_keys_dict

#: Matplotlib style of the evaluation figures (as in ``pism-evaluate-emulator``).
RCPARAMS = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "hatch.linewidth": 0.15,
    "font.size": 6,
}

#: Dataset settings read from the checkpoint when not given on the command line.
SETTINGS = (
    "glacier_dir",
    "target_template",
    "glaciers",
    "state_glob",
    "training_var",
    "target_var",
    "target_error_var",
    "target_mask_var",
    "y_lim",
    "y_transform",
)


def weighted_stats(
    pred: np.ndarray, true: np.ndarray, w: np.ndarray
) -> dict[str, float]:
    """
    Area-weighted error statistics over members and nodes.

    Parameters
    ----------
    pred, true : numpy.ndarray
        Predictions and reference, shape ``(n_members, n_nodes)``.
    w : numpy.ndarray
        Node weights, shape ``(n_nodes,)``; normalized here.

    Returns
    -------
    dict[str, float]
        ``rmse``, ``mae``, ``bias`` (prediction minus reference), ``r2``,
        ``pearson_r`` and ``median_rel_err`` (unweighted median of
        ``|pred - true| / |true|``).
    """
    w = np.broadcast_to(w / w.sum(), true.shape) / true.shape[0]
    d = pred - true
    mean_t = float((w * true).sum())
    mean_p = float((w * pred).sum())
    ss_res = float((w * d**2).sum())
    ss_tot = float((w * (true - mean_t) ** 2).sum())
    cov = float((w * (pred - mean_p) * (true - mean_t)).sum())
    var_p = float((w * (pred - mean_p) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(d) / np.abs(true)
    return {
        "rmse": float(np.sqrt(ss_res)),
        "mae": float((w * np.abs(d)).sum()),
        "bias": float((w * d).sum()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": (
            cov / np.sqrt(var_p * ss_tot) if var_p * ss_tot > 0 else float("nan")
        ),
        "median_rel_err": (
            float(np.nanmedian(rel[np.isfinite(rel)]))
            if np.isfinite(rel).any()
            else float("nan")
        ),
    }


def stats_table(
    preds: dict[str, np.ndarray],
    true: np.ndarray,
    area: np.ndarray,
    index: GlacierIndex,
    space: str,
) -> pd.DataFrame:
    """
    Compute statistics for every emulator, for all glaciers and per glacier.

    Parameters
    ----------
    preds : dict[str, numpy.ndarray]
        ``{emulator name: predictions (n_members, n_nodes)}``.
    true : numpy.ndarray
        PISM responses, shape ``(n_members, n_nodes)``.
    area : numpy.ndarray
        Node weights, shape ``(n_nodes,)``.
    index : GlacierIndex
        Layout of the node vector.
    space : str
        Label of the units, e.g. ``"m/yr"``.

    Returns
    -------
    pandas.DataFrame
        One row per emulator and glacier (``"all"`` = every glacier).
    """
    rows = []
    for name, p in preds.items():
        rows.append(
            {
                "emulator": name,
                "glacier": "all",
                "space": space,
                **weighted_stats(p, true, area),
            }
        )
        for b in index.blocks:
            s = b.nodes
            rows.append(
                {
                    "emulator": name,
                    "glacier": b.rgi_id,
                    "space": space,
                    **weighted_stats(p[:, s], true[:, s], area[s]),
                }
            )
    return pd.DataFrame(rows)


def member_table(
    pred: np.ndarray,
    true: np.ndarray,
    area: np.ndarray,
    index: GlacierIndex,
    ids: Sequence[int],
) -> pd.DataFrame:
    """
    RMSE and bias of one prediction per member and glacier.

    Parameters
    ----------
    pred, true : numpy.ndarray
        Shape ``(n_members, n_nodes)``, physical units.
    area : numpy.ndarray
        Node weights.
    index : GlacierIndex
        Layout of the node vector.
    ids : sequence of int
        Member ids of the rows.

    Returns
    -------
    pandas.DataFrame
        Columns ``member``, ``glacier``, ``rmse``, ``bias`` and ``in_training``.
    """
    used = set(index.member_ids)
    rows = []
    for r, mid in enumerate(ids):
        for b in index.blocks:
            s = b.nodes
            st = weighted_stats(pred[r : r + 1, s], true[r : r + 1, s], area[s])
            rows.append(
                {
                    "member": int(mid),
                    "glacier": b.rgi_id,
                    "rmse": st["rmse"],
                    "bias": st["bias"],
                    "in_training": int(mid) in used,
                }
            )
    return pd.DataFrame(rows)


def _member_text(
    keys: Sequence[str], x: np.ndarray, st: dict[str, float]
) -> tuple[str, str]:
    """
    Text blocks with a member's parameters and its fit.

    Parameters
    ----------
    keys : sequence of str
        Parameter names.
    x : numpy.ndarray
        Parameter values (physical units).
    st : dict[str, float]
        Statistics from :func:`weighted_stats`.

    Returns
    -------
    tuple[str, str]
        Parameter lines and fit lines.
    """
    params = "\n".join(f"{param_keys_dict.get(k, k)}: {v:.3f}" for k, v in zip(keys, x))
    fit = (
        f"MAE = {st['mae']:.1f} m/yr\nMBE = {st['bias']:.1f} m/yr\n"
        f"RMSE = {st['rmse']:.0f} m/yr\nr = {st['pearson_r']:.3f}"
    )
    return params, fit


def plot_maps(
    index: GlacierIndex,
    true: np.ndarray,
    pred: np.ndarray,
    X: np.ndarray,
    keys: Sequence[str],
    ids: Sequence[int],
    area: np.ndarray,
    out_dir: Path,
    n_members: int = 4,
    seed: int = 2021,
) -> list[Path]:
    """
    Per glacier: PISM, emulator and relative error for a few members.

    The layout follows ``pism-evaluate-emulator``: one column per member with
    rows *True*, *Predicted* and *Error* (relative, ±20 %), the member's
    parameters and its fit on this glacier below, and the score over all members
    of this glacier in the footer. Speeds use ``cmg.speed`` on a scale from 1
    to 1000 m/yr.

    Parameters
    ----------
    index : GlacierIndex
        Layout of the node vector.
    true, pred : numpy.ndarray
        PISM and emulator speeds, shape ``(n_members, n_nodes)``.
    X : numpy.ndarray
        Parameters of the members in physical units, shape ``(n_members, n_params)``.
    keys : sequence of str
        Parameter names.
    ids : sequence of int
        Member ids of the rows.
    area : numpy.ndarray
        Node weights.
    out_dir : pathlib.Path
        Output directory.
    n_members : int, optional
        Number of members (columns) to show.
    seed : int, optional
        Seed of the random choice of members.

    Returns
    -------
    list[pathlib.Path]
        The written files.
    """
    rng = np.random.default_rng(seed)
    k = min(n_members, len(ids))
    rows = sorted(rng.choice(len(ids), size=k, replace=False).tolist())
    cmap = cmg.cmaps["speed"]
    written = []
    with plt.rc_context(RCPARAMS):
        for b in index.blocks:
            s = b.nodes
            fig = plt.figure(figsize=(6.4, 4.8), layout="constrained")
            outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 0.4])
            gs_top = outer[0].subgridspec(
                3,
                k + 2,
                width_ratios=[0.12, *([1] * k), 0.06],
                wspace=0.025,
                hspace=0.025,
            )
            axs = np.array(
                [
                    [fig.add_subplot(gs_top[r, c + 1]) for c in range(k)]
                    for r in range(3)
                ]
            )
            for r, title in enumerate(["True", "Predicted", "Error"]):
                lab = fig.add_subplot(gs_top[r, 0])
                lab.axis("off")
                lab.text(
                    1.0,
                    0.5,
                    title,
                    transform=lab.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
            gs_bot = outer[1].subgridspec(
                1, k + 1, width_ratios=[0.12, *([1] * k)], wspace=0.025
            )
            axs_bot = [fig.add_subplot(gs_bot[0, c + 1]) for c in range(k)]

            im_log = im_rel = None
            for c, r in enumerate(rows):
                t2 = index.to_2d(true[r, s], b.rgi_id)
                p2 = index.to_2d(pred[r, s], b.rgi_id)
                im_log = axs[0, c].imshow(
                    t2, origin="lower", cmap=cmap, vmin=1e0, vmax=1e3
                )
                axs[1, c].imshow(p2, origin="lower", cmap=cmap, vmin=1e0, vmax=1e3)
                im_rel = axs[2, c].imshow(
                    (p2 - t2) / t2, origin="lower", vmin=-0.2, vmax=0.2, cmap="coolwarm"
                )
                axs[0, c].set_title(f"member {ids[r]}", fontsize=7)
                params, fit = _member_text(
                    keys,
                    X[r],
                    weighted_stats(pred[r : r + 1, s], true[r : r + 1, s], area[s]),
                )
                axs_bot[c].text(
                    0.2, 0.75, params, c="k", transform=axs_bot[c].transAxes
                )
                axs_bot[c].text(0.2, 0.2, fit, c="k", transform=axs_bot[c].transAxes)
            for ax in [*axs.ravel(), *axs_bot]:
                ax.set_axis_off()
            if im_log is not None and im_rel is not None:
                cb1 = fig.colorbar(
                    im_log, cax=fig.add_subplot(gs_top[0:2, k + 1]), extend="both"
                )
                cb1.set_label("m/yr (log scale)")
                cb2 = fig.colorbar(
                    im_rel, cax=fig.add_subplot(gs_top[2, k + 1]), extend="both"
                )
                cb2.set_label("Relative error")
            st = weighted_stats(pred[:, s], true[:, s], area[s])
            fig.suptitle(b.rgi_id, fontsize=8)
            fig.supxlabel(
                f"Mean Score ({len(ids)} members): MAE={st['mae']:.2f} m/yr, "
                f"MBE={st['bias']:.2f} m/yr, RMSE={st['rmse']:.0f} m/yr, "
                f"Pearson r={st['pearson_r']:.2f}, r²={st['r2']:.2f}",
                y=0.01,
                fontsize=6,
            )
            path = out_dir / f"speed_emulator_{b.rgi_id}.pdf"
            fig.savefig(path)
            plt.close(fig)
            written.append(path)
    return written


def plot_scatter(
    index: GlacierIndex,
    true: np.ndarray,
    pred: np.ndarray,
    out: Path,
    max_points: int = 20000,
) -> None:
    """
    Emulator vs PISM speed, coloured by glacier.

    Parameters
    ----------
    index : GlacierIndex
        Layout of the node vector.
    true, pred : numpy.ndarray
        Shape ``(n_members, n_nodes)``, physical units.
    out : pathlib.Path
        Output file.
    max_points : int, optional
        Points drawn per glacier at most.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for b in index.blocks:
        t, p = true[:, b.nodes].ravel(), pred[:, b.nodes].ravel()
        k = rng.choice(t.size, size=min(max_points, t.size), replace=False)
        ax.scatter(t[k], p[k], s=1, alpha=0.3, label=b.rgi_id, rasterized=True)
    lo = max(float(np.nanmin(true)), 1e-1)
    hi = float(np.nanmax(true))
    ax.plot([lo, hi], [lo, hi], "k", lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PISM speed (m/yr)")
    ax.set_ylabel("Emulator speed (m/yr)")
    ax.legend(fontsize=5, markerscale=5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _parser() -> ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--emulator-dir", required=True)
    parser.add_argument(
        "--out-dir", default=None, help="Default: EMULATOR_DIR/evaluation"
    )
    parser.add_argument("--n-plot-members", type=int, default=4)
    parser.add_argument("--glacier-dir", default=None)
    parser.add_argument("--target-template", default=None)
    parser.add_argument("--glaciers", nargs="+", default=None)
    parser.add_argument("--state-glob", default=None)
    parser.add_argument("--training-var", default=None)
    parser.add_argument("--target-var", default=None)
    parser.add_argument("--target-error-var", default=None)
    parser.add_argument("--target-mask-var", default=None)
    parser.add_argument("--y-lim", type=float, nargs=2, default=None)
    parser.add_argument("--y-transform", default=None)
    parser.add_argument(
        "EMULATOR_FILES",
        nargs="*",
        help="Checkpoints (default: EMULATOR_DIR/emulator/emulator_*.ckpt). Dataset "
        "options default to the values the first checkpoint was trained with.",
    )
    return parser


def _load_dataset(
    cfg: dict[str, Any], emulator_dir: Path
) -> tuple[MultiGlacierDataset, GlacierIndex]:
    """
    Rebuild the training data and check it against the saved glacier index.

    Parameters
    ----------
    cfg : dict
        Dataset settings (see :data:`SETTINGS`).
    emulator_dir : pathlib.Path
        Emulator directory, possibly holding ``glacier_index.nc``.

    Returns
    -------
    tuple[MultiGlacierDataset, GlacierIndex]
        The dataset, and its index with the members used in training.

    Raises
    ------
    ValueError
        If the glaciers or their nodes differ from the saved index.
    """
    mask_var = cfg["target_mask_var"] or "landice"
    y_lim = tuple(cfg["y_lim"] or (1.0, 100e3))
    y_transform = cast(YTransformName, cfg["y_transform"] or "log10")

    dataset = MultiGlacierDataset(
        discover_glaciers(
            cfg["glacier_dir"],
            cfg["target_template"],
            state_glob=cfg["state_glob"] or DEFAULT_STATE_GLOB,
            glaciers=cfg["glaciers"],
        ),
        training_var=cfg["training_var"] or "velsurf_mag",
        target_var=cfg["target_var"] or "v",
        target_error_var=cfg["target_error_var"] or "v_error",
        target_mask_var=None if mask_var.lower() == "none" else mask_var,
        y_lim=y_lim,
        y_transform=y_transform,
    )
    index = dataset.index
    saved_path = emulator_dir / GLACIER_INDEX_FILE
    if saved_path.is_file():
        saved = GlacierIndex.load(saved_path)
        if not saved.same_layout(index):
            raise ValueError(
                f"The glaciers or their nodes differ from {saved_path} (glaciers "
                f"{saved.rgi_ids} vs {index.rgi_ids}); the emulator cannot be evaluated "
                "on this data"
            )
        index = GlacierIndex(index.blocks, saved.member_ids)  # members used in training

    return dataset, index


def _predict(files: Sequence[str], X: torch.Tensor) -> dict[str, np.ndarray]:
    """
    Predict with every checkpoint, plus their mean when there are several.

    Parameters
    ----------
    files : sequence of str
        Emulator checkpoints.
    X : torch.Tensor
        Normalized parameters, shape ``(n_members, n_parameters)``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Predictions in the transformed space, by checkpoint name and
        ``"ensemble_mean"``.
    """
    preds_t: dict[str, np.ndarray] = {}
    for f in files:
        h = torch.load(f, map_location="cpu", weights_only=False)["hyper_parameters"]
        model = EMULATORS[h.get("emulator", "DNN")].load_from_checkpoint(
            f, map_location="cpu"
        )
        model.eval()
        with torch.no_grad():
            preds_t[Path(f).stem] = model(X, add_mean=True).numpy()
    if len(preds_t) > 1:
        preds_t["ensemble_mean"] = np.mean(list(preds_t.values()), axis=0)
    return preds_t


def run(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Evaluate a multi-glacier emulator ensemble and return its statistics.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command line arguments (default: ``sys.argv``).

    Returns
    -------
    pandas.DataFrame
        The per-glacier statistics table.
    """
    parser = _parser()
    args = parser.parse_args(argv)

    emulator_dir = Path(args.emulator_dir)
    files = args.EMULATOR_FILES or sorted(
        glob(str(emulator_dir / "emulator" / "emulator_*.ckpt"))
    )
    if not files:
        parser.error(f"No checkpoints found in {emulator_dir / 'emulator'}")
    hp = torch.load(files[0], map_location="cpu", weights_only=False)[
        "hyper_parameters"
    ]
    cfg = {
        k: getattr(args, k) if getattr(args, k) is not None else hp.get(k)
        for k in SETTINGS
    }
    if not cfg["glacier_dir"] or not cfg["target_template"]:
        parser.error(
            "Not a multi-glacier emulator: give --glacier-dir and --target-template"
        )
    dataset, index = _load_dataset(cfg, emulator_dir)
    y_lim, y_transform = dataset.y_lim, dataset.y_transform
    X = dataset.samples.X
    Y_t = dataset.samples.Y.numpy()
    area = dataset.samples.normed_area.numpy()
    ids = dataset.samples.ids

    def phys(a: np.ndarray) -> np.ndarray:
        return inverse_y_transform_np(
            a, name=y_transform, y_lim=y_lim, params=dataset.y_params
        )

    preds_t = _predict(files, X)
    best = "ensemble_mean" if "ensemble_mean" in preds_t else next(iter(preds_t))

    true_p = phys(Y_t)
    preds_p = {k: phys(v) for k, v in preds_t.items()}
    table = pd.concat(
        [
            stats_table(preds_p, true_p, area, index, "m/yr"),
            stats_table(preds_t, Y_t, area, index, y_transform),
        ],
        ignore_index=True,
    )

    out_dir = Path(args.out_dir) if args.out_dir else emulator_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "stats_by_glacier.csv", index=False)
    table[table.glacier == "all"].to_csv(out_dir / "stats_overall.csv", index=False)
    member_table(preds_p[best], true_p, area, index, ids).to_csv(
        out_dir / "stats_by_member.csv", index=False
    )

    X_raw = (X * dataset.samples.X_std + dataset.samples.X_mean).numpy()
    plot_maps(
        index,
        true_p,
        preds_p[best],
        X_raw,
        dataset.samples.X_keys,
        ids,
        area,
        out_dir,
        n_members=args.n_plot_members,
    )
    plot_scatter(index, true_p, preds_p[best], out_dir / "scatter.png")

    show = table[(table.emulator == best) & (table.space == "m/yr")]
    cols = ["glacier", "rmse", "mae", "bias", "r2", "pearson_r", "median_rel_err"]
    print(f"\n{best}, {len(ids)} members, m/yr (in-sample):")
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nWrote {out_dir}")
    return table


def main(argv: Sequence[str] | None = None) -> None:
    """
    Command-line entry point of ``pism-evaluate-emulator-glaciers``.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command line arguments (default: ``sys.argv``).
    """
    run(argv)


if __name__ == "__main__":
    main()
