# Copyright (C) 2019-25 Andy Aschwanden
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
Stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression

ArrayLike: TypeAlias = npt.ArrayLike


def _as_df(
    X: np.ndarray | pd.DataFrame, names: Optional[Sequence[str]]
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if names is None:
        names = [f"X{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(names))


def _all_interactions(main: Sequence[str]) -> List[Tuple[str, str]]:
    # unique i<j
    m = list(main)
    return [(m[i], m[j]) for i in range(len(m)) for j in range(i + 1, len(m))]


def _make_matrix(
    Xdf: pd.DataFrame,
    mains: Sequence[str],
    inters: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    # Build design in a stable, readable column order:
    # [main effects..., interaction effects...]
    cols = []
    for name in mains:
        cols.append(Xdf[name])

    for a, b in inters:
        col = Xdf[a] * Xdf[b]
        col.name = f"{a}*{b}"
        cols.append(col)

    if not cols:
        # Intercept-only model is out of scope for scikit-learn LinearRegression
        # but we’ll fall back to an empty matrix and let the estimator handle it.
        return pd.DataFrame(index=Xdf.index)

    return pd.concat(cols, axis=1)


def _consistent_feature_list(
    mains: Sequence[str], inters: Sequence[Tuple[str, str]]
) -> List[str]:
    out = list(mains)
    out += [f"{a}*{b}" for (a, b) in inters]
    return out


@dataclass
class Step:
    step: int
    action: str  # 'add' or 'remove' or 'init'
    term: str | None
    bic: float
    features: List[str]


def calc_bic(
    model: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for any fitted scikit-learn model.

    Automatically detects model type (regression vs classification) and
    computes BIC using the appropriate likelihood formulation.

    Parameters
    ----------
    model : fitted scikit-learn estimator
        The trained model (must implement `predict`, and optionally `predict_proba`).
    X : array-like of shape (n_samples, n_features)
        Input features.
    Y : array-like of shape (n_samples,)
        True target values.

    Returns
    -------
    float
        Bayesian Information Criterion (smaller is better).

    Notes
    -----
    - For regression models, assumes Gaussian errors:
        BIC = n * ln(RSS/n) + k * ln(n)
    - For classifiers with predict_proba(), assumes Bernoulli or multinomial likelihood.
    - The number of parameters `k` is estimated from model coefficients if available.
    - If parameter count cannot be inferred, a warning is printed and a heuristic is used.
    """

    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y = np.asarray(Y).ravel()
    n = X.shape[0]

    # --- Determine model type ---
    is_classifier = hasattr(model, "predict_proba")

    # --- Estimate number of parameters (k) ---
    if hasattr(model, "coef_"):
        # correct
        coef = np.asarray(model.coef_)
        k = coef.size
        if getattr(model, "fit_intercept", False):
            # number of intercepts: 1 for single-target, n_targets for multi-output
            k += np.size(getattr(model, "intercept_", np.array([0])))
    else:
        # Fallback heuristic: assume one parameter per feature + intercept
        k = X.shape[1] + 1
        print("[calc_bic] Warning: parameter count (k) inferred heuristically.")

    # --- Compute log-likelihood depending on model type ---
    if not is_classifier:
        # Regression case (Gaussian likelihood)
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        RSS = np.maximum(RSS, np.finfo(float).eps)
        bic = n * np.log(RSS / n) + k * np.log(n)
    else:
        # Classification case (Bernoulli or multinomial likelihood)
        proba = model.predict_proba(X)
        proba = np.clip(proba, 1e-12, 1 - 1e-12)  # numerical stability

        if proba.shape[1] == 2:  # binary
            log_likelihood = np.sum(
                Y * np.log(proba[:, 1]) + (1 - Y) * np.log(proba[:, 0])
            )
        else:  # multiclass
            log_likelihood = np.sum(np.log(proba[np.arange(n), Y]))

        bic = -2 * log_likelihood + k * np.log(n)

    return float(bic)


def stepwise_bic(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.Series | pd.DataFrame,
    *,
    varnames: Optional[Sequence[str]] = None,
    estimator: Optional[BaseEstimator] = None,
    direction: str = "both",  # 'forward', 'backward', or 'both'
    interactions: bool = True,  # first-order interactions
    start: str = "main",  # 'main' (all mains, no interactions) or 'empty'
    tol: float = 0.0,  # require improvement < -tol to accept
    max_steps: Optional[int] = None,
    verbose: bool = False,
    calc_bic_fn=None,  # pass your calc_bic function or bind it below
) -> tuple[list[str], BaseEstimator, pd.DataFrame, list[Step]]:
    """
    Stepwise model selection using Bayesian Information Criterion (BIC).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or DataFrame
        Input features.
    Y : array-like of shape (n_samples,) or (n_samples, 1)
        Target values.
    varnames : sequence of str, optional
        Names for columns of `X`. If `X` is a DataFrame, its columns are used.
    estimator : sklearn BaseEstimator, optional
        Estimator to fit at each step. Defaults to `LinearRegression()` if None.
    direction : {'forward', 'backward', 'both'}, default 'both'
        Search strategy.
    interactions : bool, default True
        If True, consider first-order interaction terms (hierarchical: only add
        `A*B` if both `A` and `B` are currently included).
    start : {'main', 'empty'}, default 'main'
        Starting model. 'main' = all main effects; 'empty' = no effects.
    tol : float, default 0.0
        Minimal BIC improvement threshold to accept a change. A change is accepted
        if `bic_candidate < bic_current - tol`.
    max_steps : int, optional
        Optional cap on the number of accepted steps.
    verbose : bool, default False
        Print progress.
    calc_bic_fn : callable
        Function with signature `calc_bic_fn(fitted_model, X, Y) -> float`.
        You should pass **your** `calc_bic` here.

    Returns
    -------
    selected : list of str
        Selected terms in order: mains followed by interactions (e.g., ["X0", "X1", "X0*X1"]).
    fitted_model : BaseEstimator
        Estimator refit on the final selected design.
    X_final : pd.DataFrame
        Final design matrix used to compute BIC (columns in the same order as `selected`).
    history : list of Step
        Step-by-step log of actions and BIC values.

    Notes
    -----
    - Interactions are first-order only (pairwise products).
    - Hierarchical principle enforced for adds: an interaction can be added only
      when both parents are in the model. Removing a main effect removes all its
      interactions.
    - At each candidate evaluation we **clone** and **refit** the estimator.
    """
    if calc_bic_fn is None:
        raise ValueError("Please pass your `calc_bic` via `calc_bic_fn=`.")

    if estimator is None:
        estimator = LinearRegression()

    # Normalize inputs
    Xdf = _as_df(X, varnames)
    y = np.asarray(Y).ravel()

    all_mains = list(Xdf.columns)
    all_inters = _all_interactions(all_mains) if interactions else []

    # Initialize selected sets
    if start == "main":
        sel_mains: List[str] = list(all_mains)
        sel_inters: List[Tuple[str, str]] = []
    elif start == "empty":
        sel_mains = []
        sel_inters = []
    else:
        raise ValueError("start must be 'main' or 'empty'")

    # Build initial design, fit, BIC
    X_curr = _make_matrix(Xdf, sel_mains, sel_inters)
    model_curr = clone(estimator)
    model_curr.fit(X_curr, y)
    bic_curr = float(calc_bic_fn(model_curr, X_curr, y))
    history: List[Step] = [
        Step(
            step=0,
            action="init",
            term=None,
            bic=bic_curr,
            features=_consistent_feature_list(sel_mains, sel_inters),
        )
    ]
    if verbose:
        print(f"[init] BIC={bic_curr:.4f}  features={history[-1].features}")

    step_count = 0
    while True:
        if (max_steps is not None) and (step_count >= max_steps):
            break

        candidates: list[Tuple[str, str, Any]] = []  # (action, term, payload)

        # ----- generate candidates -----
        if direction in ("backward", "both"):
            # Remove any main; removing a main implicitly drops interactions containing it
            for m in sel_mains:
                candidates.append(("remove_main", m, None))
            # Remove any currently selected interaction
            for a, b in sel_inters:
                candidates.append(("remove_inter", f"{a}*{b}", (a, b)))

        if direction in ("forward", "both"):
            # Add a main (not already in)
            for m in all_mains:
                if m not in sel_mains:
                    candidates.append(("add_main", m, None))
            # Add an interaction if both parents already selected and not already present
            if interactions:
                existing = set(sel_inters)
                for a, b in all_inters:
                    if (
                        (a in sel_mains)
                        and (b in sel_mains)
                        and ((a, b) not in existing)
                    ):
                        candidates.append(("add_inter", f"{a}*{b}", (a, b)))

        if not candidates:
            break

        # ----- evaluate candidates -----
        best_bic = np.inf
        best_action = None
        best_term = None
        best_sets = None
        best_X = None
        best_model = None

        for action, term, payload in candidates:
            cand_mains = list(sel_mains)
            cand_inters = list(sel_inters)

            if action == "add_main":
                cand_mains.append(term)
                cand_mains = list(dict.fromkeys(cand_mains))  # uniq
            elif action == "remove_main":
                cand_mains = [m for m in cand_mains if m != term]
                # remove any interactions involving this main
                cand_inters = [
                    (a, b) for (a, b) in cand_inters if (a != term and b != term)
                ]
            elif action == "add_inter":
                a, b = payload
                # hierarchy already ensured when generating
                cand_inters = list(cand_inters) + [(a, b)]
            elif action == "remove_inter":
                a, b = payload
                cand_inters = [
                    (ai, bi) for (ai, bi) in cand_inters if not (ai == a and bi == b)
                ]
            else:
                continue

            X_cand = _make_matrix(Xdf, cand_mains, cand_inters)
            model = clone(estimator)
            model.fit(X_cand, y)
            bic = float(calc_bic_fn(model, X_cand, y))

            if bic < best_bic:
                best_bic = bic
                best_action = action
                best_term = term
                best_sets = (cand_mains, cand_inters)
                best_X = X_cand
                best_model = model

        # ----- stopping rule -----
        if best_bic < bic_curr - tol:
            # accept move
            # mypy: entering this branch implies a best candidate was found
            assert best_sets is not None
            assert best_action is not None
            assert best_term is not None
            assert best_X is not None
            assert best_model is not None
            sel_mains, sel_inters = best_sets
            X_curr = best_X
            model_curr = best_model
            bic_curr = best_bic
            step_count += 1

            pretty_term = best_term
            if best_action == "add_inter":
                pretty_term = best_term  # already "A*B"
            elif best_action == "remove_inter":
                pretty_term = best_term

            hist_features = _consistent_feature_list(sel_mains, sel_inters)
            history.append(
                Step(
                    step=step_count,
                    action="add" if best_action.startswith("add") else "remove",
                    term=pretty_term,
                    bic=bic_curr,
                    features=hist_features,
                )
            )
            if verbose:
                print(
                    f"[step {step_count:02d}] {history[-1].action:6s} {pretty_term:>10s}  "
                    f"BIC={bic_curr:.4f}  k={X_curr.shape[1]}"
                )
        else:
            # no improvement
            break

    selected = _consistent_feature_list(sel_mains, sel_inters)
    return selected, model_curr, X_curr, history


def gelman_rubin(p: npt.ArrayLike, q: npt.ArrayLike) -> float:
    r"""
    Compute the Gelman–Rubin convergence diagnostic (:math:`\hat{R}`) for two chains.

    The Gelman–Rubin diagnostic (also called the potential scale reduction factor)
    tests for lack of convergence by comparing variance **between** chains to the
    variance **within** chains. When both chains have converged to the same target
    distribution, the within-chain and between-chain variance components should
    be similar, and :math:`\hat{R}` approaches 1.

    Parameters
    ----------
    p : array_like
        First Markov chain trace of a scalar parameter with shape ``(n_samples,)``.
    q : array_like
        Second Markov chain trace of a scalar parameter with shape ``(n_samples,)``.

    Returns
    -------
    float
        Potential scale reduction factor :math:`\hat{R}`.

    Raises
    ------
    ValueError
        If the two input chains have different lengths or contain fewer than two
        samples.

    Notes
    -----
    This implementation follows the two-chain version of the classic statistic:

    .. math::

        \hat{R} = \sqrt{\frac{\hat{V}}{W}}

    where :math:`W` is the within-chain variance (here computed from the two chains)
    and :math:`\hat{V}` is the pooled posterior variance estimate.

    This simplified implementation assumes exactly two chains and a scalar parameter.

    References
    ----------
    Gelman, A. and Rubin, D. B. (1992).
    Brooks, S. P. and Gelman, A. (1998).

    Examples
    --------
    >>> p = np.random.default_rng(0).normal(size=1000)
    >>> q = np.random.default_rng(1).normal(size=1000)
    >>> gelman_rubin(p, q)  # doctest: +ELLIPSIS
    1.0...
    """
    p_arr = np.asarray(p, dtype=float).ravel()
    q_arr = np.asarray(q, dtype=float).ravel()

    if p_arr.shape[0] != q_arr.shape[0]:
        raise ValueError("Chains p and q must have the same length")
    n = int(p_arr.shape[0])
    if n < 2:
        raise ValueError("Chains must contain at least two samples")

    # Within-chain variance component (sum of chain variances)
    W = p_arr.std(ddof=1) ** 2 + q_arr.std(ddof=1) ** 2
    if W == 0.0:
        # Both chains are constant; treat as perfectly converged.
        return 1.0

    P_mean = float(p_arr.mean())
    Q_mean = float(q_arr.mean())
    mean = (P_mean + Q_mean) / 2.0

    # Between-chain variance component
    B = n * ((P_mean - mean) ** 2 + (Q_mean - mean) ** 2)

    # Pooled posterior variance estimate
    V_hat = (1.0 - 1.0 / n) * W + (1.0 / n) * B

    R_hat_sq = V_hat / W
    return float(np.sqrt(R_hat_sq))


def kl_divergence(p: ArrayLike, q: ArrayLike) -> float:
    r"""
    Compute the Kullback–Leibler (KL) divergence :math:`D_{KL}(P \,\|\, Q)`.

    The KL divergence measures how one probability distribution ``p`` diverges
    from a second distribution ``q``. It is commonly interpreted as the expected
    log difference between the probabilities under ``p`` and ``q``:

    .. math::

        D_{KL}(P \,\|\, Q) = \sum_i p_i \log\left(\frac{p_i}{q_i}\right)

    Parameters
    ----------
    p : numpy.ndarray
        Discrete probability distribution :math:`P`. Must be broadcast-compatible
        with ``q``. Values should be non-negative; typically ``p.sum() == 1``.
    q : numpy.ndarray
        Discrete probability distribution :math:`Q`. Must be broadcast-compatible
        with ``p``. Values should be non-negative; typically ``q.sum() == 1``.

    Returns
    -------
    float
        The KL divergence :math:`D_{KL}(P \,\|\, Q)` computed with natural logarithms.

    Notes
    -----
    This implementation follows the standard discrete definition but treats terms
    with ``p == 0`` or ``q == 0`` (or non-finite ``p/q``) as contributing zero.
    If you prefer stricter behavior (e.g., returning ``inf`` when ``q_i == 0`` and
    ``p_i > 0``), adjust the masking logic accordingly.

    References
    ----------
    Wikipedia: Kullback–Leibler divergence.

    Examples
    --------
    >>> p = np.array([0.5, 0.5])
    >>> q = np.array([0.9, 0.1])
    >>> kl_divergence(p, q)  # doctest: +ELLIPSIS
    0.5108...
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    ratio = p_arr / q_arr
    mask = (p_arr != 0) & (q_arr != 0) & np.isfinite(ratio)
    return float(np.sum(np.where(mask, p_arr * np.log(ratio), 0.0)))
