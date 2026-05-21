"""
model_hmm_mapmatch.py
=====================
Stage 5 — HMM map-matching (Newson-Krumm-style) for GLH correction.

A classical, non-learned baseline that does what none of the per-point or
per-sequence ML approaches could: enforce route consistency between
neighbouring corrected positions. Each session's GLH points are matched
to the most likely sequence of road / path edges via Viterbi over a
candidate set, with an emission term penalising large perpendicular
distance and a transition term penalising inconsistent inter-point
projections.

Algorithm (per session)
-----------------------
1. For each GLH point (east, north), find the K nearest network edges
   within max_radius_m. For each candidate edge, compute the perpendicular
   distance d and the projected on-edge position (proj_east, proj_north).

2. Emission log-probability:
        log_emit[t, k] = -d[t, k]² / (2 σ_z²)
   (Gaussian; σ_z ~ GPS accuracy radius, default 10 m.)

3. Transition log-probability between candidate `k_prev` at step t-1 and
   `k_curr` at step t:
        Δ_proj = || proj_t,k_curr - proj_{t-1},k_prev ||
        Δ_obs  = || obs_t        - obs_{t-1}          ||
        log_trans = -|Δ_proj - Δ_obs| / β
   (Laplacian/exponential; β default 50 m. Penalises pairs whose on-edge
   displacement is wildly inconsistent with the observation displacement.)
   Note: this is the simplified "route-ratio proxy" rather than full
   road-graph shortest-path routing — adequate when candidates per point
   are spatially close to the observation.

4. Viterbi DP yields the most likely candidate sequence.

5. Output per timestep: chosen edge id, projected position on that edge,
   perpendicular distance.

Public API
----------
    find_candidates(points_df, network, max_radius_m=100, k=5) → list[Candidate]
    run_session_viterbi(candidates_per_t, obs_xy, sigma_z, beta)
    apply_hmm_to_df(df, network, ...) → df with hmm_pred_* columns added
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .projection import bng_distance, bng_to_wgs84


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Candidate:
    """A single candidate edge for one GLH point."""
    edge_label: object            # label of the matched row in network
    perp_distance_m: float        # perpendicular distance from obs to edge
    proj_east: float              # projected on-edge position (BNG)
    proj_north: float


def find_candidates_per_session(
    session_df: pd.DataFrame,
    network,
    *,
    east_col: str = "glh_east",
    north_col: str = "glh_north",
    max_radius_m: float = 100.0,
    k: int = 5,
) -> list[list[_Candidate]]:
    """
    For each row in `session_df`, return up to `k` nearest candidate edges
    within `max_radius_m` of (east, north). Each candidate carries its
    edge label, perpendicular distance, and on-edge projection.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    # Build / reuse a spatial index on the network
    geoms = network.geometry.values
    sindex = network.sindex

    out: list[list[_Candidate]] = []
    for _, row in session_df.iterrows():
        e = float(row[east_col])
        n = float(row[north_col])
        if not (np.isfinite(e) and np.isfinite(n)):
            out.append([])
            continue
        # Bbox query for performance — pull all edges whose envelope is
        # within max_radius_m of the point, then refine by actual distance.
        env = (e - max_radius_m, n - max_radius_m,
               e + max_radius_m, n + max_radius_m)
        candidate_ix = list(sindex.intersection(env))
        if not candidate_ix:
            out.append([])
            continue

        p = Point(e, n)
        # Compute actual distance for each, drop those outside max_radius
        scored: list[tuple[float, int]] = []
        for ix in candidate_ix:
            d = float(geoms[ix].distance(p))
            if d <= max_radius_m:
                scored.append((d, ix))
        if not scored:
            out.append([])
            continue
        # Keep the k closest
        scored.sort(key=lambda x: x[0])
        scored = scored[:k]

        cands: list[_Candidate] = []
        for d, ix in scored:
            edge_geom = geoms[ix]
            s = edge_geom.project(p)
            snap_pt = edge_geom.interpolate(s)
            label = network.index[ix]
            cands.append(_Candidate(
                edge_label=label,
                perp_distance_m=d,
                proj_east=float(snap_pt.x),
                proj_north=float(snap_pt.y),
            ))
        out.append(cands)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Viterbi
# ─────────────────────────────────────────────────────────────────────────────

def run_session_viterbi(
    candidates_per_t: list[list[_Candidate]],
    obs_xy: np.ndarray,
    *,
    sigma_z: float = 10.0,
    beta: float = 50.0,
) -> tuple[list[Optional[_Candidate]], list[float]]:
    """
    Run Viterbi over a session.

    Parameters
    ----------
    candidates_per_t : list of T elements; each is a list[Candidate] of
                       up to K candidates for that timestep.
    obs_xy : ndarray of shape (T, 2) — observation (east, north) at each t.
    sigma_z : emission std for perpendicular distance (m).
    beta   : transition scale for the route-ratio proxy (m).

    Returns
    -------
    chosen : list of length T — Candidate per timestep (None for empty steps).
    log_total : list of length T — log-prob of the chosen path up to t.
    """
    T = len(candidates_per_t)
    if T == 0:
        return [], []

    # Initialise with the first timestep's emissions only.
    chosen: list[Optional[_Candidate]] = [None] * T
    # log_alpha[t] = list of log-probabilities for each candidate at t.
    # backptr[t] = list of indices into candidates_per_t[t-1].
    log_alpha: list[np.ndarray] = []
    backptr: list[Optional[np.ndarray]] = [None]

    cands0 = candidates_per_t[0]
    if not cands0:
        # No candidates at t=0 — punt and return raw at every step.
        return chosen, [float("-inf")] * T
    d0 = np.array([c.perp_distance_m for c in cands0])
    log_alpha.append(-(d0 ** 2) / (2.0 * sigma_z ** 2))

    for t in range(1, T):
        cands_t = candidates_per_t[t]
        cands_prev = candidates_per_t[t - 1]
        if not cands_t:
            log_alpha.append(np.array([]))
            backptr.append(None)
            continue
        if not cands_prev:
            # Restart: emissions only at this step.
            d = np.array([c.perp_distance_m for c in cands_t])
            log_alpha.append(-(d ** 2) / (2.0 * sigma_z ** 2))
            backptr.append(np.full(len(cands_t), -1))
            continue

        # Compute transition log-probs.
        Δ_obs = float(np.linalg.norm(obs_xy[t] - obs_xy[t - 1]))
        # On-edge displacements between every (k_prev, k_curr).
        prev_xy = np.array([(c.proj_east, c.proj_north) for c in cands_prev])  # (K_prev, 2)
        curr_xy = np.array([(c.proj_east, c.proj_north) for c in cands_t])     # (K_curr, 2)
        # Pairwise distances, shape (K_prev, K_curr)
        diff = prev_xy[:, None, :] - curr_xy[None, :, :]
        Δ_proj = np.linalg.norm(diff, axis=-1)
        log_trans = -np.abs(Δ_proj - Δ_obs) / beta            # (K_prev, K_curr)

        # Emission at t
        d = np.array([c.perp_distance_m for c in cands_t])
        log_emit = -(d ** 2) / (2.0 * sigma_z ** 2)            # (K_curr,)

        # DP update
        # scores[k_prev, k_curr] = log_alpha_prev[k_prev] + log_trans[k_prev, k_curr]
        scores = log_alpha[-1][:, None] + log_trans
        best_prev = np.argmax(scores, axis=0)                  # (K_curr,)
        best_score = scores[best_prev, np.arange(len(cands_t))]  # (K_curr,)
        log_alpha.append(best_score + log_emit)
        backptr.append(best_prev)

    # Backtrack
    log_total: list[float] = []
    # Find best last step with non-empty candidates
    last_t = T - 1
    while last_t >= 0 and len(log_alpha[last_t]) == 0:
        last_t -= 1
    if last_t < 0:
        return chosen, [float("-inf")] * T

    best_k = int(np.argmax(log_alpha[last_t]))
    chosen[last_t] = candidates_per_t[last_t][best_k]
    log_total = [float("-inf")] * T
    log_total[last_t] = float(log_alpha[last_t][best_k])
    for t in range(last_t, 0, -1):
        if backptr[t] is None or len(backptr[t]) == 0:
            best_k = -1
            continue
        prev_k = int(backptr[t][best_k])
        if prev_k < 0 or len(candidates_per_t[t - 1]) == 0:
            chosen[t - 1] = None
            best_k = 0  # restart marker
            continue
        chosen[t - 1] = candidates_per_t[t - 1][prev_k]
        log_total[t - 1] = float(log_alpha[t - 1][prev_k])
        best_k = prev_k

    return chosen, log_total


# ─────────────────────────────────────────────────────────────────────────────
# Apply to a matched_corrected DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def apply_hmm_to_df(
    df: pd.DataFrame,
    network,
    *,
    east_col: str = "glh_east",
    north_col: str = "glh_north",
    max_radius_m: float = 100.0,
    k: int = 5,
    sigma_z: float = 10.0,
    beta: float = 50.0,
    min_session_len: int = 2,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run HMM map-matching across every (volunteer, glh_session_id) session
    in `df` and add corrected-position columns.

    Adds:
        hmm_pred_glh_east, hmm_pred_glh_north
        hmm_pred_glh_lat,  hmm_pred_glh_lon
        hmm_perp_distance_m         distance from raw to chosen edge
        hmm_pred_deviation_m        distance from corrected to GPX truth
        hmm_improvement_vs_raw_m
        hmm_improvement_vs_stage2_m
    """
    out = df.copy()
    if out.empty:
        for c in ("hmm_pred_glh_east", "hmm_pred_glh_north",
                  "hmm_pred_glh_lat", "hmm_pred_glh_lon",
                  "hmm_perp_distance_m", "hmm_pred_deviation_m",
                  "hmm_improvement_vs_raw_m", "hmm_improvement_vs_stage2_m"):
            out[c] = pd.Series(dtype=float)
        return out

    # Allocate output columns
    for c in ("hmm_pred_glh_east", "hmm_pred_glh_north", "hmm_perp_distance_m"):
        out[c] = np.nan

    keys = ["volunteer", "glh_session_id"]
    n_sessions = 0
    n_pred = 0
    for (v, sid), g in out.sort_values(keys + ["timestamp"]).groupby(keys, sort=True):
        if len(g) < min_session_len:
            continue
        if not (g[east_col].notna() & g[north_col].notna()).any():
            continue
        candidates = find_candidates_per_session(
            g, network,
            east_col=east_col, north_col=north_col,
            max_radius_m=max_radius_m, k=k,
        )
        obs_xy = g[[east_col, north_col]].astype(float).values
        chosen, _ = run_session_viterbi(
            candidates, obs_xy, sigma_z=sigma_z, beta=beta,
        )
        idx = g.index
        for i, cand in enumerate(chosen):
            if cand is None:
                continue
            out.at[idx[i], "hmm_pred_glh_east"] = cand.proj_east
            out.at[idx[i], "hmm_pred_glh_north"] = cand.proj_north
            out.at[idx[i], "hmm_perp_distance_m"] = cand.perp_distance_m
            n_pred += 1
        n_sessions += 1
        if verbose and n_sessions % 50 == 0:
            print(f"  ... {n_sessions} sessions processed, "
                  f"{n_pred} corrected points so far")

    # WGS84
    out["hmm_pred_glh_lat"] = np.nan
    out["hmm_pred_glh_lon"] = np.nan
    mask = out["hmm_pred_glh_east"].notna() & out["hmm_pred_glh_north"].notna()
    if mask.any():
        lat, lon = bng_to_wgs84(
            out.loc[mask, "hmm_pred_glh_east"],
            out.loc[mask, "hmm_pred_glh_north"],
        )
        out.loc[mask, "hmm_pred_glh_lat"] = lat
        out.loc[mask, "hmm_pred_glh_lon"] = lon

    # Deviations
    if {"gpx_east", "gpx_north"}.issubset(out.columns):
        out["hmm_pred_deviation_m"] = bng_distance(
            out["hmm_pred_glh_east"], out["hmm_pred_glh_north"],
            out["gpx_east"], out["gpx_north"],
        )
        if "deviation_m" in out.columns:
            out["hmm_improvement_vs_raw_m"] = (
                out["deviation_m"] - out["hmm_pred_deviation_m"]
            )
        else:
            out["hmm_improvement_vs_raw_m"] = np.nan
        if "corrected_deviation_m" in out.columns:
            out["hmm_improvement_vs_stage2_m"] = (
                out["corrected_deviation_m"] - out["hmm_pred_deviation_m"]
            )
        else:
            out["hmm_improvement_vs_stage2_m"] = np.nan
    else:
        out["hmm_pred_deviation_m"] = np.nan
        out["hmm_improvement_vs_raw_m"] = np.nan
        out["hmm_improvement_vs_stage2_m"] = np.nan

    if verbose:
        print(f"HMM done: {n_sessions} sessions, {n_pred} corrected points")
    return out
