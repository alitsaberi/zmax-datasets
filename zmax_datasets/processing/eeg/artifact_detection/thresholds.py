
from typing import Literal

import numpy as np
import pandas as pd

from zmax_datasets.processing.eeg.artifact_detection.constants import (
    EPS,
    K1,
    K2,
    KEEP_Q_FALLBACK,
    MIN_FRAC_KEEP,
    MIN_KEEP,
    MIN_QUIET_EPOCHS,
    QUIET_QUANTILE,
    QUIET_RELAXATION_QUANTILES,
    SIGMA,
    TAIL_QUANTILE,
)


def _mad_loc_scale(signal: np.ndarray):
    """Compute robust location and scale using median and MAD.

    Uses the median as a robust location estimate and MAD (median absolute
    deviation) scaled by 1.4826 as a robust approximation to standard deviation
    under Gaussian assumptions.

    Args:
        signal: 1D array of values.

    Returns:
        A tuple (median, robust_sigma), where robust_sigma ~= std for Gaussian.
    """
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    sigma = SIGMA * mad
    return med, sigma

def _validate_quiet_threshold_inputs(
    features: pd.DataFrame,
    feature_name: str,
) -> tuple[list[str], list[str]]:
    required_cols = ["channel", "line_length", "frac_beta", "max_abs", feature_name]
    numeric_cols = ["line_length", "frac_beta", "max_abs", feature_name]

    missing = [c for c in required_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required columns for quiet thresholds: {missing}")

    return required_cols, numeric_cols


def _filter_finite_rows(
        channel_df: pd.DataFrame,
        numeric_cols: list[str],
    ) -> pd.DataFrame:
        
    finite_mask = np.isfinite(
        channel_df[numeric_cols].to_numpy(dtype=float)
    ).all(axis=1)
    
    return channel_df.loc[finite_mask]


def _quiet_mask_for_quantile(
    channel_df: pd.DataFrame,
    *,
    q: float,
    include_max_abs: bool,
) -> np.ndarray:
    line_length = channel_df["line_length"].to_numpy(dtype=float)
    frac_beta = channel_df["frac_beta"].to_numpy(dtype=float)

    ll_thr = float(np.percentile(line_length, q))
    beta_thr = float(np.percentile(frac_beta, q))

    mask = (line_length <= ll_thr) & (frac_beta <= beta_thr)

    if include_max_abs:
        max_abs = channel_df["max_abs"].to_numpy(dtype=float)
        ma_thr = float(np.percentile(max_abs, q))
        mask = mask & (max_abs <= ma_thr)

    return mask


def _select_quiet_values_with_relaxation(
        channel_df: pd.DataFrame,
        feature_name: str,
        *,
        quiet_quantile: float,
        include_max_abs: bool,
        min_quiet_epochs: int,
        fallback: Literal["relax", "mad"],
    ) -> tuple[np.ndarray, np.ndarray, float, str]:
        
    """Return (quiet_values, quiet_mask, q_used, note)."""
    
    q_used = float(quiet_quantile)
    note = ""

    mask = _quiet_mask_for_quantile(
        channel_df, q=q_used, include_max_abs=include_max_abs
        )
    
    quiet_values = channel_df.loc[mask, feature_name].to_numpy(dtype=float)

    if quiet_values.size < min_quiet_epochs and fallback == "relax":
        for q_try in QUIET_RELAXATION_QUANTILES:
            mask = _quiet_mask_for_quantile(
                channel_df, q=q_try, include_max_abs=include_max_abs
                )
            quiet_values = channel_df.loc[mask, feature_name].to_numpy(dtype=float)
            q_used = float(q_try)
            if quiet_values.size >= min_quiet_epochs:
                note = f"relaxed quiet to q={q_try:g}"
                break

    return quiet_values, mask, q_used, note



def calculate_quiet_thresholds_per_channel(
    features: pd.DataFrame,
    feature_name: str,
    *,
    quiet_quantile: float = QUIET_QUANTILE,
    tail_quantile: float = TAIL_QUANTILE,
    include_max_abs: bool = True,
    min_quiet_epochs: int = MIN_QUIET_EPOCHS,
    fallback: Literal["relax", "mad"] = "relax",
) -> pd.DataFrame:
    """Compute per-channel tail thresholds within "quiet" epochs."""
    _, numeric_cols = _validate_quiet_threshold_inputs(features, feature_name)

    results: list[dict[str, object]] = []

    for channel in features["channel"].unique():
        channel_df = features.loc[features["channel"] == channel].copy()
        channel_df = _filter_finite_rows(channel_df, numeric_cols)

        if channel_df.empty:
            results.append(
                {
                    "channel": channel,
                    "feature": feature_name,
                    "method": f"quiet_Q{tail_quantile:g}",
                    "thr": np.nan,
                    "quiet_frac": 0.0,
                    "n_quiet": 0,
                    "note": "no finite data",
                }
            )
            continue

        quiet_values, mask, q_used, note = _select_quiet_values_with_relaxation(
            channel_df,
            feature_name,
            quiet_quantile=quiet_quantile,
            include_max_abs=include_max_abs,
            min_quiet_epochs=min_quiet_epochs,
            fallback=fallback,
        )

        too_few = (
            quiet_values.size == 0 
            or quiet_values.size < max(10, min_quiet_epochs // 3)
            )
        quiet_frac = float(mask.mean())
        n_quiet = int(mask.sum())

        if too_few:
            if fallback == "mad":
                all_values = channel_df[feature_name].to_numpy(dtype=float)
                thr = _mad_loc_scale(all_values)
                results.append(
                    {
                        "channel": channel,
                        "feature": feature_name,
                        "method": f"quiet_Q{tail_quantile:g}",
                        "thr": thr,
                        "quiet_frac": quiet_frac,
                        "n_quiet": n_quiet,
                        "note": "fallback MAD_10",
                    }
                )
            else:
                results.append(
                    {
                        "channel": channel,
                        "feature": feature_name,
                        "method": f"quiet_Q{tail_quantile:g}",
                        "thr": np.nan,
                        "quiet_frac": quiet_frac,
                        "n_quiet": n_quiet,
                        "note": f"too few quiet epochs (n={quiet_values.size})",
                    }
                )
            continue

        thr = float(np.percentile(quiet_values, tail_quantile))
        results.append(
            {
                "channel": channel,
                "feature": feature_name,
                "method": f"quiet_Q{tail_quantile:g}",
                "thr": thr,
                "quiet_frac": quiet_frac,
                "n_quiet": n_quiet,
                "note": note if note else f"q_used={q_used:g}",
            }
        )

    return pd.DataFrame(results)



def calculate_two_pass_mad_thresholds(
        df: pd.DataFrame,
        feature: str,
        k1: float = K1,
        k2: float = K2,
        min_keep: int = MIN_KEEP,
        min_frac_keep: float = MIN_FRAC_KEEP,
        eps: float = EPS,
        keep_q_fallback: float = KEEP_Q_FALLBACK,
        add_qc: bool = True,
    ) -> pd.DataFrame:
    
    rows: list[dict[str, object]] = []

    for ch in df["channel"].unique():
        x = df.loc[df["channel"] == ch, feature].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n < 10:
            continue

        # ---- pass 1 ----
        med1, sig1 = _mad_loc_scale(x)
        thr1 = float(med1 + k1 * max(sig1, eps))

        keep = x <= thr1
        n_kept_pass1 = int(keep.sum())
        frac_kept_pass1 = n_kept_pass1 / max(n, 1)

        did_fallback = False

        # ---- fallback if pass1 is too aggressive ----
        if (
                (n_kept_pass1 < min_keep and n >= min_keep) 
        or (frac_kept_pass1 < min_frac_keep) 
        ):
            cut = float(np.quantile(x, keep_q_fallback))
            keep = x <= cut
            did_fallback = True

        x2 = x[keep]
        if x2.size < 10:
            x2 = x

        # ---- pass 2 ----
        med2, sig2 = _mad_loc_scale(x2)
        thr2 = float(med2 + k2 * max(sig2, eps))

        row: dict[str, object] = {
            "channel": ch,
            "feature": feature,
            "method": f"two_pass_MAD_{k1:g}_{k2:g}",
            "thr": thr2,
        }

        if add_qc:
            row.update(
                {
                    "thr_pass1": thr1,
                    "kept_frac_pass1": float(frac_kept_pass1),
                    "n": int(n),
                    "n_kept": int(x2.size),
                    "fallback_used": did_fallback,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)






