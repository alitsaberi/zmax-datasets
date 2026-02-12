
import numpy as np
import pandas as pd

from zmax_datasets.processing.eeg.artifact_detection.constants import (
    EPS,
    FEATURES,
    FRACTION_VLF,
    GUARD_COLUMNS,
    K1,
    K2,
    MIN_KEEP,
    MIN_QUIET_EPOCHS,
    PTP_RATIO,
    QUIET_QUANTILE,
    READMIT_BETA,
    READMIT_DELTA,
    TAIL_QUANTILE,
    ZERO_CROSSING_RATE,
)
from zmax_datasets.processing.eeg.artifact_detection.thresholds import (
    calculate_quiet_thresholds_per_channel,
    calculate_two_pass_mad_thresholds,
)
from zmax_datasets.utils.exceptions import (
    InvalidChannelCountError,
    MissingChannelThresholdError,
    MissingFeatureThresholdError,
)


def detect_artifacts(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
    
    """Compute artifact labels for a single-channel feature dataframe.

    Args:
        df: Feature dataframe with one row per epoch.
       
    Returns:
        DataFrame with per-epoch artifact flags and thresholds applied.
    """
    
    ######### MAD-BASED THRESHOLDS #################
    thr_2pass = pd.concat(
    [
        calculate_two_pass_mad_thresholds(
            df, 
            feature=f, 
            k1=K1, 
            k2=K2, 
            min_keep=MIN_KEEP
            )
        for f in FEATURES
    ],
    ignore_index=True,
    )

    ############### QUIET-EEG THRESHOLDS ######

    thr_quiet = pd.concat(
        [
            calculate_quiet_thresholds_per_channel(
                df,
                feature_name=f,
                quiet_quantile=QUIET_QUANTILE,
                tail_quantile=TAIL_QUANTILE,
                include_max_abs=True,
                min_quiet_epochs=MIN_QUIET_EPOCHS,
                fallback="relax",
            )
            for f in FEATURES
        ],
        ignore_index=True,
    )

    
    
    
    ############### COMPUTE FINAL STATS THRESHOLDS ######################### 
    thr_all_independent = pd.concat([thr_2pass, thr_quiet], ignore_index=True)
    thr_combo = (
        thr_all_independent
        .pivot_table(
            index=["channel", "feature"],
            columns="method",
            values="thr",
            aggfunc="first",
        )

        .reset_index()
    )
    thr_combo["threshold_final"] = np.nanmax(
        np.column_stack([
            thr_combo.get("two_pass_MAD_10_8", np.nan).astype(float),
            thr_combo.get("quiet_Q99.9", np.nan).astype(float),
        ]),
        axis=1
    )
    threshold_final = thr_combo[["channel","feature","threshold_final"]].copy()


    ################ APPLY THRESHOLDS TO EPOCHS ######################
   
    dfp, thr_wide = _add_feature_flags_single_channel(df, threshold_final)
    ############### GET BAD EPOCHS ####################################
    
    dfp= _get_bad_epochs(dfp,df,thr_wide) 
    info = build_info(dfp, FEATURES)
    
    return dfp,info

def build_info(
    dfp: pd.DataFrame,
    features: list[str],
    *,
    bad_col: str = "bad_independent",
    flatline_col: str = "_bad_flatline_2s",
    flatline_label: str = "FLATLINE_2s",
    prefix_good: str = "IND_GOOD",
    prefix_bad: str = "IND_BAD",
    include_hits_when_good: bool = True,
) -> np.ndarray:
    """Build per-epoch info strings for plotting.

    Produces one string per epoch for a single-channel dfp.

    Rules:
      - hits = features whose `_bad_<feature>` is True
      - optionally add FLATLINE_2s if `flatline_col` is True
      - if `bad_col` is True => "IND_BAD: <hits or UNKNOWN>"
      - else => "IND_GOOD" (optionally with "(hits: ...)" if include_hits_when_good)

    Missing columns are treated as False.
    """
    # Ensure epoch order and integer epoch index
    sub = dfp.sort_values("epoch").set_index("epoch")

    # Build output over all epochs present in dfp
    epochs = sub.index.to_numpy()
    msgs = np.empty(len(epochs), dtype=object)

    bad_flag = sub.get(bad_col, False)
    if not isinstance(bad_flag, pd.Series):
        bad_flag = pd.Series(False, index=sub.index)

    flat_flag = sub.get(flatline_col, False)
    if not isinstance(flat_flag, pd.Series):
        flat_flag = pd.Series(False, index=sub.index)

    # Ensure booleans and fill NaNs
    bad_flag = bad_flag.fillna(False).astype(bool)
    flat_flag = flat_flag.fillna(False).astype(bool)

    # Precompute feature flags safely
    feat_flags: dict[str, pd.Series] = {}
    for f in features:
        col = f"_bad_{f}"
        s = sub.get(col, False)
        if not isinstance(s, pd.Series):
            s = pd.Series(False, index=sub.index)
        feat_flags[f] = s.fillna(False).astype(bool)

    for i, e in enumerate(epochs):
        hits = [f for f in features if bool(feat_flags[f].loc[e])]
        if bool(flat_flag.loc[e]):
            hits.append(flatline_label)

        if bool(bad_flag.loc[e]):
            msgs[i] = f"{prefix_bad}: {','.join(hits) if hits else 'UNKNOWN'}"
        else:
            if include_hits_when_good and hits:
                msgs[i] = f"{prefix_good} (hits: {','.join(hits)})"
            else:
                msgs[i] = prefix_good

    return msgs


def _get_bad_epochs(
        dfp: pd.DataFrame,
        df_feat: pd.DataFrame,
        thr_wide: pd.DataFrame,
    ) -> pd.DataFrame:
    
    """Compute final per-epoch artifact labels from per-feature flags and guards.

    Adds a `bad_independent` boolean column to `dfp` using a combination of:
      - Hard thresholds (always bad)
      - Soft thresholds (bad if enough soft features are exceeded)
      - Physiological "rescue" guard for N3-like epochs
      - Sweat/drift signature guard

    The function expects `dfp` to already contain per-feature `_bad_<feature>`
    boolean columns.

    Args:
        dfp: DataFrame with one row per epoch. Must include:
            - frac_delta, frac_beta, frac_vlf, ptp_ratio, zcr
            - _bad_max_abs, _bad_sub_ptp_max_2s
            - _bad_ptp_robust, _bad_max_cusum, _bad_mean_abs_diff

    Returns:
        The same DataFrame with an added `bad_independent` boolean column.
    """
    
    # N3-like physiology guard 
    n3_like = (dfp["frac_delta"] >=READMIT_DELTA) & (dfp["frac_beta"] <= READMIT_BETA)
    
    # sweat/drift signature (tune thresholds empirically)
    sweat_like = (
    (dfp["frac_vlf"] > FRACTION_VLF) &
    (dfp["ptp_ratio"] > PTP_RATIO) &
    (dfp["zcr"] < ZERO_CROSSING_RATE)
    )

    
    ptp_ratio_thr = df_feat.groupby("channel")["ptp_ratio"].quantile(0.995).to_dict()

    # dfp["_bad_stepdrop"] = dfp["_bad_max_cusum"] & (
    #     dfp["ptp_ratio"] > dfp["channel"].map(ptp_ratio_thr)
    # )

    ptp_ratio_thr = df_feat.groupby("channel")["ptp_ratio"].quantile(0.98).to_dict()
    cusum_thr     = df_feat.groupby("channel")["max_cusum"].quantile(0.98).to_dict()
    
    dfp["_bad_stepdrop"] = (
        (dfp["ptp_ratio"] > dfp["channel"].map(ptp_ratio_thr)) &
        (dfp["max_cusum"]  > dfp["channel"].map(cusum_thr))
    )
    
    # ---- Near-threshold multi-hit rule ----
    near = pd.DataFrame(
        {
            f: dfp[f] / (dfp["channel"].map(thr_wide[f]) + EPS)
            for f in FEATURES
        }
    )

    dfp["_near_count"] = (near > 0.85).sum(axis=1)
    dfp["_bad_near"] = dfp["_near_count"] >= 3
    
    # ---- Make cusum hard ----
    dfp["_bad_step_hard"] = dfp["_bad_max_cusum"]
    
    '''
    # ---- Two-tier flatline ----
    flat_amp_thr = (
        df_feat.groupby("channel")["sub_ptp_p10_2s"].quantile(0.005).to_dict()
        )
    dfp["_bad_flatline_2s"] = (
        (
            (dfp["flat_frac_2s"] >= 0.40)
            & (dfp["sub_ptp_p10_2s"] <= dfp["channel"].map(flat_amp_thr))
        )
        | (
            (dfp["flat_frac_2s"] >= 0.30)
            & (dfp["sub_ptp_p10_2s"] <= dfp["channel"].map(flat_amp_thr) * 1.2)
        )
    )
    '''
    
    hard = (
        dfp["_bad_max_abs"]
        | dfp["_bad_sub_ptp_max_2s"]
        | dfp["_bad_flatline_2s"]
        | dfp["_bad_step_hard"]
        | dfp["_bad_near"]
    )
    
   
    
    soft_cols = ["_bad_ptp_robust", "_bad_max_cusum", "_bad_mean_abs_diff"]
    soft = (dfp[soft_cols].sum(axis=1) >= 2)
    
    # Rescue only if N3-like AND NOT sweat-like
    dfp["bad_independent"] = hard | (soft & (~n3_like | sweat_like))
    return dfp

def _add_feature_flags_single_channel(
    df_feat: pd.DataFrame,
    threshold_final: pd.DataFrame,
    *,
    features: list[str] | None = None,
    guard_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-feature `_bad_<feature>` flags for a single-channel feature table.

    Args:
        df_feat: Feature dataframe for a single channel (e.g. df_L or df_R).
            Must contain columns: `epoch`, `channel`, and all `features`/`guard_cols`.
        threshold_final: Threshold dataframe for the same channel
        (e.g. threshold_final_L or threshold_final_R).

            Expected columns: `channel`, `feature`, `threshold_final`.
        features: Features to flag using thresholds.
        guard_cols: Extra columns to keep (not used for thresholding here).

    Returns:
        dfp: Copy of `df_feat` with added boolean columns `_bad_<feature>`.
        thr_wide: Wide threshold table indexed by channel with one column per feature.

    Raises:
        ValueError: If df_feat contains multiple channels or thresholds are missing.
    """
    features = FEATURES or [
    "sub_ptp_max_2s",
    "ptp_robust",
    "max_abs",
    "mean_abs_diff",
    "max_cusum",
    ]
    guard_columns = GUARD_COLUMNS or [
        "frac_delta",
        "frac_beta",
        "frac_vlf",
        "ptp_ratio",
        "zcr",
    ]

    channels = df_feat["channel"].unique()
    if len(channels) != 1:
        raise InvalidChannelCountError(len(channels))
    ch = channels[0]

    dfp = df_feat[[
        "epoch", "channel",
        *features,
        *guard_columns,
        "stuck_frac_2s",          # <-- ADD THIS
        "diff_rms_min_2s",
        "uniq_p10_2s",
    ]].copy()
    
    num_cols = features + guard_columns + [
    "stuck_frac_2s",
    "diff_rms_min_2s",
    "uniq_p10_2s",
    ]

    dfp[num_cols] = dfp[num_cols].apply(pd.to_numeric, errors="coerce")
    
    #Additional flatline metric
    dfp["_bad_flatline_2s"] = (
        (dfp["stuck_frac_2s"] >= 0.50) &          # e.g., >= 3 of 6 windows stuck
        (dfp["diff_rms_min_2s"] <= 0.05e-6) &     # confirms near-constant
        (dfp["uniq_p10_2s"] <= 8)                 # confirms quantized / stuck
    )
    
    # thresholds wide (index=channel, columns=feature)
    thr_wide = threshold_final.pivot(
        index="channel",
        columns="feature",
        values="threshold_final",
    )

    if ch not in thr_wide.index:
        raise MissingChannelThresholdError(ch)

    # per-feature flags (channel-independent now; one channel only)
    for f in features:
        if f not in thr_wide.columns:
            raise MissingFeatureThresholdError(f, ch)

        thr = float(thr_wide.loc[ch, f])
        dfp[f"_bad_{f}"] = dfp[f].values > thr

    return dfp, thr_wide