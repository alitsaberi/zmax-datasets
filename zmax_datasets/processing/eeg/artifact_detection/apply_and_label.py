
import numpy as np
import pandas as pd

from zmax_datasets.processing.eeg.artifact_detection.constants import (
    EPS,
    FEATURES,
    FRACTION_VLF,
    GUARD_COLS,
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
    LOW_AMP_THR
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
    n_epochs: int | None = None,
    epoch_col: str = "epoch",
    bad_col: str = "bad_independent",
    flatline_col: str = "_bad_flatline_2s",
    flatline_label: str = "FLATLINE_2s",
    flat_stuck_col: str = "_bad_flatline_stuck",
    flat_stuck_label: str = "FLAT_STUCK",
    low_amp_col: str = "_bad_low_amp_flat",
    low_amp_label: str = "LOW_AMP_FLAT",
    harmonic_col: str = "_bad_harmonic_artifact",
    harmonic_label: str = "HARMONIC_ARTIFACT",
    break_col: str = "_bad_break_artifact",
    break_label: str = "BREAK_ARTIFACT",
    prefix_good: str = "IND_GOOD",
    prefix_bad: str = "IND_BAD",
    include_hits_when_good: bool = True,
) -> np.ndarray:
    """
    Build per-epoch info strings for plotting for a single channel.

    Replicates the old behavior:
      - hits = features whose `_bad_<feature>` is True
      - add FLAT_STUCK / LOW_AMP_FLAT / FLATLINE_2s / HARMONIC_ARTIFACT / BREAK_ARTIFACT
      - if `bad_col` is True => "IND_BAD: <hits or UNKNOWN>"
      - else => "IND_GOOD" or "IND_GOOD (hits: ...)"

    Missing columns are treated as False.

    Args:
        dfp: Single-channel dataframe with one row per epoch.
        features: Feature names like ["sub_ptp_max_2s", ...].
        n_epochs: If provided, reindex to [0, ..., n_epochs-1]. If None, use epochs present in dfp.

    Returns:
        np.ndarray of shape (n_epochs,) if n_epochs is provided, else (n_present_epochs,).
    """
    sub = dfp.sort_values(epoch_col).set_index(epoch_col)

    if n_epochs is None:
        epoch_index = pd.Index(sub.index.unique().sort_values())
    else:
        epoch_index = pd.Index(np.arange(n_epochs))

    sub = sub.reindex(epoch_index)

    msgs = np.empty(len(epoch_index), dtype=object)
    msgs[:] = ""

    def _get_bool_series(col_name: str) -> pd.Series:
        s = sub.get(col_name, False)
        if not isinstance(s, pd.Series):
            s = pd.Series(False, index=sub.index)
        return s.fillna(False).astype(bool)

    bad_flag = _get_bool_series(bad_col)
    flat_flag = _get_bool_series(flatline_col)
    flat_stuck_flag = _get_bool_series(flat_stuck_col)
    low_amp_flag = _get_bool_series(low_amp_col)
    harmonic_flag = _get_bool_series(harmonic_col)
    break_flag = _get_bool_series(break_col)

    feat_flags: dict[str, pd.Series] = {}
    for feature in features:
        feat_flags[feature] = _get_bool_series(f"_bad_{feature}")

    for i, epoch in enumerate(epoch_index):
        hits = [feature for feature in features if bool(feat_flags[feature].loc[epoch])]

        # Match old precedence/behavior
        if bool(flat_stuck_flag.loc[epoch]):
            hits.append(flat_stuck_label)

        if bool(low_amp_flag.loc[epoch]):
            hits.append(low_amp_label)
        elif bool(flat_flag.loc[epoch]):
            hits.append(flatline_label)

        if bool(harmonic_flag.loc[epoch]):
            hits.append(harmonic_label)

        if bool(break_flag.loc[epoch]):
            hits.append(break_label)

        if bool(bad_flag.loc[epoch]):
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
        | dfp["_bad_harmonic_artifact"]
        | dfp["_bad_break_artifact"]
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
    "max_block_median_jump"
    ]
    guard_columns = GUARD_COLS or [
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

    sel_cols = [
    "epoch", "channel",
    *features,
    *guard_columns,
    "stuck_frac_2s",
    "diff_rms_min_2s",
    "uniq_p10_2s",
    "harmonic_flag",
    "harmonic_f0",
    "harmonic_score",
    "harmonic_n",
    ]
    sel_cols = list(dict.fromkeys(sel_cols))
    
    dfp = df_feat[sel_cols].copy()
    
    num_cols = [c for c in sel_cols if c not in ["epoch", "channel"]]
    dfp[num_cols] = dfp[num_cols].apply(pd.to_numeric, errors="coerce")
    
    # Additional flatline metric
    dfp["_bad_flatline_stuck"] = (
        (dfp["stuck_frac_2s"] >= 0.50) &          # e.g., >= 3 of 6 windows stuck
        (dfp["diff_rms_min_2s"] <= 0.05e-6) &     # confirms near-constant
        (dfp["uniq_p10_2s"] <= 8)                 # confirms quantized / stuck
    )
    
    # ---- simple absolute low-amplitude detector ----
    dfp["_bad_low_amp_flat"] = dfp["ptp_robust"] < LOW_AMP_THR
    
    
    # combined flatline-like detector
    dfp["_bad_flatline_2s"] = dfp["_bad_flatline_stuck"] | dfp["_bad_low_amp_flat"]
    
    # ---- harmonic artifact detector ----
    dfp["_bad_harmonic_artifact"] = (
        dfp["harmonic_flag"].fillna(False).astype(bool)
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
    
    # break / disconnect artifact: make block-median jump a hard rule
    dfp["_bad_break_artifact"] = dfp["_bad_max_block_median_jump"]
    

    return dfp, thr_wide