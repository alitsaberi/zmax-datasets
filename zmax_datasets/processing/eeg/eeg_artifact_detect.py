from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yasa
from loguru import logger
from matplotlib.widgets import Slider
from scipy.signal import welch, medfilt
from scipy.stats import kurtosis
from yasa import sleep_statistics, transition_matrix

from zmax_datasets.settings import ARTIFACT_DETECTION, EEG_BANDS_HZ, EEG_SIGNAL_BAND
from zmax_datasets.utils.data import Data,epochify,detect_voltage_unit,microvolts_to_volts, volts_to_microvolts,ensure_volts,ensure_microvolts
from zmax_datasets.processing.spectral import integrate_bandpower

ROBUST_PTP_Q_HIGH = 99.5
ROBUST_PTP_Q_LOW = 0.5
SUB_WIN_SEC = 2.0
READMIT_DELTA = 0.65
READMIT_BETA = 0.1
FRAC_VLF= 1.5
PTP_RATIO=1.4
ZCR=0.02


FEATURES = ["sub_ptp_max_2s", 
            "ptp_robust", 
            "max_abs", 
            "mean_abs_diff", 
            "max_cusum"]

GUARD_COLS = ["frac_delta", "frac_beta", "frac_vlf", "ptp_ratio", "zcr"]

# Percentile rates from the distribution plots 
FEATURE_RATES = {
"max_abs": 0.005,
"ptp_robust": 0.005,
"sub_ptp_max_2s": 0.005,
"mean_abs_diff": 0.001,
"max_cusum": 0.001,
}






def detrend_median(x, sf, win_sec=2.0):
    """Remove slow trends using a running median filter.

    The median is computed over a sliding window and subtracted from the
    signal. This is commonly used to remove baseline drift while preserving
    transient activity.

    Args:
        x: Input signal array. Expected shape is
            ``(n_epochs, n_channels, n_samples)``.
        sf: Sampling frequency in Hz.
        win_sec: Length of the median filter window in seconds.

    Returns:
        Detrended signal with the same shape as ``x``.

    Raises:
        ValueError: If the computed kernel size is less than 1.
    """
    k = int(win_sec * sf)
    if k < 1:
        raise ValueError("Median filter window must be at least one sample.")

    if k % 2 == 0:
        k += 1
    # medfilt works along last axis; apply per epoch/channel
    return x - medfilt(x, kernel_size=(1, 1, k))


def get_bad_epochs(dfp):
    
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
    (dfp["frac_vlf"] > FRAC_VLF) &
    (dfp["ptp_ratio"] > PTP_RATIO) &
    (dfp["zcr"] < ZCR)
    )

    
    hard = dfp["_bad_max_abs"] | dfp["_bad_sub_ptp_max_2s"]
    
    soft_cols = ["_bad_ptp_robust", "_bad_max_cusum", "_bad_mean_abs_diff"]
    soft = (dfp[soft_cols].sum(axis=1) >= 2)
    
    # Rescue only if N3-like AND NOT sweat-like
    dfp["bad_independent"] = hard | (soft & (~n3_like | sweat_like))
    #dfp["bad_independent"] = hard | (soft & ~(n3_like & ~sweat_like)) | sweat_like
    return dfp

def add_feature_flags_single_channel(
    df_feat: pd.DataFrame,
    thr_final: pd.DataFrame,
    *,
    features: list[str] | None = None,
    guard_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-feature `_bad_<feature>` flags for a single-channel feature table.

    Args:
        df_feat: Feature dataframe for a single channel (e.g. df_L or df_R).
            Must contain columns: `epoch`, `channel`, and all `features`/`guard_cols`.
        thr_final: Threshold dataframe for the same channel (e.g. thr_final_L or thr_final_R).
            Expected columns: `channel`, `feature`, `thr_final`.
        features: Features to flag using thresholds.
        guard_cols: Extra columns to keep (not used for thresholding here).

    Returns:
        dfp: Copy of `df_feat` with added boolean columns `_bad_<feature>`.
        thr_wide: Wide threshold table indexed by channel with one column per feature.

    Raises:
        ValueError: If df_feat contains multiple channels or thresholds are missing.
    """
    FEATURES = features or ["sub_ptp_max_2s", "ptp_robust", "max_abs", "mean_abs_diff", "max_cusum"]
    GUARD_COLS = guard_cols or ["frac_delta", "frac_beta", "frac_vlf", "ptp_ratio", "zcr"]

    channels = df_feat["channel"].unique()
    if len(channels) != 1:
        raise ValueError(f"df_feat must contain exactly one channel, got {channels}")
    ch = channels[0]

    # keep only needed columns
    dfp = df_feat[["epoch", "channel", *FEATURES, *GUARD_COLS]].copy()

    # ensure numeric
    num_cols = [*FEATURES, *GUARD_COLS]
    dfp[num_cols] = dfp[num_cols].apply(pd.to_numeric, errors="coerce")

    # thresholds wide (index=channel, columns=feature)
    thr_wide = thr_final.pivot(index="channel", columns="feature", values="thr_final")

    # sanity: ensure this channel exists in thr_wide
    if ch not in thr_wide.index:
        raise ValueError(f"No thresholds found for channel {ch!r}")

    # per-feature flags (channel-independent now; one channel only)
    for f in FEATURES:
        if f not in thr_wide.columns:
            raise ValueError(f"Missing threshold for feature {f!r} (channel {ch!r})")

        thr = float(thr_wide.loc[ch, f])
        dfp[f"_bad_{f}"] = dfp[f].values > thr

    return dfp, thr_wide


def plot_distribution_with_tails(df, channel, feature, qs=(99, 99.5, 99.9)):
    
    """Plot a feature distribution for a channel with percentile threshold lines.

    Plots a histogram of `feature` values for the specified `channel`, and
    overlays vertical lines at the given percentiles.

    Args:
        df: Feature DataFrame containing `channel` and `feature` columns.
        channel: Channel identifier to filter by (e.g., "EEG_L").
        feature: Feature column name to plot.
        qs: Percentiles to show as dashed vertical lines (e.g., (99, 99.5, 99.9)).

    Returns:
        None. Displays the plot using matplotlib.
    """
    x = df[df.channel == channel][feature].values

    plt.figure(figsize=(6,4))
    plt.hist(x, bins=200, density=True, alpha=0.7)
    
    for q in qs:
        thr = np.percentile(x, q)
        plt.axvline(thr, linestyle="--", label=f"Q{q} = {thr:.3e}")

    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"{channel} — {feature}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def quantile_thresholds_by_feature_rate(df, feature_rates):
    
    """Compute per-channel quantile thresholds from desired tail rates.

   For each (feature, rate) pair, computes the quantile threshold at
   q = 100 * (1 - rate) independently for each channel.

   Args:
       df: DataFrame containing `channel` and feature columns.
       feature_rates: Mapping from feature name to tail rate (e.g., 0.005 means
           threshold at the 99.5th percentile).

   Returns:
       DataFrame with columns:
           - channel
           - feature
           - method (always "quantile_rate")
           - thr (threshold value)
   """
    
    rows = []
    for feature, rate in feature_rates.items():
        q = 100 * (1 - rate)
        for ch in df["channel"].unique():
            x = df.loc[(df["channel"] == ch) & np.isfinite(df[feature]), feature].values
            thr = np.percentile(x, q) if len(x) else np.nan
            rows.append({
                "channel": ch,
                "feature": feature,
                "method": f"quantile_rate",  # nicer label
                "thr": float(thr),
            })
    return pd.DataFrame(rows)


def _mad_loc_scale(x: np.ndarray):
    """Compute robust location and scale using median and MAD.

    Uses the median as a robust location estimate and MAD (median absolute
    deviation) scaled by 1.4826 as a robust approximation to standard deviation
    under Gaussian assumptions.

    Args:
        x: 1D array of values.

    Returns:
        A tuple (median, robust_sigma), where robust_sigma ~= std for Gaussian.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return med, sigma

def two_pass_mad_thresholds(
    df: pd.DataFrame,
    feature: str,
    k1: float = 10.0,
    k2: float = 8.0,
    min_keep: int = 50,
    min_frac_keep: float = 0.20,
    eps: float = 1e-30,
    keep_q_fallback: float = 0.80,   # fallback keep lowest 80% if pass1 removes too much
    add_qc: bool = True,
):
    """Compute a two-pass robust (MAD-based) threshold per channel.

    The threshold is computed in two stages:
      Pass 1:
        thr1 = med1 + k1 * sigma1
        Keep values <= thr1. If too few values are kept, fall back to keeping
        the lowest `keep_q_fallback` fraction.
      Pass 2:
        thr2 = med2 + k2 * sigma2 computed on the kept values.

    Args:
        df: DataFrame containing `channel` and `feature` columns.
        feature: Feature column name to threshold.
        k1: Multiplier for robust sigma in pass 1.
        k2: Multiplier for robust sigma in pass 2.
        min_keep: Minimum number of points to keep after pass 1 (when possible).
        min_frac_keep: Minimum fraction of points to keep after pass 1.
        eps: Minimum scale to avoid division by zero / degenerate MAD.
        keep_q_fallback: Fallback quantile for keeping values if pass 1 is too
            aggressive (e.g., 0.80 keeps the lowest 80%).
        add_qc: Whether to include QC fields (pass1 threshold, kept fraction, etc.).

    Returns:
        DataFrame with one row per channel containing:
            - channel, feature, method, thr
        If `add_qc=True`, also includes:
            - thr_pass1, kept_frac_pass1, n, n_kept, fallback_used
    """
    rows = []
    for ch in df["channel"].unique():
        x = df.loc[df["channel"] == ch, feature].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 10:
            continue

        # ---- pass 1 ----
        med1, sig1 = _mad_loc_scale(x)
        thr1 = med1 + k1 * max(sig1, eps)

        keep = x <= thr1
        n_kept = int(keep.sum())
        frac_kept = n_kept / max(n, 1)

        # ---- fallback if pass1 is too aggressive ----
        if (n_kept < min_keep and n >= min_keep) or (frac_kept < min_frac_keep):
            # keep lowest keep_q_fallback fraction
            cut = np.quantile(x, keep_q_fallback)
            keep = x <= cut
            n_kept = int(keep.sum())
            frac_kept = n_kept / max(n, 1)

        x2 = x[keep]
        if x2.size < 10:
            # last resort: use all points
            x2 = x

        # ---- pass 2 ----
        med2, sig2 = _mad_loc_scale(x2)
        thr2 = med2 + k2 * max(sig2, eps)

        row = {
            "channel": ch,
            "feature": feature,
            "method": f"two_pass_MAD_{k1:g}_{k2:g}",
            "thr": float(thr2),
        }
        if add_qc:
            row.update({
                "thr_pass1": float(thr1),
                "kept_frac_pass1": float(frac_kept),
                "n": int(n),
                "n_kept": int(x2.size),
                "fallback_used": bool((n_kept < min_keep and n >= min_keep) or (frac_kept < min_frac_keep)),
            })
        rows.append(row)

    return pd.DataFrame(rows)


def quiet_thresholds_per_channel(
    df,
    feature,
    q_quiet=50,
    q_tail=99.9,
    include_max_abs=True,
    min_quiet=30,          # require at least this many quiet epochs
    fallback="relax"       # "relax" or "mad"
):
    """Compute quiet-epoch tail thresholds per channel.

    Defines a set of "quiet" epochs using low quantiles of guard features
    (line_length, frac_beta, and optionally max_abs). The threshold for the
    target feature is then computed as the `q_tail` percentile within the quiet
    subset.

    If the quiet subset is too small, behavior depends on `fallback`:
      - "relax": progressively relaxes the quiet quantiles (e.g., 60/70/80).
      - "mad": falls back to a MAD-based threshold on all epochs.

    Args:
        df: DataFrame containing `channel`, guard features, and `feature`.
        feature: Feature column name to threshold.
        q_quiet: Quantile used to define quiet epochs (smaller is stricter).
        q_tail: Tail percentile used as the threshold within quiet epochs.
        include_max_abs: Whether to include max_abs in the quiet-epoch definition.
        min_quiet: Minimum number of quiet epochs required.
        fallback: Strategy if too few quiet epochs are found ("relax" or "mad").

    Returns:
        DataFrame with one row per channel containing:
            - channel, feature, method, thr, quiet_frac, n_quiet, note
    """
    out = []

    for ch in df["channel"].unique():
        dch = df[df["channel"] == ch].copy()

        # Keep only finite values needed for masks + target feature
        needed = ["line_length", "frac_beta", "max_abs", feature]
        for col in needed:
            dch = dch[np.isfinite(dch[col].values)]

        if len(dch) == 0:
            out.append({
                "channel": ch, "feature": feature, "method": f"quiet_Q{q_tail}",
                "thr": np.nan, "quiet_frac": 0.0, "n_quiet": 0,
                "note": "no finite data"
            })
            continue

        def make_quiet_mask(q_ll, q_b, q_ma):
            ll_thr   = np.percentile(dch["line_length"].values, q_ll)
            beta_thr = np.percentile(dch["frac_beta"].values,  q_b)
            m = (dch["line_length"].values <= ll_thr) & (dch["frac_beta"].values <= beta_thr)
            if include_max_abs:
                ma_thr = np.percentile(dch["max_abs"].values, q_ma)
                m = m & (dch["max_abs"].values <= ma_thr)
            return m

        # Pass 1: requested criteria
        quiet_mask = make_quiet_mask(q_quiet, q_quiet, q_quiet)
        xq = dch.loc[quiet_mask, feature].values

        note = ""
        q_used = (q_quiet, q_quiet, q_quiet)

        # If too few quiet epochs, relax criteria automatically
        if len(xq) < min_quiet and fallback == "relax":
            # try progressively less strict until enough samples
            for q_try in [60, 70, 80]:  # enlarge quiet set
                quiet_mask = make_quiet_mask(q_try, q_try, q_try)
                xq = dch.loc[quiet_mask, feature].values
                q_used = (q_try, q_try, q_try)
                if len(xq) >= min_quiet:
                    note = f"relaxed quiet to q={q_try}"
                    break

        # If still too small, fallback to MAD on "all" (or on LL+beta only)
        if len(xq) == 0 or len(xq) < max(10, min_quiet // 3):
            if fallback == "mad":
                x = dch[feature].values
                med = np.median(x)
                mad = np.median(np.abs(x - med))
                robust_sigma = 1.4826 * mad
                thr = float(med + 10 * robust_sigma)  # or parameterize k
                out.append({
                    "channel": ch, "feature": feature, "method": f"quiet_Q{q_tail}",
                    "thr": thr, "quiet_frac": float(quiet_mask.mean()),
                    "n_quiet": int(quiet_mask.sum()),
                    "note": "fallback MAD_10"
                })
                continue
            else:
                out.append({
                    "channel": ch, "feature": feature, "method": f"quiet_Q{q_tail}",
                    "thr": np.nan, "quiet_frac": float(quiet_mask.mean()),
                    "n_quiet": int(quiet_mask.sum()),
                    "note": f"too few quiet epochs (n={len(xq)})"
                })
                continue

        thr = float(np.percentile(xq, q_tail))

        out.append({
            "channel": ch,
            "feature": feature,
            "method": f"quiet_Q{q_tail}",
            "thr": thr,
            "quiet_frac": float(quiet_mask.mean()),
            "n_quiet": int(quiet_mask.sum()),
            "note": note if note else f"q_used={q_used}"
        })

    return pd.DataFrame(out)












def plot_eeg_arti_event(
    t,
    *signals,
    fs=256,
    win_sec=30,
    labels=None,
    bad_mask=None,
    info=None,
    spindles=None,
    slow_waves=None,
):
    """
    Sliding-window EEG viewer with:
      - per-channel artifact highlighting
      - per-channel info text (why BAD/GOOD)
      - optional spindle shading per channel
      - optional slow-wave shading per channel

    Parameters
    ----------
    t : 1D array
        Time in seconds from start, same length as each signal.
    *signals : 1D arrays
        One or more signals of shape (n_samples,).
    fs : float
        Sampling frequency (Hz), used only to compute number of windows.
    win_sec : float
        Window size in seconds (e.g. 30).
    labels : list of str
        Channel labels, length = number of signals.
        IMPORTANT: if `spindles` / `slow_waves` are given, their dict
        keys should match these labels.
    bad_mask : None or array (n_epochs, n_channels) or (n_epochs,)
        Boolean array; True = BAD. Used to tint background pink.
    info : None or array (n_epochs, n_channels) or (n_epochs,)
        String descriptions of why an epoch/channel is BAD/GOOD.
    spindles : None or dict[label -> DataFrame]
        For each label, a DataFrame with at least columns 'Start' and 'End'
        (seconds from start), e.g. from spindles.summary().
    slow_waves : None or dict[label -> DataFrame]
        Same idea as `spindles`, but for slow waves.
    """

    n_ch = len(signals)
    if labels is None:
        labels = [f"Ch{i + 1}" for i in range(n_ch)]

    total_duration = t[-1] - t[0]
    n_windows = max(1, int(np.floor(total_duration / win_sec)))

    # --- initial window ---
    win_idx0 = 0
    start0 = win_idx0 * win_sec
    end0 = start0 + win_sec
    mask0 = (t >= start0) & (t < end0)

    # --- create figure & axes ---
    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=(10, 3 * n_ch))
    if n_ch == 1:
        axes = [axes]

    plt.subplots_adjust(bottom=0.15)

    # --- plot initial data + info text placeholders ---
    lines = []
    info_texts = []
    spindle_patches = [[] for _ in range(n_ch)]
    sw_patches = [[] for _ in range(n_ch)]

    for ax, sig, lbl in zip(axes, signals, labels):
        (line,) = ax.plot(t[mask0], sig[mask0])
        lines.append(line)

        ax.set_ylabel(lbl)
        ax.set_xlim(start0, end0)
        if mask0.any():
            ax.set_ylim(sig[mask0].min(), sig[mask0].max())
        ax.set_facecolor("white")

        txt = ax.text(
            0.01,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        info_texts.append(txt)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Sliding-window EEG viewer")

    # --- slider ---
    ax_slider = plt.axes([0.1, 0.04, 0.8, 0.03])
    slider = Slider(
        ax_slider,
        label=f"Window index ({win_sec}s each)",
        valmin=0,
        valmax=n_windows - 1,
        valinit=win_idx0,
        valstep=1,
    )

    # --- update callback ---
    def update(win_idx):
        win_idx = int(win_idx)
        start = win_idx * win_sec
        end = start + win_sec
        mask = (t >= start) & (t < end)

        for ch_idx, (line, sig, ax, txt) in enumerate(
            zip(lines, signals, axes, info_texts)
        ):
            # update signal
            line.set_data(t[mask], sig[mask])
            ax.set_xlim(start, end)
            if mask.any():
                ax.set_ylim(sig[mask].min(), sig[mask].max())

            # background color from bad_mask
            if bad_mask is not None:
                if bad_mask.ndim == 1:
                    is_bad = bool(bad_mask[win_idx])
                else:
                    is_bad = bool(bad_mask[win_idx, ch_idx])
                ax.set_facecolor("mistyrose" if is_bad else "white")
            else:
                ax.set_facecolor("white")

            # info text
            if info is not None:
                if info.ndim == 1:
                    msg = str(info[win_idx])
                else:
                    msg = str(info[win_idx, ch_idx])
            else:
                msg = ""
            txt.set_text(msg)

            # remove old spindle & SW patches
            for p in spindle_patches[ch_idx]:
                p.remove()
            spindle_patches[ch_idx] = []

            for p in sw_patches[ch_idx]:
                p.remove()
            sw_patches[ch_idx] = []

            ch_label = labels[ch_idx]

            # ---- add spindle shading ----
            if spindles is not None:
                sp_df = spindles.get(ch_label, None)
                if sp_df is not None and not sp_df.empty:
                    m = (sp_df["End"] >= start) & (sp_df["Start"] <= end)
                    for _, row in sp_df[m].iterrows():
                        s0 = row["Start"]
                        s1 = row["End"]
                        s0_clip = max(s0, start)
                        s1_clip = min(s1, end)
                        patch = ax.axvspan(s0_clip, s1_clip, alpha=0.35, color="gold")
                        spindle_patches[ch_idx].append(patch)

            # ---- add slow-wave shading ----
            if slow_waves is not None:
                sw_df = slow_waves.get(ch_label, None)
                if sw_df is not None and not sw_df.empty:
                    # YASA SW summary also has 'Start' and 'End' columns
                    m = (sw_df["End"] >= start) & (sw_df["Start"] <= end)
                    for _, row in sw_df[m].iterrows():
                        s0 = row["Start"]
                        s1 = row["End"]
                        s0_clip = max(s0, start)
                        s1_clip = min(s1, end)
                        patch = ax.axvspan(
                            s0_clip, s1_clip, alpha=0.25, color="lightskyblue"
                        )
                        sw_patches[ch_idx].append(patch)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig.show()
    return fig, slider









def create_and_process_mne(eeg: pd.DataFrame, sample_rate: float):
    """Create MNE objects and epochs for available EEG channels.

    Processes left and right EEG channels independently if present in the input
    dataframe. Channels that are missing or all-NaN are skipped.

    Args:
        eeg: Timeseries dataframe containing `timestamp` and optionally "EEG L"
            and/or "EEG R".
        sample_rate: Sampling rate in Hz.

    Returns:
        Dictionary mapping channel name ("EEG_L", "EEG_R") to per-channel outputs
        produced by `process_eeg_channel`.

    Raises:
        ValueError: If neither channel is available (missing or all-NaN).
    """
    outputs = {}

    out_L = process_eeg_channel(
         eeg,
        channel_column="EEG L",
        channel_name="EEG_L",
        sample_rate=sample_rate,
    )
    
    if out_L is not None:
        outputs["EEG_L"] = out_L

    out_R = process_eeg_channel(
        eeg,
        channel_column="EEG R",
        channel_name="EEG_R",
        sample_rate=sample_rate,
    )
    if out_R is not None:
        outputs["EEG_R"] = out_R

    if not outputs:
        raise ValueError("No valid EEG channels found (EEG L / EEG R are missing or all-NaN).")

    return outputs

def process_eeg_channel(
    eeg: pd.DataFrame,
    *,
    channel_column: str,
    channel_name: str,
    sample_rate: float,
    epoch_duration: float = 30.0,
    notch_freq: float = 50.0,
    phys_band: tuple[float, float] = (0.3, 35.0),
    vlf_band: tuple[float, float] = (0.05, 35.0),
    ):
    """Create an MNE Raw object for one EEG channel and run phys/VLF pipelines.

    Args:
        eeg: Timeseries dataframe containing `channel_column` and `timestamp`.
        channel_column: Column name in `eeg` (e.g. "EEG L").
        channel_name: MNE channel name to use (e.g. "EEG_L").
        sample_rate: Sampling rate in Hz.
        epoch_duration: Epoch duration in seconds (fixed-length epochs).
        notch_freq: Notch filter frequency in Hz (e.g. 50 Hz).
        phys_band: Bandpass (low, high) for phys pipeline.
        vlf_band: Bandpass (low, high) for VLF pipeline.

    A dictionary with:
            - channel: The channel name used in MNE.
            - raw: Base Raw object (notch-filtered).
            - raw_phys: Raw object filtered with `phys_band`.
            - raw_vlf: Raw object filtered with `vlf_band`.
            - epochs_phys: Fixed-length epochs from `raw_phys`.
            - phys_signal: 1D NumPy array of phys-filtered signal.
            - vlf_signal: 1D NumPy array of VLF-filtered signal.
            - epochs_data: 2D NumPy array of shape (n_epochs, n_times).

    Raises:
        KeyError: If required columns are missing.
        ValueError: If timestamps are invalid or data is empty.
    """
    if channel_column not in eeg.columns:
        raise KeyError(f"Missing column: {channel_column}")
    if "timestamp" not in eeg.columns:
        raise KeyError("Missing column: timestamp")
    if len(eeg) == 0:
        raise ValueError("Input dataframe is empty")

    # Convert signal to numpy and ensure it's in volts.
    signal = eeg[channel_column].to_numpy()
    # All NaN -> skip (channel not available)
    if not np.isfinite(signal).any():
        return None
    
    
    signal_v = ensure_volts(signal)
    
    # If ensure_volts returns NaNs only (extremely defensive)
    if not np.isfinite(signal_v).any():
        return None

    meas_date = pd.Timestamp(eeg["timestamp"].iloc[0])

    info = mne.create_info([channel_name], sfreq=sample_rate, ch_types="eeg")
    raw = mne.io.RawArray(signal_v[np.newaxis, :], info)
    raw.set_meas_date(meas_date)

    # Notch filter (applied before branching into pipelines)
    raw.notch_filter(notch_freq)

    # --- PHYS pipeline (artifact detection) ---
    raw_phys = raw.copy()
    raw_phys.filter(phys_band[0], phys_band[1], phase="zero-double", fir_design="firwin")

    # --- VLF pipeline (sweat audit) ---
    raw_vlf = raw.copy()
    raw_vlf.filter(vlf_band[0], vlf_band[1], phase="zero-double", fir_design="firwin")

    # Epochs from phys pipeline
    epochs_phys = mne.make_fixed_length_epochs(raw_phys, duration=epoch_duration, preload=True)

    # Extract 1D filtered signals
    phys_signal = raw_phys.get_data(picks=[channel_name])[0]
    vlf_signal = raw_vlf.get_data(picks=[channel_name])[0]

    # Extract epochs data (n_epochs, n_channels=1, n_times) -> (n_epochs, n_times)
    epochs_data = epochs_phys.get_data(picks=[channel_name])[:, 0, :]

    return {
        'channel': channel_name,
        'raw': raw,
        'raw_phys': raw_phys,
        'raw_vlf': raw_vlf,
        'epochs_phys': epochs_phys,
        'phys_signal': phys_signal,
        'vlf_signal': vlf_signal,
        'epochs_data': epochs_data
        }








def epoch_features_from_single_channel_epochs(
    epochs: "mne.Epochs",
    *,
    channel_name: str | None = None,
    fmin: float = 0.5,
    fmax_signal: float = 30.0,
    fmax_psd: float = 35.0,
    fmin_vlf: float = 0.05,
    fmax_vlf: float = 0.30,
    n_fft_sec: float = 8.0,
    n_overlap_sec: float = 4.0,
    lf_n_fft_sec: float = 30.0,
    detrend_win_sec: float = 2.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract epoch-level features from a single-channel MNE Epochs object.

    Args:
        epochs: MNE Epochs. Must contain one EEG channel, or provide
            `channel_name` to select one.
        channel_name: Channel to use if `epochs` has multiple channels.
        fmin: Minimum frequency for PSD features (Hz).
        fmax_signal: Upper limit for bandpower fractions and entropy (Hz).
        fmax_psd: Upper limit for PSD computation (Hz).
        fmin_vlf: Minimum frequency for VLF PSD (Hz).
        fmax_vlf: Upper limit for VLF band (Hz).
        n_fft_sec: FFT length for Welch PSD (seconds).
        n_overlap_sec: Overlap for Welch PSD (seconds).
        lf_n_fft_sec: FFT length for low-frequency PSD (seconds).
        detrend_win_sec: Window length for median detrend (seconds).

    Returns:
        A tuple of:
            - df_feat: DataFrame with one row per epoch (single channel).
            - freqs: Frequency vector for the main PSD.
            - psd_lin: Linear PSD array of shape (n_epochs, n_freqs).

    Raises:
        ValueError: If the selected channel is not found or no EEG channel exists.
    """
    sf = float(epochs.info["sfreq"])

    # Pick a single channel
    if channel_name is None:
        chs = epochs.info["ch_names"]
        if len(chs) != 1:
            raise ValueError(
                f"epochs has {len(chs)} channels; provide channel_name to select one."
            )
        channel_name = chs[0]
    else:
        if channel_name not in epochs.info["ch_names"]:
            raise ValueError(f"Channel {channel_name!r} not found in epochs.")

    # X: (N, T)
    X = epochs.get_data(picks=[channel_name])[:, 0, :]
    N, T = X.shape

    eps = 1e-30

    # ---- PSD (main) ----
    psd = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax_psd,
        n_fft=int(sf * n_fft_sec),
        n_overlap=int(sf * n_overlap_sec),
        picks=[channel_name],
        average="mean",
        verbose=False,
    )
    psd_lin_3d, freqs = psd.get_data(return_freqs=True)  # (N, 1, F)
    psd_lin = psd_lin_3d[:, 0, :]  # (N, F)

    # Bandpowers
    total = integrate_bandpower(psd_lin, freqs, *EEG_SIGNAL_BAND)
    P_delta = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["delta"])
    P_theta = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["theta"])
    P_alpha = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["alpha"])
    P_sigma = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["sigma"])
    
    beta_lo, beta_hi = EEG_BANDS_HZ["beta"]
    P_beta = integrate_bandpower(psd_lin, freqs, beta_lo, min(beta_hi, fmax_signal))


    # Fractions
    frac_delta = P_delta / np.maximum(total, eps)
    frac_theta = P_theta / np.maximum(total, eps)
    frac_alpha = P_alpha / np.maximum(total, eps)
    frac_sigma = P_sigma / np.maximum(total, eps)
    frac_beta = P_beta / np.maximum(total, eps)

    # 20–35 Hz fraction
    P_20_35 = integrate_bandpower(psd_lin, freqs, 20.0, min(35.0, fmax_psd))
    frac_20_35 = P_20_35 / np.maximum(total, eps)

    # Spectral entropy (0.5–30)
    mask = (freqs >= fmin) & (freqs < fmax_signal)
    P = psd_lin[:, mask]
    Psum = np.sum(P, axis=1, keepdims=True)
    Pnorm = P / np.maximum(Psum, eps)
    spec_entropy = -np.sum(Pnorm * np.log(np.maximum(Pnorm, eps)), axis=1)
    spec_entropy = spec_entropy / np.log(Pnorm.shape[1])

    # ---- Step/jump detector (CUSUM) ----
    X0 = X - X.mean(axis=1, keepdims=True)
    cs = np.cumsum(X0, axis=1)
    max_cusum = np.max(np.abs(cs), axis=1)

    # NOTE: PSD is computed via MNE for epoch-aware processing.
    # ---- Low-freq PSD (VLF fraction) ----
    psd_lf = epochs.compute_psd(
        method="welch",
        fmin=fmin_vlf,
        fmax=4.0,
        n_fft=int(sf * lf_n_fft_sec),
        n_overlap=0,
        picks=[channel_name],
        average="mean",
        verbose=False,
    )
    psd_lf_3d, freqs_lf = psd_lf.get_data(return_freqs=True)  # (N, 1, F_lf)
    psd_lf_lin = psd_lf_3d[:, 0, :]

    P_vlf = integrate_bandpower(psd_lf_lin, freqs_lf, fmin_vlf, fmax_vlf)
    P_delta_lf = integrate_bandpower(psd_lf_lin, freqs_lf, 0.5, 4.0)
    frac_vlf = P_vlf / np.maximum(P_delta_lf, eps)

    # ---- Time-domain features ----
    absX = np.abs(X)
    max_abs = absX.max(axis=1)
    rms = np.sqrt((X**2).mean(axis=1))
    std = X.std(axis=1)

    q_hi = np.percentile(X, ROBUST_PTP_Q_HIGH, axis=1)
    q_lo = np.percentile(X, ROBUST_PTP_Q_LOW, axis=1)

    ptp_robust = q_hi - q_lo

    dx = np.diff(X, axis=1)
    line_length = np.sum(np.abs(dx), axis=1)
    diff_var = np.var(dx, axis=1)
    mean_abs_diff = np.mean(np.abs(dx), axis=1)

    kurt = kurtosis(X, axis=1, fisher=True, bias=False)

    # Sub-epoch stats (2 s windows)
    sub_win = min(int(SUB_WIN_SEC * sf), T) 
    sub_win = max(sub_win, 1) 
    
    n_sub = T // sub_win
    Xc = X[:, : n_sub * sub_win].reshape(N, n_sub, sub_win)
    
    sub_ptp = Xc.max(axis=-1) - Xc.min(axis=-1)   
    sub_rms = np.sqrt((Xc**2).mean(axis=-1)) 
    
    sub_ptp_max_2s = sub_ptp.max(axis=1)
    sub_rms_max_2s = sub_rms.max(axis=1)

    # Zero-crossing rate
    sgn = np.sign(X)
    sgn[sgn == 0] = 1
    zcr = np.mean(sgn[:, 1:] != sgn[:, :-1], axis=1)

    # Drift sensitivity
    X_det = detrend_median(X[:, None, :], sf, win_sec=detrend_win_sec)[:, 0, :]
    ptp_raw = X.max(axis=1) - X.min(axis=1)
    ptp_det = X_det.max(axis=1) - X_det.min(axis=1)
    ptp_ratio = ptp_raw / np.maximum(ptp_det, eps)

    df_feat = pd.DataFrame(
        {
            "epoch": np.arange(N),
            "channel": channel_name,
            # time-domain
            "rms": rms,
            "std": std,
            "max_abs": max_abs,
            "ptp_robust": ptp_robust,
            "line_length": line_length,
            "diff_var": diff_var,
            "kurtosis": kurt,
            "sub_ptp_max_2s": sub_ptp_max_2s,
            "sub_rms_max_2s": sub_rms_max_2s,
            # freq-domain
            "P_total": total,
            "P_delta": P_delta,
            "P_theta": P_theta,
            "P_alpha": P_alpha,
            "P_sigma": P_sigma,
            "P_beta": P_beta,
            "P_20_35": P_20_35,
            # fractions
            "frac_delta": frac_delta,
            "frac_theta": frac_theta,
            "frac_alpha": frac_alpha,
            "frac_sigma": frac_sigma,
            "frac_beta": frac_beta,
            "frac_20_35": frac_20_35,
            # spectrum shape
            "spec_entropy": spec_entropy,
            # step/drift/sweat proxies
            "max_cusum": max_cusum,
            "mean_abs_diff": mean_abs_diff,
            "frac_vlf": frac_vlf,
            "zcr": zcr,
            "ptp_raw": ptp_raw,
            "ptp_det": ptp_det,
            "ptp_ratio": ptp_ratio,
        }
    )

    return df_feat, freqs, psd_lin





def build_raw_and_epochs_from_data(
    data: Data,
    eeg_left_label: str = "EEG_L",
    eeg_right_label: str = "EEG_R",
    bipolar_label: str = "EEG_R-EEG_L",
    epoch_len: float = 30.0,
):
    """
    Convert Data(µV) -> mne.Raw with L/R + bipolar + 30s epochs.
    """
    chs = list(data.channel_names)
    for lab in (eeg_left_label, eeg_right_label):
        if lab not in chs:
            raise ValueError(f"Missing channel {lab!r} in {chs}")

    sf = data.sample_rate

    # extract in volts for MNE
    L_uv = data.array[:, chs.index(eeg_left_label)]
    R_uv = data.array[:, chs.index(eeg_right_label)]
    L_v = L_uv / 1e6
    R_v = R_uv / 1e6

    info = mne.create_info([eeg_left_label, eeg_right_label], sf, ch_types="eeg")
    raw = mne.io.RawArray(np.vstack([L_v, R_v]), info)

    # timestamps -> meas_date
    if data.timestamps is not None and len(data.timestamps):
        meas_date = pd.to_datetime(data.timestamps[0], unit="s")
        raw.set_meas_date(meas_date)

    # filters (same as your script)
    raw.notch_filter(50)
    raw.filter(0.3, 35, phase="zero-double", fir_design="firwin")

    # bipolar
    raw = mne.set_bipolar_reference(
        raw,
        anode=eeg_right_label,
        cathode=eeg_left_label,
        ch_name=bipolar_label,
        drop_refs=False,
    )

    epochs_all = mne.make_fixed_length_epochs(raw, duration=epoch_len, preload=True)

    # build X_all (N, 3, T)
    X_left = epochs_all.get_data(picks=eeg_left_label)
    X_right = epochs_all.get_data(picks=eeg_right_label)
    X_bip = epochs_all.get_data(picks=bipolar_label)

    X_all = np.stack(
        [X_left[:, 0, :], X_right[:, 0, :], X_bip[:, 0, :]],
        axis=1,
    )

    return raw, epochs_all, X_all





def data_to_eeg_df(
    data: Data,
    left_label: str | None = "EEG_L",
    right_label: str | None = "EEG_R",
) -> pd.DataFrame:
    """Convert a Data object into the EEG dataframe format used by the pipeline.

    The returned dataframe always contains the columns:
        - timestamp
        - EEG L
        - EEG R

    If a requested channel is missing, its column is filled with NaNs. At least
    one of the requested channels must exist.

    Args:
        data: Input data.
        left_label: Channel name in `data.channel_names` for left EEG.
            If None, left channel is treated as missing.
        right_label: Channel name in `data.channel_names` for right EEG.
            If None, right channel is treated as missing.

    Returns:
        A pandas DataFrame with columns ["timestamp", "EEG L", "EEG R"].

    Raises:
        ValueError: If neither left nor right channel is available.
    """
    chs = list(data.channel_names)
    n = data.array.shape[0]

    # Determine availability
    has_L = left_label is not None and left_label in chs
    has_R = right_label is not None and right_label in chs

    if not has_L and not has_R:
        raise ValueError(
            "No EEG channels found. Expected at least one of "
            f"{left_label!r} or {right_label!r} in data.channel_names, got {chs}."
        )

    # Extract channels or fill with NaN
    arr = data.array  # (n_samples, n_channels)
    eeg_L = arr[:, chs.index(left_label)] if has_L else np.full(n, np.nan, dtype=float)
    eeg_R = arr[:, chs.index(right_label)] if has_R else np.full(n, np.nan, dtype=float)

    # --- timestamps handling ---
    ts = None
    if getattr(data, "timestamps", None) is not None:
        # Try nanoseconds first (common for this repo), then fall back.
        ts = pd.to_datetime(data.timestamps, errors="coerce", unit="ns", origin="unix")
        if ts.isna().all():
            ts = pd.to_datetime(data.timestamps, errors="coerce", unit="s", origin="unix")

    if ts is None or ts.isna().all():
        # Fall back to relative time from 0 based on sample_rate.
        t_sec = np.arange(n) / float(data.sample_rate)
        ts = pd.to_datetime(t_sec, unit="s", origin="unix")

    eeg_df = pd.DataFrame(
        {
            "timestamp": ts,
            "EEG L": eeg_L,
            "EEG R": eeg_R,
        }
    )

    # Keep only rows with valid timestamps
    eeg_df = eeg_df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return eeg_df





def build_info(dfp: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Build per-epoch info strings indicating which features triggered.

    For each epoch, inspects `_bad_<feature>` columns and produces:
      - "GOOD" if no features are flagged
      - "BAD: <feature1>, <feature2>, ..." otherwise

    Args:
        dfp: DataFrame with one row per epoch containing `_bad_<feature>` columns.
        features: List of feature names corresponding to `_bad_<feature>` columns.

    Returns:
        A NumPy array of strings of shape (n_epochs,), ordered by epoch index.
    """
    dfp = dfp.sort_values("epoch")
    msgs = []
    for _, row in dfp.iterrows():
        bad_feats = [f for f in features if bool(row.get(f"_bad_{f}", False))]
        msgs.append("BAD: " + ", ".join(bad_feats) if bad_feats else "GOOD")
    return np.array(msgs, dtype=object)




    

def detect_artifacts(df,freqs,psd):
    ########### QUANTILE BASED THRESHOLDS #################
    thr_quant = quantile_thresholds_by_feature_rate(df, FEATURE_RATES)
    
    ######### MAD-BASED THRESHOLDS #################
    thr_2pass = pd.concat(
    [two_pass_mad_thresholds(df, f, k1=10, k2=8, min_keep=50) for f in FEATURES],
    ignore_index=True
    )
    
    '''
    Sanity Checks:
    two_pass_grouped=thr_2pass_L.groupby("channel")[["kept_frac_pass1","n_kept"]].describe()
    two_pass_sorted= thr_2pass_L.sort_values("kept_frac_pass1").head(10)
    '''
    
    ############### QUIET-EEG THRESHOLDS ######

    thr_quiet = pd.concat(
        [quiet_thresholds_per_channel(df, f, q_quiet=50, q_tail=99.9,
                                      include_max_abs=True, min_quiet=30, fallback="relax")
         for f in FEATURES],
        ignore_index=True
    )
    
    
    
    ############### COMPUTE FINAL STATS THRESHOLDS ######################### 
    thr_all_independent = pd.concat([thr_2pass, thr_quiet], ignore_index=True)
    thr_combo = (
        thr_all_independent
        .pivot_table(index=["channel","feature"], columns="method", values="thr", aggfunc="first")
        .reset_index()
    )
    thr_combo["thr_final"] = np.nanmax(
        np.column_stack([
            thr_combo.get("two_pass_MAD_10_8", np.nan).astype(float),
            thr_combo.get("quiet_Q99.9", np.nan).astype(float),
        ]),
        axis=1
    )
    thr_final = thr_combo[["channel","feature","thr_final"]].copy()


    ################ APPLY THRESHOLDS TO EPOCHS ######################
   
    dfp, thr_wide = add_feature_flags_single_channel(df, thr_final)
    ############### GET BAD EPOCHS ####################################
    
    dfp= get_bad_epochs(dfp) 
    
    return dfp


def run_full_zmax_artifact_pipeline_from_data(
    data: Data,
    left_label: str = "EEG_L",
    right_label: str = "EEG_R",
    sample_rate: float | None = None,
    plot_eeg: bool = True,
) -> dict:
    """Run the full rule-based artifact labeling pipeline on a Data object.

    This is a wrapper that converts a `Data` object into the EEG dataframe format,
    runs per-channel preprocessing (MNE), extracts epoch-level features, derives
    per-feature thresholds, and produces a final per-epoch artifact label for
    each available channel.

    If only one EEG channel is available, the missing channel output is returned
    as NaNs.

    Args:
        data: Input time-series data.
        left_label: Channel name in `data.channel_names` for left EEG.
        right_label: Channel name in `data.channel_names` for right EEG.
        sample_rate: Sampling rate in Hz. If None, uses `data.sample_rate`.
        plot_eeg: Whether to display the interactive sliding-window viewer.

    Returns:
        Dictionary with:
            - bad_L: Per-epoch labels for left channel (float array; NaN if missing).
            - bad_R: Per-epoch labels for right channel (float array; NaN if missing).
    """

    if sample_rate is None:
        sample_rate = float(data.sample_rate)
    
    eeg_df = data_to_eeg_df(
        data,
        left_label=left_label,
        right_label=right_label,
    )

    ################## CREATE AND PROCESS MNE RAW ####################
    processed_outputs = create_and_process_mne(eeg_df,sample_rate)
    logger.info(f"Available channels: {list(processed_outputs.keys())}")
    

    if "EEG_L" in processed_outputs:
        ################## EXTRACT EPOCH FEATURES ########################
        df_L, freqs_L, psd_L = epoch_features_from_single_channel_epochs(
            processed_outputs["EEG_L"]["epochs_phys"],
            channel_name="EEG_L",
        )
        
        ############## THRESHOLD DISCOVERY ############################
        '''
        SANITY CHECKS:
        ch_list = ['EEG_L','EEG_R']
        for ch in df_L.channel.unique():
            for f in FEATURES:
                plot_distribution_with_tails(df_L, ch, f)
                
        N = df_L[df_L.channel=="EEG_L"]["epoch"].nunique()
        print("epochs per channel:", N)
        for q in [99, 99.5, 99.9]:
            print(q, "expected flagged per feature ~", (1-q/100)*N)
        '''
        dfp_L = detect_artifacts(df_L,freqs_L,psd_L)

        
    
    if 'EEG_R' in processed_outputs:
        df_R, freqs_R, psd_R = epoch_features_from_single_channel_epochs(
            processed_outputs["EEG_R"]["epochs_phys"],
            channel_name="EEG_R",
        )
        
        dfp_R = detect_artifacts(df_R,freqs_R,psd_R)
        
    ############### PLOT THE LABELED CHANNELS ###############
    if plot_eeg:
        signals = []
        labels = []
        bad_cols = []
        info_cols = []

        # Use the time axis from whichever channel exists
        if "EEG_L" in processed_outputs:
            t = processed_outputs["EEG_L"]["raw_phys"].times
        else:
            t = processed_outputs["EEG_R"]["raw_phys"].times

        if "EEG_L" in processed_outputs:
            signals.append(processed_outputs["EEG_L"]["phys_signal"])
            labels.append("EEG_L")
            bad_cols.append(
                dfp_L.sort_values("epoch")["bad_independent"].to_numpy(dtype=bool)
            )
            info_cols.append(build_info(dfp_L, FEATURES))

        if "EEG_R" in processed_outputs:
            signals.append(processed_outputs["EEG_R"]["phys_signal"])
            labels.append("EEG_R")
            bad_cols.append(
                dfp_R.sort_values("epoch")["bad_independent"].to_numpy(dtype=bool)
            )
            info_cols.append(build_info(dfp_R, FEATURES))

        bad_mask = np.column_stack(bad_cols) if bad_cols else None
        info = np.column_stack(info_cols) if info_cols else None

        plot_eeg_arti_event(
            t,
            *signals,
            fs=sample_rate,
            labels=labels,
            win_sec=30,
            bad_mask=bad_mask,
            info=info,
        )

    ############### BUILD OUTPUT ARRAYS (NaNs for missing channel) ###############
    if "EEG_L" in processed_outputs:
        bad_L = dfp_L.sort_values("epoch")["bad_independent"].to_numpy(dtype=float)
        n_epochs = bad_L.shape[0]
    else:
        bad_L = None

    if "EEG_R" in processed_outputs:
        bad_R = dfp_R.sort_values("epoch")["bad_independent"].to_numpy(dtype=float)
        n_epochs = bad_R.shape[0] if bad_L is None else n_epochs
    else:
        bad_R = None

    # If only one channel exists, fill the other with NaNs of matching length
    if bad_L is None:
        bad_L = np.full(n_epochs, np.nan, dtype=float)
    if bad_R is None:
        bad_R = np.full(n_epochs, np.nan, dtype=float)

    # If both exist, enforce alignment
    if not np.isnan(bad_L).all() and not np.isnan(bad_R).all():
        if bad_L.shape[0] != bad_R.shape[0]:
            raise ValueError(
                f"Epoch count mismatch: L={bad_L.shape[0]} R={bad_R.shape[0]}"
            )

    return {"bad_L": bad_L, "bad_R": bad_R}

