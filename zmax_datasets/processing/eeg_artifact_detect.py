# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:58:23 2025

@author: selaca
"""

# zmax_datasets/processing/artifact_detection.py

from __future__ import annotations

from typing import Tuple
import numpy as np
from loguru import logger
from scipy.signal import welch

import pandas as pd
import mne
import yasa
from collections import Counter
from pathlib import Path
import pandas.api.types as pdt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from scipy.signal import spectrogram
from yasa import sleep_statistics
from yasa import transition_matrix


from zmax_datasets.settings import ARTIFACT_DETECTION
from zmax_datasets.utils.data import Data

INT2LAB = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
LAB2INT = {v: k for k, v in INT2LAB.items()}



def _epochify(data: Data) -> tuple[np.ndarray, int, int, Data]:
    """
    Convert a continuous Data object into epochs.

    Returns
    -------
    array : np.ndarray
        Shape (n_epochs, n_channels, n_samples_per_epoch)
    n_epochs : int
    epoch_length : int
        Number of samples per epoch
    trimmed_data : Data
        Input data trimmed so that it contains only full epochs.
    """
    if data.sample_rate != ARTIFACT_DETECTION["sampling_frequency"]:
        raise ValueError(
            f"Data must have sample rate "
            f"{ARTIFACT_DETECTION['sampling_frequency']}, "
            f"got {data.sample_rate}"
        )

    epoch_length = int(ARTIFACT_DETECTION["epoch_duration"] * data.sample_rate)
    n_epochs = data.length // epoch_length
    samples_to_keep = n_epochs * epoch_length

    logger.info(
        f"[artifact_detection] samples={data.length}, "
        f"epoch_length={epoch_length}, n_epochs={n_epochs}, "
        f"samples_to_keep={samples_to_keep}"
    )

    if n_epochs == 0:
        raise ValueError(
            "No complete epochs found in the data",
            epoch_length=epoch_length,
            data_length=data.length,
        )

    if samples_to_keep < data.length:
        logger.info(
            f"[artifact_detection] Dropping "
            f"{data.length - samples_to_keep} samples at the end."
        )
        data = data[:samples_to_keep]

    array = data.array.reshape(n_epochs, epoch_length, data.n_channels).transpose(
        0, 2, 1
    )  # (N, C, T)
    return array, n_epochs, epoch_length, data


def integ_all(psd_lin: np.ndarray, freqs: np.ndarray, f1: float, f2: float) -> np.ndarray:
    """Integrate PSD between f1–f2 Hz over the last axis."""
    mask = (freqs >= f1) & (freqs <= f2)
    return np.trapz(psd_lin[:, :, mask], freqs[mask], axis=-1)


def detect_artifacts_rule_based(
    data: Data,
    eeg_left_label: str = "EEG_L",
    eeg_right_label: str = "EEG_R",
    bipolar_label: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Data]:
    """
    Hybrid artifact detection per epoch/channel.

    Parameters
    ----------
    data : Data
        Channels in µV. Must contain EEG_L and EEG_R (and optionally a bipolar channel).

    Returns
    -------
    bad_final : (n_epochs, n_channels) bool
        Final bad mask per epoch/channel.
    delta : (n_epochs, n_channels) float
        Fractional delta power.
    beta : (n_epochs, n_channels) float
        Fractional beta power.
    ptp_robust : (n_epochs, n_channels) float
        Robust peak-to-peak (µV).
    trimmed_data : Data
        Input data trimmed to full epochs.
    """
    # --- ensure / reorder channels ---
    chs = list(data.channel_names)
    required = [eeg_left_label, eeg_right_label]
    if bipolar_label is not None:
        required.append(bipolar_label)

    missing = [c for c in required if c not in chs]
    if missing:
        raise ValueError(f"Missing required EEG channels: {missing}, have {chs}")

    # reorder to [L, R, BIP?]
    ordered = [eeg_left_label, eeg_right_label] + ([bipolar_label] if bipolar_label else [])
    data = data[:, ordered]

    # epochify: (N, C, T) in µV
    X_all, n_epochs, epoch_len, data_trimmed = _epochify(data)
    sf = data_trimmed.sample_rate
    N, C, T = X_all.shape

    # convert µV -> V for PSD thresholds that assumed volts
    X_v = X_all / 1e6

    # ---------------- PSD + band fractions (in V²/Hz) ----------------
    freqs, psd = welch(
        X_v.reshape(N * C, T),
        fs=sf,
        nperseg=int(sf * 8),
        noverlap=int(sf * 4),
        axis=-1,
    )
    psd_lin = psd.reshape(N, C, -1)

    total = integ_all(psd_lin, freqs, 0.5, 30.0)
    delta = integ_all(psd_lin, freqs, 0.5, 4.0) / np.maximum(total, 1e-30)
    theta = integ_all(psd_lin, freqs, 4.0, 8.0) / np.maximum(total, 1e-30)
    alpha = integ_all(psd_lin, freqs, 8.0, 12.0) / np.maximum(total, 1e-30)
    sigma = integ_all(psd_lin, freqs, 12.0, 16.0) / np.maximum(total, 1e-30)
    beta = integ_all(psd_lin, freqs, 16.0, 30.0) / np.maximum(total, 1e-30)

    # ---------------- thresholds (your tuned values) ----------------
    amp_cap_uv        = 200.0     # ±200 µV
    pct_limit         = 0.01      # > 1% above cap
    robust_cap_uv     = 1600.0    # 1.6 mV
    extreme_cap_uv    = 2500.0    # for readmission guard
    line_ratio_max    = 0.10      # >10% of power in 50 Hz band
    extreme_abs_cap   = 3000.0    # hard max abs amplitude (3 mV)
    sub_win_sec       = 2.0       # 2 s windows
    sub_ptp_cap_uv    = 600.0     # 600 µV sub-epoch PTP

    # ---- 1) amplitude % rule ----
    pct_over = (np.abs(X_all) > amp_cap_uv).mean(axis=2)
    bad_amp_pct = pct_over > pct_limit

    # ---- 2) robust PTP in µV ----
    q_hi = np.percentile(X_all, 99.5, axis=2)
    q_lo = np.percentile(X_all, 0.5, axis=2)
    ptp_robust = q_hi - q_lo
    bad_ptp = ptp_robust > robust_cap_uv

    # ---- 3) line noise ratio ----
    line_band = integ_all(psd_lin, freqs, 45.0, 55.0) / np.maximum(total, 1e-30)
    bad_line = line_band > line_ratio_max

    # ---- 4) hard rules: max abs + sub-epoch PTP ----
    max_abs = np.max(np.abs(X_all), axis=2)
    bad_max_abs = max_abs > extreme_abs_cap

    sub_win = int(sub_win_sec * sf)
    if sub_win > 0 and T >= sub_win:
        n_sub = T // sub_win
        X_chunks = X_all[:, :, : n_sub * sub_win].reshape(N, C, n_sub, sub_win)
        ptp_sub = X_chunks.max(axis=-1) - X_chunks.min(axis=-1)
        bad_sub = (ptp_sub > sub_ptp_cap_uv).any(axis=2)
    else:
        bad_sub = np.zeros((N, C), dtype=bool)

    hard_bad = bad_max_abs | bad_sub
    soft_bad = bad_amp_pct | bad_ptp | bad_line

    bad_any = hard_bad | soft_bad

    # ---- 5) N3-like readmission (per channel) ----
    readmit_delta = 0.65
    readmit_beta  = 0.10
    is_n3_like = (
        (delta >= readmit_delta)
        & (beta <= readmit_beta)
        & (ptp_robust <= extreme_cap_uv)
        & ~hard_bad
    )

    bad_final = bad_any & ~is_n3_like

    logger.info(
        "[artifact_detection] kept per channel: "
        + ", ".join(
            f"{data_trimmed.channel_names[c]} {np.sum(~bad_final[:, c])}/{N}"
            for c in range(C)
        )
    )

    return bad_final, delta, beta, ptp_robust, data_trimmed


def apply_yasa_artifact_detection(
    data: Data,
    bad_final: np.ndarray,
    eeg_labels: list[str] | None = None,
    window_sec: float = 5.0,
) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Combine rule-based mask with YASA's art_detect (5-s windows -> 30-s epochs).

    Parameters
    ----------
    data : Data
        EEG channels in µV.
    bad_final : (n_epochs, n_channels) bool
        Current bad mask.
    eeg_labels : list of str
        Channel names to run YASA on (subset of data.channel_names).
        Default: all channels.
    window_sec : float
        Window length for art_detect.

    Returns
    -------
    bad_final_yasa : (n_epochs, n_channels) bool
    yasa_bad_per_channel : dict[name -> (n_epochs,) bool]
    """
    if eeg_labels is None:
        eeg_labels = list(data.channel_names)

    sf = data.sample_rate
    epoch_len = int(30 * sf)
    n_epochs = bad_final.shape[0]

    yasa_bad_per_channel: dict[str, np.ndarray] = {}
    bad_final_yasa = bad_final.copy()

    for ch_name in eeg_labels:
        if ch_name not in data.channel_names:
            continue
        ch_idx = data.channel_names.index(ch_name)

        sig_uv = data.array[:, ch_idx]  # (n_samples,)
        sig = sig_uv / 1e6              # µV -> V for YASA

        # YASA expects (n_chan, n_times)
        art_5s, _ = yasa.art_detect(
            sig.reshape(1, -1),
            sf=sf,
            window=window_sec,
            method="std",
            threshold=3,
            n_chan_reject=1,
            verbose=False,
        )

        k = int(30 // window_sec)  # number of 5s windows per 30s epoch
        usable_len = min(len(art_5s), n_epochs * k)
        art_5s = art_5s[:usable_len].reshape(-1, k)
        yasa_bad_30s = (art_5s == 1).any(axis=1)

        yasa_bad_per_channel[ch_name] = yasa_bad_30s

        L = min(len(yasa_bad_30s), bad_final_yasa.shape[0])
        bad_final_yasa[:L, ch_idx] = bad_final_yasa[:L, ch_idx] | yasa_bad_30s[:L]

    return bad_final_yasa, yasa_bad_per_channel


def fill_nan_labels(labels):
    arr = np.asarray(labels, dtype=object)
    mask = pd.isna(arr)
    arr[mask] = "Unscored"
    return arr


def majority_label(row):
    vals, counts = np.unique(row, return_counts=True)
    return vals[np.argmax(counts)]


def count_events_per_epoch(df, epoch_len, n_epochs):
    if df is None or len(df) == 0:
        return np.zeros(n_epochs, dtype=int)
    idx = np.floor(df["Start"].values / epoch_len).astype(int)
    idx = idx[(idx >= 0) & (idx < n_epochs)]
    counts = np.bincount(idx, minlength=n_epochs)
    return counts


def compute_epoch_usability(
    results_per_channel: dict,
    bad_final_yasa: np.ndarray,
    spindles_per_channel: dict,
    sw_per_channel: dict,
) -> pd.DataFrame:
    labels_L = results_per_channel["EEG_L"]["Labels"]
    labels_R = results_per_channel["EEG_R"]["Labels"]
    labels_bi = results_per_channel["EEG_R-EEG_L"]["Labels"]

    labels_L_f = fill_nan_labels(labels_L)
    labels_R_f = fill_nan_labels(labels_R)
    labels_bi_f = fill_nan_labels(labels_bi)

    labels_all = np.column_stack([labels_L_f, labels_R_f, labels_bi_f])
    labels_consensus = np.apply_along_axis(majority_label, 1, labels_all)

    epoch_len = 30.0
    n_epochs = len(labels_L)

    confidence_L = results_per_channel["EEG_L"]["Confidence"]
    confidence_R = results_per_channel["EEG_R"]["Confidence"]
    confidence_bi = results_per_channel["EEG_R-EEG_L"]["Confidence"]

    conf_all = np.column_stack([confidence_L, confidence_R, confidence_bi])

    n_bad_chan = bad_final_yasa.sum(axis=1)

    conf_mat = conf_all.astype(float)
    conf_mat[bad_final_yasa] = np.nan
    conf_epoch = np.nanmean(conf_mat, axis=1)
    conf_epoch = np.where(np.isnan(conf_epoch), 0.5, conf_epoch)

    sp_counts = sum(
        count_events_per_epoch(df, epoch_len, n_epochs)
        for df in spindles_per_channel.values()
    )
    sw_counts = sum(
        count_events_per_epoch(df, epoch_len, n_epochs)
        for df in sw_per_channel.values()
    )

    art_factor = np.where(
        n_bad_chan >= 2,
        0.0,
        np.where(n_bad_chan == 1, 0.4, 1.0),
    )

    conf_factor = np.clip((conf_epoch - 0.5) / 0.5, 0, 1)

    event_factor = np.full(n_epochs, 0.7)
    for e in range(n_epochs):
        st = labels_consensus[e]
        if st == "N2":
            event_factor[e] = np.clip(sp_counts[e] / 2.0, 0, 1) * 0.7 + 0.3
        elif st == "N3":
            event_factor[e] = np.clip(sw_counts[e] / 2.0, 0, 1) * 0.7 + 0.3

    w_art, w_conf, w_evt = 0.5, 0.3, 0.2
    usability = (
        w_art * art_factor
        + w_conf * conf_factor
        + w_evt * event_factor
    )

    epoch_idx = np.arange(n_epochs)
    usability_df = pd.DataFrame(
        {
            "epoch": epoch_idx,
            "label_L": labels_L,
            "label_R": labels_R,
            "label_BIP": labels_bi,
            "label_consensus": labels_consensus,
            "conf_L": confidence_L,
            "conf_R": confidence_R,
            "conf_BIP": confidence_bi,
            "conf_epoch": conf_epoch,
            "n_bad_chan": n_bad_chan,
            "spindles_total": sp_counts,
            "slow_waves_total": sw_counts,
            "art_factor": art_factor,
            "conf_factor": conf_factor,
            "event_factor": event_factor,
            "usability": usability,
        }
    )
    return usability_df

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
        labels = [f"Ch{i+1}" for i in range(n_ch)]

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
                        patch = ax.axvspan(
                            s0_clip, s1_clip, alpha=0.35, color="gold"
                        )
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




def plot_spindle_examples(signal_uv, sf, spindles_df,
                          n_examples=20,
                          win_sec=3.0,
                          random_state=None,
                          channel_name="EEG",
                          title_prefix="Spindle example"):
    """
    Plot zoomed windows around detected spindles for visual QC.

    Parameters
    ----------
    signal_uv : 1D array
        EEG signal in microvolts (same signal passed to yasa.spindles_detect).
    sf : float
        Sampling frequency (Hz).
    spindles_df : pandas.DataFrame
        Output of `spindles.summary()` for one channel.
        Must contain at least columns: 'Start', 'End'.
        If 'Peak' exists, it is used as the spindle center.
    n_examples : int
        Number of spindles to show (rows of subplots).
    win_sec : float
        Length of window in seconds around the spindle center.
        (e.g., 3 s → 1.5 s before and after center)
    random_state : int or None
        Seed for reproducible random sampling of spindles.
    channel_name : str
        Name used in y-axis label.
    title_prefix : str
        Text prefix for the figure title.
    """

    import pandas as pd

    if spindles_df is None or len(spindles_df) == 0:
        print("No spindles to plot.")
        return

    n_total = len(spindles_df)
    n_plot = min(n_examples, n_total)

    rng = np.random.default_rng(random_state)
    idx_sel = rng.choice(n_total, size=n_plot, replace=False)
    df_sel = spindles_df.iloc[idx_sel].sort_values("Start").reset_index(drop=True)

    n_rows = len(df_sel)
    fig, axes = plt.subplots(n_rows, 1, sharex=False,
                             figsize=(8, 2.2 * n_rows))
    if n_rows == 1:
        axes = [axes]

    n_samples = len(signal_uv)
    total_dur = n_samples / sf

    for ax, (_, row) in zip(axes, df_sel.iterrows()):
        # center of spindle: Peak if available, else midpoint of Start/End
        if "Peak" in row:
            t0 = row["Peak"]
        else:
            t0 = 0.5 * (row["Start"] + row["End"])

        # define window around the spindle
        half = win_sec / 2.0
        t_start = max(0.0, t0 - half)
        t_end   = min(total_dur, t0 + half)

        i0 = int(np.floor(t_start * sf))
        i1 = int(np.ceil(t_end * sf))
        i1 = min(i1, n_samples)

        # time axis centered on spindle (0 = spindle center)
        t_seg = np.arange(i0, i1) / sf - t0
        x = signal_uv[i0:i1]

        ax.plot(t_seg, x, linewidth=0.8)
        ax.set_ylabel(f"{channel_name}\n(µV)")

        # shade spindle interval (relative to center)
        s_rel = row["Start"] - t0
        e_rel = row["End"] - t0
        s_rel_clip = max(s_rel, t_seg[0])
        e_rel_clip = min(e_rel, t_seg[-1])
        ax.axvspan(s_rel_clip, e_rel_clip, alpha=0.3, color="gold")

        # small title line with index + duration/freq if available
        label_parts = []
        if "Duration" in row:
            label_parts.append(f"dur={row['Duration']:.2f}s")
        if "Frequency" in row:
            label_parts.append(f"f={row['Frequency']:.1f}Hz")
        if "RMS" in row:
            label_parts.append(f"RMS={row['RMS']:.1f}µV")
        extra = " | ".join(label_parts)

        ax.set_title(f"{title_prefix} {row.name}  (t≈{t0:.1f}s)  {extra}",
                     fontsize=9)

        ax.axvline(0, color="k", linestyle="--", alpha=0.4)  # spindle center
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time relative to spindle center (s)")
    fig.suptitle(f"{title_prefix}s on {channel_name}  (showing {n_plot}/{n_total})",
                 y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_eeg_result(t,filtered_L,filtered_R,filtered_BIP,bad_final_yasa,status_info,results):
    fig, slider = plot_eeg_arti_event(
    t,
    filtered_L,
    filtered_R,
    filtered_BIP,
    labels=["EEG_L", "EEG_R", "EEG_R-EEG_L"],
    win_sec=30,
    bad_mask=bad_final_yasa,
    info=status_info,
    spindles=results["spindles_per_channel"],
    slow_waves=results["sw_per_channel"],
    )




def analyze_channel(eeg_raw, channel_name, bad_epochs=None, sf=256):
    """
    Run sleep staging + spindles + slow waves + stats for a single EEG channel,
    optionally ignoring artifact epochs.
    """

    if bad_epochs is None:
        bad_epochs = np.array([], dtype=int)
    else:
        bad_epochs = np.asarray(bad_epochs, dtype=int)

    # ------------------------------------------------------------------
    # 1) Sleep staging on this single channel
    # ------------------------------------------------------------------
    scorer = yasa.SleepStaging(eeg_raw,
                          eeg_name=channel_name,
                          eog_name=None,
                          emg_name=None,
                          metadata=None)

    labels = np.array(scorer.predict())         # e.g. ['W','N2',...]
    proba  = scorer.predict_proba()
    confidence = proba.max(1)

    # Mark bad epochs as "Unscored" (for hypnogram + stats)
    labels_with_nan = labels.copy()
    labels_with_nan[bad_epochs] = np.nan

    # For statistics, drop bad epochs completely
    good_mask = ~np.isin(np.arange(len(labels)), bad_epochs)
    labels_good = labels[good_mask]

    mapping = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
    hypno_good = np.array([mapping[l] for l in labels_good])

    if len(hypno_good) < 2:
        print(f"[{channel_name}] Not enough clean epochs for hypnogram.")
        return {}

    # ------------------------------------------------------------------
    # 2) Sleep statistics & transition matrix (on clean epochs only)
    # ------------------------------------------------------------------
    ss = sleep_statistics(hypno_good, sf_hyp=1/30)
    counts, probs = transition_matrix(hypno_good)

    # ------------------------------------------------------------------
    # 3) Build df_eeg with timestamps + sleep stage per sample
    # ------------------------------------------------------------------
    data = eeg_raw.get_data(picks=[channel_name])
    eeg_values = data.squeeze()
    times = eeg_raw.times

    # meas_date from your original df
    meas_date = pd.Timestamp(eeg_raw.info['meas_date'])
    timestamps = meas_date + pd.to_timedelta(times, unit='s')
    df_eeg = pd.DataFrame({'time': timestamps, 'value': eeg_values})

    epoch_duration_sec = 30
    t0 = df_eeg['time'].iloc[0]
    df_eeg['epoch'] = ((df_eeg['time'] - t0) /
                       pd.Timedelta(seconds=1)) // epoch_duration_sec

    def map_epoch_to_stage(epoch_idx, labels_array):
        if epoch_idx < len(labels_array):
            return labels_array[int(epoch_idx)]
        else:
            return np.nan

    df_eeg['sleep_stage'] = df_eeg['epoch'].apply(
        lambda x: map_epoch_to_stage(x, labels_with_nan)
    )
    df_eeg = df_eeg.dropna(subset=['sleep_stage'])

    # ------------------------------------------------------------------
    # 4) Prepare long hypno + signal for spindles / slow waves
    # ------------------------------------------------------------------
    eeg_signal = eeg_values
    sf = eeg_raw.info['sfreq']
    eeg_signal_uv = eeg_signal * 1e6        # V → µV

    epoch_len = 30
    samples_per_epoch = int(sf * epoch_len)

    # Full hypno (including bad epochs) in numeric form
    hypno_full = np.array(
        [mapping[l] if isinstance(l, str) else -1 for l in labels_with_nan]
    )
    hypno_long = np.repeat(hypno_full, samples_per_epoch)

    # Trim to signal length
    min_len = min(len(hypno_long), len(eeg_signal_uv))
    hypno_long = hypno_long[:min_len]
    eeg_signal_uv = eeg_signal_uv[:min_len]

    # ------------------------------------------------------------------
    # 5) Spindles (only N2/N3 via include=(2,3), bad epochs => -1 & ignored)
    # ------------------------------------------------------------------
    try:
        spindles = yasa.spindles_detect(
            data=eeg_signal_uv, sf=sf, hypno=hypno_long, include=(2, 3),
            freq_sp=(11, 16),
            duration=(0.4, 2.0),
            thresh={"rel_pow": 0.12, "corr": 0.60, "rms": 1.2},
            remove_outliers=True, verbose=False
        )
        spindles_summary = spindles.summary() if spindles is not None else []
    except Exception as e:
        print(f"[{channel_name}] Spindle detection failed:", e)
        spindles, spindles_summary = None, []

    # ------------------------------------------------------------------
    # 6) Slow waves
    # ------------------------------------------------------------------
    try:
        slow_waves = yasa.sw_detect(
            data=eeg_signal_uv, sf=sf, hypno=hypno_long, include=(2, 3),
            freq_sw=(0.4, 1.2),
            dur_neg=(0.3, 1.4),dur_pos=(0.1, 1.0),
            amp_neg=(50, 400),amp_pos=(20, 250),amp_ptp=(75, 600),
            coupling=True, remove_outliers=True, verbose=False
        )
        
        slow_waves_summary = slow_waves.summary() if slow_waves is not None else []
    except Exception as e:
        print(f"[{channel_name}] Slow-wave detection failed:", e)
        slow_waves, slow_waves_summary = None, []

    # ------------------------------------------------------------------
    # 7) Build output dict for this channel
    # ------------------------------------------------------------------
    eeg_dict = {}
    eeg_dict[channel_name] = df_eeg
    eeg_dict['Labels'] = labels_with_nan
    eeg_dict['Proba'] = proba
    eeg_dict['Confidence'] = confidence
    eeg_dict['Hypno_Clean'] = hypno_good
    eeg_dict['Sleep_Statistics'] = ss
    eeg_dict['Transition_Matrix'] = {
        'Counts': counts,
        'Probs': probs
    }
    eeg_dict['Spindles'] = {
        'Spindles': spindles,
        'Spindles_Summary': spindles_summary
    }
    eeg_dict['Slow_Waves'] = {
        'Slow_Waves': slow_waves,
        'Slow_Waves_Summary': slow_waves_summary
    }

    # ------------------------------------------------------------------
    # 8) Optional: hypnogram plot for this channel
    # ------------------------------------------------------------------
    stage_to_numeric = {
        'Unscored': -1,
        'W': 0,
        'N1': 1,
        'N2': 2,
        'N3': 3,
        'R': 4
    }

    numeric_labels = [
        stage_to_numeric[s] if isinstance(s, str) else stage_to_numeric['Unscored']
        for s in labels_with_nan
    ]
    '''
    n_epochs = len(numeric_labels)
    epochs = np.arange(n_epochs)
    
    plt.figure(figsize=(10, 3))
    plt.step(epochs, numeric_labels, where='post')
    plt.xlabel('Epoch')
    plt.ylabel('Sleep Stage')
    plt.title(f'Hypnogram – {channel_name}')
    plt.yticks(list(stage_to_numeric.values()), list(stage_to_numeric.keys()))
    plt.ylim(-1.5, max(stage_to_numeric.values()) + 0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''
    return eeg_dict








def create_and_process_mne(eeg):
    sf = 256
    # Convert microvolts → volts
    L = eeg['EEG L'].to_numpy() / 1e6
    R = eeg['EEG R'].to_numpy() / 1e6
    meas_date = pd.to_datetime(eeg["timestamp"].iloc[0], utc=True)
    
    info = mne.create_info(['EEG_L','EEG_R'], sf, ch_types='eeg')
    raw = mne.io.RawArray(np.vstack([L, R]), info)
    raw.set_meas_date(meas_date.to_pydatetime())

    
    ################## PREPROCESS RAW ####################
    # Filtering (continuous)
    raw.notch_filter(50)
    raw.filter(0.3, 35, phase='zero-double', fir_design='firwin')
    
    ################## CREATE BI-POLAR REFERENCE ###########
    raw = mne.set_bipolar_reference(
    raw,
    anode='EEG_R',
    cathode='EEG_L',
    ch_name='EEG_R-EEG_L',
    drop_refs=False    # <--- keep original channels
    )
    
    
    # -- After filtering and bipolar referencing --
    filtered_L   = raw.get_data('EEG_L')[0]
    filtered_R   = raw.get_data('EEG_R')[0]
    filtered_BIP = raw.get_data('EEG_R-EEG_L')[0] 
    '''
    fig_eeg, slider = plot_eeg(
        t,
        filtered_L,
        filtered_R,
        filtered_BIP,
        labels=["Left (filtered)", "Right (filtered)", "Bipolar (filtered)"]
    )
    '''

    #################### EPOCH DATA ########################
    epochs_all = mne.make_fixed_length_epochs(raw, duration=30.0, preload=True)
    n_ep = len(epochs_all)
    sf = raw.info['sfreq']
    
    # Extract epoch data (n_epochs × n_times)
    X_bipolar = epochs_all.get_data(picks='EEG_R-EEG_L')
    X_left = epochs_all.get_data(picks='EEG_L')
    X_right = epochs_all.get_data(picks='EEG_R')
    
    # Squeeze the singleton channel dim and stack → (N, 3, T)
    X_all = np.stack(
        [
            X_left[:, 0, :],     # channel 0: left
            X_right[:, 0, :],    # channel 1: right
            X_bipolar[:, 0, :]   # channel 2: bipolar
        ],
        axis=1
    )  # shape: (n_ep, 3, n_times)
    
    return filtered_L,filtered_R,filtered_BIP,raw,epochs_all,X_all

    



def artifact_rejection_stats(epochs_all,X_all):
    sf=256
    ################### COMPUTE LINEAR PSD + BAND FRACTIONS ########## 
    # Welch PSD in linear units (V²/Hz)
    psd = epochs_all.compute_psd(method='welch', fmin=0.5, fmax=30.0,
                                 n_fft=int(sf*8), n_overlap=int(sf*4),
                                 picks='eeg', average='mean')
    
    psd_lin, freqs = psd.get_data(return_freqs=True)  # shape: (N, 3, F)

    
    
    total = integ_all(psd_lin, freqs, 0.5, 30.0)  # shape: (N, 3)
    
    delta = integ_all(psd_lin, freqs, 0.5, 4.0)  / np.maximum(total, 1e-30)
    theta = integ_all(psd_lin, freqs, 4.0, 8.0)  / np.maximum(total, 1e-30)
    alpha = integ_all(psd_lin, freqs, 8.0, 12.0) / np.maximum(total, 1e-30)
    sigma = integ_all(psd_lin, freqs, 12.0, 16.0)/ np.maximum(total, 1e-30)
    beta  = integ_all(psd_lin, freqs, 16.0, 30.0)/ np.maximum(total, 1e-30)

    
    # ---------------------------------------------------------------------
    # 5. HYBRID ARTIFACT REJECTION (channel-based, with hard + sub-epoch rules)
    # ---------------------------------------------------------------------
    # ---- thresholds (tuned) ----
    amp_cap        = 200e-6    # ±500 µV (in volts)
    pct_limit      = 0.01      # mark bad if >1% samples exceed amp_cap
    robust_cap     = 1.6e-3    # 1.6 mV ~95th percentile robust PTP
    extreme_cap    = 2.5e-3    # hard guardrail for readmission
    readmit_delta  = 0.65      # N3-like if delta ≥ 0.65
    readmit_beta   = 0.10      # and beta ≤ 0.10
    line_ratio_max = 0.10      # drop if 50 Hz line noise >10% of total power
    
    # NEW: extra hard-artifact thresholds
    extreme_abs_cap = 3000e-6  # 3000 µV hard max abs amplitude
    sub_win_sec     = 2.0      # sub-epoch length for local PTP (seconds)
    sub_ptp_cap     = 600e-6  # 1200 µV local PTP cap
    
    N, C, T = X_all.shape   # C = 3 for L, R, Bipolar
    
    # ---- 1) amplitude % rule, per channel ----
    pct_over = (np.abs(X_all) > amp_cap).mean(axis=2)      # (N, C)
    bad_amp_pct = pct_over > pct_limit                     # (N, C)
    
    # ---- 2) robust PTP, per channel ----
    q_hi = np.percentile(X_all, 99.5, axis=2)              # (N, C)
    q_lo = np.percentile(X_all, 0.5,  axis=2)              # (N, C)
    ptp_robust = q_hi - q_lo                               # (N, C)
    bad_ptp = ptp_robust > robust_cap                      # (N, C)
    
    # ---- 3) line noise ratio, per channel ----
    line_band = integ_all(psd_lin, freqs, 45.0, 55.0) / np.maximum(total, 1e-30)
    bad_line = line_band > line_ratio_max                  # (N, C)
    
    # ---- 4) HARD rules: max abs amplitude + sub-epoch PTP ----
    # 4a) hard max abs amplitude
    max_abs = np.max(np.abs(X_all), axis=2)                # (N, C)
    bad_max_abs = max_abs > extreme_abs_cap                # (N, C)
    
    # 4b) sub-epoch PTP
    sub_win = int(sub_win_sec * sf)                        # samples in sub-window
    n_sub = T // sub_win                                   # number of full sub-wins
    
    if n_sub > 0:
        X_chunks = X_all[:, :, :n_sub * sub_win] \
                     .reshape(N, C, n_sub, sub_win)        # (N, C, n_sub, sub_win)
        ptp_sub = X_chunks.max(axis=-1) - X_chunks.min(axis=-1)  # (N, C, n_sub)
        bad_sub = (ptp_sub > sub_ptp_cap).any(axis=2)            # (N, C)
    else:
        bad_sub = np.zeros((N, C), dtype=bool)
    
    hard_bad  = bad_max_abs | bad_sub                      # never rescued
    soft_bad  = bad_amp_pct | bad_ptp | bad_line
    
    # ---- 5) initial bad mask per channel ----
    bad_any = hard_bad | soft_bad                          # (N, C)
    
    # ---- 6) N3 readmission per channel ----
    # delta, beta, ptp_robust already (N, C)
    is_n3_like_chan = (
        (delta >= readmit_delta) &
        (beta  <= readmit_beta) &
        (ptp_robust <= extreme_cap) &
        ~hard_bad                                         # don't rescue "hard" bads
    )
    
    # Re-include epochs that look N3-like on that channel
    bad_final = bad_any & ~is_n3_like_chan                 # (N, C)
    
    # ---- 7) Final kept mask per channel ----
    kept_mask = ~bad_final                                 # (N, C)
    
    print("Kept epochs per channel:")
    for c in range(C):
        kept_idx_ch = np.where(kept_mask[:, c])[0]
        print(f"  Channel {c}: keep {len(kept_idx_ch)} / {N}")
    
    kept_idx_per_channel = [np.where(kept_mask[:, c])[0] for c in range(C)]
    bad_idx_per_channel  = [np.where(bad_final[:, c])[0] for c in range(C)]
    
    # ---------------------------------------------------------------------
    # 8. Build status_info strings (why BAD / why GOOD) per epoch & channel
    # ---------------------------------------------------------------------
    status_info = np.empty((N, C), dtype=object)
    
    for e in range(N):
        for c in range(C):
            reasons = []
    
            if bad_amp_pct[e, c]:
                reasons.append(
                    f"amp {pct_over[e, c]*100:.1f}% > {pct_limit*100:.0f}%"
                )
            if bad_ptp[e, c]:
                reasons.append(
                    f"PTP {ptp_robust[e, c]*1e6:.0f}µV > {robust_cap*1e6:.0f}µV"
                )
            if bad_line[e, c]:
                reasons.append(
                    f"line {line_band[e, c]*100:.1f}% > {line_ratio_max*100:.0f}%"
                )
            if bad_max_abs[e, c]:
                reasons.append(
                    f"max |amp| {max_abs[e, c]*1e6:.0f}µV > {extreme_abs_cap*1e6:.0f}µV"
                )
            if bad_sub[e, c]:
                reasons.append(
                    f"sub-PTP > {sub_ptp_cap*1e6:.0f}µV in {sub_win_sec:.0f}s window"
                )
    
            if bad_final[e, c]:
                if reasons:
                    status_info[e, c] = "BAD: " + ", ".join(reasons)
                else:
                    status_info[e, c] = "BAD (unspecified reason)"
            else:
                if bad_any[e, c] and is_n3_like_chan[e, c]:
                    status_info[e, c] = (
                        f"GOOD (rescued as N3-like: "
                        f"δ={delta[e, c]:.2f}, β={beta[e, c]:.2f})"
                    )
                else:
                    status_info[e, c] = "GOOD: within thresholds"

    return bad_final, kept_idx_per_channel, bad_idx_per_channel, status_info




def artifact_rejection_yasa(bad_final,epochs_all,X_all,raw,status_info):
    sf=256
    channel_names = ["EEG_L", "EEG_R", "EEG_R-EEG_L"]
    yasa_bad_per_channel = {}
    ep_ch_names = epochs_all.info['ch_names']   # ensures correct channel order
    n_ep = X_all.shape[0]    # number of 30s epochs
    k = 30 // 5              # number of 5s windows in 30s → 6
    
    for ch in channel_names:
    
        # extract channel in µV
        sig = raw.copy().pick_channels([ch]).get_data()[0] * 1e6  # 1D μV
    
        # YASA artifacts per 5-second window
        art_5s, zscores = yasa.art_detect(
            sig.reshape(1, -1),
            sf=sf,
            window=5,
            method='std',
            threshold=3,
            n_chan_reject=1,
            verbose=False
        )
    
        # ---- Convert 5-second YASA output into 30-second decisions ----
    
        # trim in case the last window is incomplete
        usable_len = min(len(art_5s), n_ep * k)
        art_5s = art_5s[:usable_len]
    
        # reshape into (n_ep, 6) windows
        art_5s = art_5s.reshape(-1, k)     # shape ≈ (n_ep, 6)
    
        # mark epoch bad if ANY of its 6×5-s windows is bad
        yasa_bad_30s = (art_5s == 1).any(axis=1)   # boolean (n_ep,)
    
        # store
        yasa_bad_per_channel[ch] = yasa_bad_30s

    

    # We assume epochs_all.info['ch_names'] is ['EEG_L', 'EEG_R', 'EEG_R-EEG_L']
    ep_ch_names = epochs_all.info['ch_names']    # same order as X_all / bad_final
    
    bad_final_yasa = bad_final.copy()
    
    for ch_name in channel_names:  # ["EEG_L", "EEG_R", "EEG_R-EEG_L"]
        # index of this channel in your epoch data
        ch_idx = ep_ch_names.index(ch_name)
    
        ymask = yasa_bad_per_channel[ch_name]   # 1D, len = n_epochs_yasa
        L = min(len(ymask), bad_final_yasa.shape[0])  # avoid off-by-one at the end
    
        bad_final_yasa[:L, ch_idx] = (
            bad_final_yasa[:L, ch_idx] | ymask[:L]
        )

    
    # Add YASA reason to status_info
    for ch_name in channel_names:
        ch_idx = ep_ch_names.index(ch_name)
        ymask = yasa_bad_per_channel[ch_name]
        L = min(len(ymask), status_info.shape[0])
    
        for e in np.where(ymask[:L])[0]:
            old = status_info[e, ch_idx]
            if old.startswith("GOOD"):
                status_info[e, ch_idx] = "BAD: YASA art_detect"
            else:
                status_info[e, ch_idx] = old + ", YASA art_detect"

    
    ep_ch_names = epochs_all.info['ch_names']   # ['EEG_L', 'EEG_R', 'EEG_R-EEG_L']
    bad_mask    = bad_final_yasa               # or bad_final if you don’t use YASA
    bad_idx_per_channel = {}
    
    for c, name in enumerate(ep_ch_names):
        idx = np.where(bad_mask[:, c])[0]
        bad_idx_per_channel[name] = idx
        print(f"{name}: {len(idx)} bad epochs")
        print(f"  indices: {idx}\n")
        
    return bad_final_yasa, bad_idx_per_channel, status_info





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
    left_label: str = "EEG_L",
    right_label: str = "EEG_R",
) -> pd.DataFrame:
    """
    Convert a zmax_datasets.utils.data.Data object (µV) into the
    EEG dataframe format expected by run_full_zmax_artifact_pipeline:

        columns: ['timestamp', 'EEG L', 'EEG R']

    Parameters
    ----------
    data : Data
        Must contain EEG channels with names matching left_label/right_label.
        Assumes samples are in microvolts.
    left_label, right_label : str
        Names of the left and right EEG channels in data.channel_names.

    Returns
    -------
    pd.DataFrame
    """
    chs = list(data.channel_names)
    if left_label not in chs or right_label not in chs:
        raise ValueError(
            f"Data must contain channels {left_label!r} and {right_label!r}, "
            f"got {chs}"
        )

    iL = chs.index(left_label)
    iR = chs.index(right_label)

    # data.array is (n_samples, n_channels) in µV (according to your repo)
    arr = data.array
    eeg_L = arr[:, iL]
    eeg_R = arr[:, iR]

    # --- timestamps handling ---
    if getattr(data, "timestamps", None) is not None:
        # data.timestamps could be numeric or datetime-like; be defensive
        ts = pd.to_datetime(data.timestamps, errors="coerce", unit="s", origin="unix")
    else:
        # fall back to relative time from 0, based on sample_rate
        n = arr.shape[0]
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

def run_full_zmax_artifact_pipeline_from_data(
    data: Data,
    left_label: str = "EEG_L",
    right_label: str = "EEG_R",
    sf: float | None = None,
    plot_eeg : bool = True
) -> dict:
    """
    Thin wrapper around run_full_zmax_artifact_pipeline that accepts
    a zmax Data object instead of a raw EEG dataframe.

    It preserves the original order:

      1) create_and_process_mne (build raw, epochs_all, X_all)
      2) artifact_rejection_stats
      3) artifact_rejection_yasa
      4) analyze_channel
      5) compute_epoch_usability
    """
    
    if sf is None:
        sf = float(data.sample_rate)

    eeg_df = data_to_eeg_df(
        data,
        left_label=left_label,
        right_label=right_label,
    )
    
    ################## CREATE AND PROCESS MNE RAW ####################
    sf=256
    filtered_L,filtered_R,filtered_BIP, raw, epochs_all,X_all = create_and_process_mne(eeg_df)
    
    ################### COMPUTE LINEAR PSD + BAND FRACTIONS ########## 
    bad_final, kept_idx_per_channel, bad_idx_per_channel, status_info = artifact_rejection_stats(epochs_all,X_all)
    
    #################### YASA ARTIFACT DETECT ########################
    bad_final_yasa, bad_idx_per_channel,status_info = artifact_rejection_yasa(bad_final,epochs_all,X_all,raw,status_info)
    
    ############# SLEEP STAGE AND EVENT DETECTION ################
    ep_ch_names = epochs_all.info['ch_names'] 
    results_per_channel = {}

    for ch in ep_ch_names:
        bad_epochs_ch = bad_idx_per_channel.get(ch, [])
        print(f"\n=== Analyzing channel {ch} ===")
        res = analyze_channel(raw,channel_name=ch,
                              bad_epochs=bad_epochs_ch)
        results_per_channel[ch] = res
    
    
    ############## YASA STATS ###########
    left_spindles = results_per_channel['EEG_L']['Spindles']['Spindles_Summary']
    right_spindles = results_per_channel['EEG_R']['Spindles']['Spindles_Summary']
    bipolar_spindles = results_per_channel['EEG_R-EEG_L']['Spindles']['Spindles_Summary']
    
   
    spindles_per_channel = {
    'EEG_L': left_spindles,
    'EEG_R': right_spindles,
    'EEG_R-EEG_L': bipolar_spindles,
    }
    
    left_sw = results_per_channel['EEG_L']['Slow_Waves']['Slow_Waves_Summary']
    right_sw = results_per_channel['EEG_R']['Slow_Waves']['Slow_Waves_Summary']
    bipolar_sw = results_per_channel['EEG_R-EEG_L']['Slow_Waves']['Slow_Waves_Summary']
    
   
    sw_per_channel = {
    'EEG_L': left_sw,
    'EEG_R': right_sw,
    'EEG_R-EEG_L': bipolar_sw,
    }

        
    fig_eeg = slider_eeg = None
    if plot_eeg:
        sf = raw.info["sfreq"]
        t = data.timestamps
    
        fig_eeg, slider_eeg = plot_eeg_arti_event(
            t,
            filtered_L,         # → µV
            filtered_R,
            filtered_BIP,
            labels=["EEG_L", "EEG_R", "EEG_R-EEG_L"],
            win_sec=30,
            bad_mask=bad_final_yasa,
            info=status_info,
            spindles=spindles_per_channel,
            slow_waves=sw_per_channel,
        )

    
    '''
    # Plot single spindle event
    # Example for EEG_L
    sf = 256  # or raw.info['sfreq']
    
    eeg_L_uv = results_per_channel['EEG_L']['EEG_L']['value'].to_numpy() * 1e6  # if stored in volts
    spindles_L = results_per_channel['EEG_L']['Spindles']['Spindles_Summary']
    
    plot_spindle_examples(
        signal_uv=eeg_L_uv,
        sf=sf,
        spindles_df=spindles_L,
        n_examples=20,
        win_sec=3.0,
        random_state=0,
        channel_name="EEG_L"
    )

    eeg_R_uv = results_per_channel['EEG_R']['EEG_R']['value'].to_numpy() * 1e6
    spindles_R = results_per_channel['EEG_R']['Spindles']['Spindles_Summary']
    
    plot_spindle_examples(eeg_R_uv, sf, spindles_R, n_examples=20,
                          win_sec=3.0, random_state=1, channel_name="EEG_R")
    
    eeg_BIP_uv = results_per_channel['EEG_R-EEG_L']['EEG_R-EEG_L']['value'].to_numpy() * 1e6
    spindles_BIP = results_per_channel['EEG_R-EEG_L']['Spindles']['Spindles_Summary']
    
    plot_spindle_examples(eeg_BIP_uv, sf, spindles_BIP, n_examples=20,
                          win_sec=3.0, random_state=2, channel_name="EEG_R-EEG_L")
    
    '''
    
    
    ##### USABILITY SCORE ####### 
    usability_df = compute_epoch_usability(results_per_channel,bad_final_yasa,spindles_per_channel,sw_per_channel)
    usability_df["usability_score"] = (usability_df["usability"] > 0.4).astype(int)
    usability_df["artifact"] = (usability_df["n_bad_chan"] > 0).astype(int)

    return dict(
        raw=raw,
        epochs_all=epochs_all,
        X_all=X_all,
        bad_final=bad_final,
        bad_final_yasa=bad_final_yasa,
        bad_idx_per_channel=bad_idx_per_channel,
        status_info=status_info,
        results_per_channel=results_per_channel,
        spindles_per_channel=spindles_per_channel,
        sw_per_channel=sw_per_channel,
        usability_df=usability_df,
        fig_eeg=fig_eeg,
        slider_eeg=slider_eeg,
    )



