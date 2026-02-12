
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yasa
from scipy.signal import spectrogram

from zmax_datasets.processing.eeg.artifact_detection.artifact_detection import (
    run_full_zmax_artifact_pipeline_from_data,
)
from zmax_datasets.processing.eeg.artifact_detection.constants import EPOCH_DURATION
from zmax_datasets.processing.eeg.artifact_detection.visualizations import plot_eeg
from zmax_datasets.transforms.eeg import EEGArtifactDetection
from zmax_datasets.utils.data import Data


def test_artifact_transform_output(
    data: Data,
    transform: EEGArtifactDetection,
    *,
    win_sec: float = EPOCH_DURATION,
) -> None:
    """
    Quick test harness for EEGArtifactDetection transform.

    - checks output shape
    - checks values are binary
    - checks epoch count is consistent
    - optionally visualizes first channel with plot_eeg_artifacts
    """
    # 1) Run transform
    out_data = transform(data)

    # 2) Basic checks on returned Data
    assert isinstance(out_data, Data)
    assert out_data.array.ndim == 2, f"Expected 2D array, got {out_data.array.ndim}D"
    n_epochs_out, n_ch_out = out_data.array.shape

    # Output should have same number of channels as input.
    # (Current transform does that.)

    assert n_ch_out == data.n_channels, (n_ch_out, data.n_channels)

    # Values should be binary-ish (0/1)
    unique_vals = np.unique(out_data.array[~np.isnan(out_data.array)])
    assert np.all(np.isin(unique_vals, [0.0, 1.0])), (
        f"Non-binary values found: {unique_vals}"
        )

    print(f"✅ Transform output OK: shape={out_data.array.shape}, unique={unique_vals}")

    # 3) Independently re-run test pipeline per channel to validate "info" length
    # (since transform currently discards info)
    for ch_idx, ch_name in enumerate(data.channel_names):
        ch_data = Data(
            array=data.array[:, ch_idx:ch_idx+1],
            sample_rate=float(data.sample_rate),
            channel_names=[ch_name],
            timestamps=data.timestamps,
        )
        res = run_full_zmax_artifact_pipeline_from_data(ch_data)
        bad = res["artifact_epochs"]
        info = res["info"]

        assert bad.ndim == 1
        assert info.ndim == 1
        assert bad.shape[0] == info.shape[0], f"{ch_name}: bad and info length mismatch"
        assert bad.shape[0] == n_epochs_out, (
                f"{ch_name}: epoch count mismatch vs transform output"
            )

        print(f"✅ {ch_name}: epochs={bad.size}, bad={bad.sum():.0f}")

   
def test_artifact_transform_visualization(
        data: Data,
        *,
        win_sec: float = EPOCH_DURATION,
        plot: bool = False,
    ) -> None:

        # 4) Optional visualization (first channel only)
        if plot and data.n_channels >= 1:
            # build raw t + signal for first channel
            fs = float(data.sample_rate)
            signal = data.array[:, 0]
            t = np.arange(signal.size) / fs
    
            # rerun pipeline for first channel to get info + bad_mask
            ch_name = data.channel_names[0]
            ch_data = Data(
                array=data.array[:, 0:1],
                sample_rate=fs,
                channel_names=[ch_name],
                timestamps=data.timestamps,
            )
            res = run_full_zmax_artifact_pipeline_from_data(ch_data)
            bad_mask = res["artifact_epochs"].astype(bool)
            info = res["info"]
            
            fig, slider = plot_eeg(
                t,
                signal,
                fs=fs,
                win_sec=win_sec,
                labels=[ch_name],
                bad_mask=bad_mask,
                info=info,
            )
            
            return fig, slider



def build_epoch_timeline(
        n_epochs: int, *, win_sec: float = 30.0, t0: float = 0.0
    ) -> pd.DataFrame:

    e = np.arange(n_epochs)
    return pd.DataFrame(
        {
            "epoch": e,
            "t_start": t0 + e * win_sec,
            "t_end": t0 + (e + 1) * win_sec,
        }
    )


def bad_spans_from_timeline(
        timeline: pd.DataFrame, 
        col: str, 
        t0: float, t1: float
    ) -> list[tuple[float, float]]:
    """Return [(start, end), ...] within [t0,t1] where timeline[col] is True."""
    sub = timeline[(timeline.t_end > t0) & (timeline.t_start < t1)].sort_values(
    "t_start"
    )

    spans: list[tuple[float, float]] = []
    on = False
    s0: float | None = None

    for _, r in sub.iterrows():
        bad = bool(r[col])
        if bad and not on:
            on = True
            s0 = float(r["t_start"])
        if on and not bad:
            spans.append((float(s0), float(r["t_start"])))
            on = False

    if on and s0 is not None:
        spans.append((float(s0), float(sub["t_end"].max())))

    return spans


def plot_window_spec_with_ind(
    x: np.ndarray,
    sf: float,
    timeline: pd.DataFrame,
    ind_col: str,
    *,
    start_min: float = 0.0,
    dur_min: float = 10.0,
    fmax: float = 35.0,
    nperseg_sec: float = 10.0,
    overlap_frac: float = 0.9,
    title: str = "",
):
    t0 = start_min * 60.0
    t1 = t0 + dur_min * 60.0

    i0 = int(t0 * sf)
    i1 = int(t1 * sf)
    xx = x[i0:i1]

    nperseg = int(nperseg_sec * sf)
    noverlap = int(overlap_frac * nperseg)

    f, tt, Sxx = spectrogram(
        xx,
        fs=sf,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
        )

    keep = f <= fmax
    f = f[keep]
    Sxx = Sxx[keep, :]
    Sdb = 10 * np.log10(Sxx + 1e-20)

    t_abs = t0 + tt
    spans = bad_spans_from_timeline(timeline, ind_col, t0, t1)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(16, 5), sharex=True,
        gridspec_kw={"height_ratios": [0.7, 4]},
    )

    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_title(title or f"{ind_col} — {start_min:.1f}–{(start_min+dur_min):.1f} min")
    for a, b in spans:
        ax0.axvspan(a, b, color="red", alpha=0.85, lw=0)
    ax0.text(t0, 0.5, "IND bad", va="center", ha="left")

    im = ax1.pcolormesh(t_abs, f, Sdb, shading="auto")
    ax1.set_ylabel("Hz")
    ax1.set_xlabel("Time (s)")
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Power (dB)")

    for a, b in spans:
        ax1.axvspan(a, b, color="red", alpha=0.25, lw=0)

    plt.tight_layout()
    plt.show()
    return fig


def plot_window_vlf_spec_with_ind(
    x: np.ndarray,
    sf: float,
    timeline: pd.DataFrame,
    ind_col: str,
    *,
    start_min: float = 0.0,
    dur_min: float = 10.0,
    fmax: float = 2.0,
    nperseg_sec: float = 30.0,
    overlap_frac: float = 0.9,
    title: str = "",
):
    t0 = start_min * 60.0
    t1 = t0 + dur_min * 60.0

    i0 = int(t0 * sf)
    i1 = int(t1 * sf)
    xx = x[i0:i1]

    nperseg = int(nperseg_sec * sf)
    noverlap = int(overlap_frac * nperseg)

    f, tt, Sxx = spectrogram(
        xx,
        fs=sf,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )

    keep = (f >= 0.01) & (f <= fmax)
    f = f[keep]
    Sxx = Sxx[keep, :]
    Sdb = 10 * np.log10(Sxx + 1e-20)

    t_abs = t0 + tt
    spans = bad_spans_from_timeline(timeline, ind_col, t0, t1)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(16, 5), sharex=True,
        gridspec_kw={"height_ratios": [0.7, 4]},
    )

    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_title(
    title
    or f"VLF view — {ind_col} — {start_min:.1f}–{(start_min + dur_min):.1f} min"
    )

    for a, b in spans:
        ax0.axvspan(a, b, color="red", alpha=0.85, lw=0)
    ax0.text(t0, 0.5, "IND bad", va="center", ha="left")

    im = ax1.pcolormesh(t_abs, f, Sdb, shading="auto")
    ax1.set_ylabel("Hz (VLF)")
    ax1.set_xlabel("Time (s)")
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Power (dB)")

    for a, b in spans:
        ax1.axvspan(a, b, color="red", alpha=0.25, lw=0)

    plt.tight_layout()
    plt.show()
    return fig






def visualize_spectogram(data: Data):
    sample_rate = float(data.sample_rate)
    x = data.array[:, 0]  # 1D
    
    res = run_full_zmax_artifact_pipeline_from_data(data)
    bad_ind = res["artifact_epochs"].astype(bool)   # shape (n_epochs,)

    
    timeline = build_epoch_timeline(len(bad_ind), win_sec=30.0, t0=0.0)
    timeline["ind"] = bad_ind
    
    plot_window_spec_with_ind(
        x, sample_rate, timeline, "ind",
        start_min=0, dur_min=10,
        title=f"{data.channel_names[0]} — RAW spectrogram + bad overlay",
    )





def yasa_artifacts_from_data(
    data: Data,
    bad_epochs: np.ndarray,
    *,
    window_sec: float = 5.0,         # YASA window (seconds)
    epoch_sec: float = 30.0 ,         # your epoch length (seconds)
    method: str = "std",
    threshold: float = 3.0,
    n_chan_reject: int = 1,
    assume_microvolts: bool = False,
    channel_names: list[str] | None = None,
    return_info: bool = True,
) -> dict[str, Any]:
    
    
    """
    Compute YASA artifacts per channel and compare against your bad_epochs.

    Args:
        data: Data object with shape (n_samples, n_channels).
        bad_epochs: Your pipeline decisions.
            - shape (n_epochs,) for single-channel, or
            - shape (n_epochs, n_channels) for multi-channel.
            Values can be bool or 0/1.
        params: YASA + epoch parameters.
        assume_microvolts: If True, treat data.array as already in µV.
            If False, convert from V -> µV (multiply by 1e6).
        channel_names: Optional subset/order of channels to evaluate.
            Defaults to data.channel_names.
        return_info: If True, returns per-epoch strings ("BAD: YASA art_detect")
            shaped like bad_epochs (n_epochs,) or (n_epochs, n_channels).

    Returns:
        dict with:
          - yasa_bad_matrix: bool array (n_epochs, n_channels_eval)
          - comparison_df: per-channel metrics vs your bad_epochs
          - info_yasa: (optional) strings per epoch/channel for plotting
    """
    fs = float(data.sample_rate)

    # --- choose channels to evaluate ---
    all_ch = list(data.channel_names)
    eval_ch = channel_names if channel_names is not None else all_ch

    # map eval channel -> column index in data
    ch_to_idx = {name: i for i, name in enumerate(all_ch)}
    missing = [ch for ch in eval_ch if ch not in ch_to_idx]
    if missing:
        raise ValueError(f"Channels not found in data.channel_names: {missing}")

    # --- normalize bad_epochs to (n_epochs, n_eval_ch) boolean ---
    bad_arr = np.asarray(bad_epochs)
    if bad_arr.ndim == 1:
        bad_arr = bad_arr.reshape(-1, 1)
    bad_arr = bad_arr.astype(bool)

    # if bad_epochs provided for ALL channels but eval is subset, slice by eval channels
    if bad_arr.shape[1] != len(eval_ch):
        # try interpret bad_arr columns match data.channel_names
        if bad_arr.shape[1] == data.n_channels:
            bad_arr = bad_arr[:, [ch_to_idx[ch] for ch in eval_ch]]
        else:
            raise ValueError(
                f"bad_epochs has {bad_arr.shape[1]} channels but eval_ch has "
                f"{len(eval_ch)} (and data has {data.n_channels})."
            )



    n_epochs = bad_arr.shape[0]

    # --- compute YASA per-channel, convert 5s windows -> 30s epochs ---
    k = int(round(epoch_sec / window_sec))
    if abs(k * window_sec - epoch_sec) > 1e-6:
        raise ValueError("epoch_sec must be an integer multiple of window_sec.")

    yasa_bad = np.zeros((n_epochs, len(eval_ch)), dtype=bool)

    for j, ch in enumerate(eval_ch):
        sig = data.array[:, ch_to_idx[ch]].astype(float)

        # convert to µV for YASA
        sig_uv = sig if assume_microvolts else (sig * 1e6)

        art_5s, _zs = yasa.art_detect(
            sig_uv.reshape(1, -1),
            sf=fs,
            window=window_sec,
            method=method,
            threshold=threshold,
            n_chan_reject=n_chan_reject,
            verbose=False,
        )

        # art_5s is 1D length n_windows
        art_5s = np.asarray(art_5s).astype(int).ravel()

        usable_len = min(art_5s.size, n_epochs * k)
        art_5s = art_5s[:usable_len]

        # if recording shorter than expected epochs, pad with zeros
        if art_5s.size < n_epochs * k:
            art_5s = np.pad(art_5s, (0, n_epochs * k - art_5s.size), constant_values=0)

        art_5s = art_5s.reshape(n_epochs, k)
        yasa_bad[:, j] = (art_5s == 1).any(axis=1)

    # --- comparison metrics per channel ---
    rows: list[dict[str, object]] = []
    for j, _ch in enumerate(eval_ch):
        ours = bad_arr[:, j]
        theirs = yasa_bad[:, j]

        tp = int(np.sum(ours & theirs))
        tn = int(np.sum(~ours & ~theirs))
        fp = int(np.sum(~ours & theirs))
        fn = int(np.sum(ours & ~theirs))

        denom = max(tp + tn + fp + fn, 1)
        agreement = (tp + tn) / denom
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        rows.append(
            {
                "channel": ch,
                "n_epochs": int(n_epochs),
                "ours_bad": int(ours.sum()),
                "yasa_bad": int(theirs.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "agreement": float(agreement),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

    comparison_df = pd.DataFrame(rows)

    # --- optional info strings for plotting ---
    info_yasa = None
    if return_info:
        info_yasa = np.empty((n_epochs, len(eval_ch)), dtype=object)
        info_yasa[:] = ""
        for j, _ch in enumerate(eval_ch):
            for e in range(n_epochs):
                info_yasa[e, j] = "BAD: YASA art_detect" if yasa_bad[e, j] else "GOOD"
        # if single channel, collapse to 1D to match your viewer usage
        if info_yasa.shape[1] == 1:
            info_yasa = info_yasa[:, 0]

    return {
        "yasa_bad_matrix": yasa_bad,
        "comparison_df": comparison_df,
        "info_yasa": info_yasa,
    }

def validate_artifact_detection(data: Data):  
    transform = EEGArtifactDetection()
    out = transform(data)
    ### Test the output of the transform
    test_artifact_transform_output(data, transform)
    
    ### Confirm the bad indices visually
    fig, slider = test_artifact_transform_visualization(data, plot=True)
    
    ### Inspect Spectogram 
    visualize_spectogram(data)
    
    ### Compare against yasa algorithm
    res = run_full_zmax_artifact_pipeline_from_data(data)
    bad = res["artifact_epochs"].astype(bool)   # shape (n_epochs,)

    bad = out.array[:, 0].astype(bool)         # (n_epochs,)
    res_yasa = yasa_artifacts_from_data(data, bad, return_info=False)
    
    print(res_yasa["comparison_df"])
    yasa_bad = res_yasa["yasa_bad_matrix"][:, 0]    # (n_epochs,)
    print(yasa_bad)
    
    


