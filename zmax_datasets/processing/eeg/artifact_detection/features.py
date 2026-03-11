
import mne
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import kurtosis
from scipy.signal import find_peaks

from zmax_datasets.processing.eeg.artifact_detection.constants import (
    DEFAULT_DETREND_WINDOW_SEC,
    DETREND_WIN_SEC,
    DIFF_RMS_THR_V,
    EPS,
    F_MIN,
    FLAT_PTP_THRESHOLD,
    FMAX_PSD,
    FMAX_SIGNAL,
    FMAX_VLF,
    FMIN_VLF,
    LF_N_FFT_SEC,
    N_FFT_SEC,
    N_OVERLAP_SEC,
    PTP_THR_V,
    ROBUST_PTP_Q_HIGH,
    ROBUST_PTP_Q_LOW,
    SUB_WIN_SEC,
    UNIQUE_THR,
)
from zmax_datasets.processing.eeg.spectral_helpers import integrate_bandpower
from zmax_datasets.settings import EEG_BANDS_HZ, EEG_SIGNAL_BAND
from zmax_datasets.utils.exceptions import (
    ChannelNotFoundError,
    InvalidFilterWindowError,
    MultipleChannelsError,
)


def _select_single_channel_name(epochs: mne.Epochs, channel_name: str | None) -> str:
    chs = epochs.info["ch_names"]
    if channel_name is None:
        if len(chs) != 1:
            raise MultipleChannelsError(len(chs))
        return chs[0]
    if channel_name not in chs:
        raise ChannelNotFoundError(channel_name)
    return channel_name


def _compute_psd_single_channel(
    epochs: mne.Epochs,
    *,
    channel_name: str,
    sample_rate: float,
    fmin: float,
    fmax_psd: float,
    n_fft_sec: float,
    n_overlap_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    psd = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax_psd,
        n_fft=int(sample_rate * n_fft_sec),
        n_overlap=int(sample_rate * n_overlap_sec),
        picks=[channel_name],
        average="mean",
        verbose=False,
    )
    psd_lin_3d, freqs = psd.get_data(return_freqs=True)  # (N, 1, F)
    return psd_lin_3d[:, 0, :], freqs  # (N, F), (F,)


def _compute_spectral_features(
    psd_lin: np.ndarray,
    freqs: np.ndarray,
    *,
    fmin: float,
    fmax_signal: float,
    fmax_psd: float,
    eps: float,
) -> dict[str, np.ndarray]:
    total = integrate_bandpower(psd_lin, freqs, *EEG_SIGNAL_BAND)
    P_delta = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["delta"])
    P_theta = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["theta"])
    P_alpha = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["alpha"])
    P_sigma = integrate_bandpower(psd_lin, freqs, *EEG_BANDS_HZ["sigma"])

    beta_lo, beta_hi = EEG_BANDS_HZ["beta"]
    P_beta = integrate_bandpower(psd_lin, freqs, beta_lo, min(beta_hi, fmax_signal))

    max_total = np.maximum(total, eps)
    frac_delta = P_delta / max_total
    frac_theta = P_theta / max_total
    frac_alpha = P_alpha / max_total
    frac_sigma = P_sigma / max_total
    frac_beta = P_beta / max_total

    P_20_35 = integrate_bandpower(psd_lin, freqs, 20.0, min(35.0, fmax_psd))
    frac_20_35 = P_20_35 / np.maximum(total, eps)

    mask = (freqs >= fmin) & (freqs < fmax_signal)
    P = psd_lin[:, mask]
    Psum = np.sum(P, axis=1, keepdims=True)
    Pnorm = P / np.maximum(Psum, eps)
    spec_entropy = -np.sum(Pnorm * np.log(np.maximum(Pnorm, eps)), axis=1)
    spec_entropy = spec_entropy / np.log(Pnorm.shape[1])

    return {
        "P_total": total,
        "P_delta": P_delta,
        "P_theta": P_theta,
        "P_alpha": P_alpha,
        "P_sigma": P_sigma,
        "P_beta": P_beta,
        "P_20_35": P_20_35,
        "frac_delta": frac_delta,
        "frac_theta": frac_theta,
        "frac_alpha": frac_alpha,
        "frac_sigma": frac_sigma,
        "frac_beta": frac_beta,
        "frac_20_35": frac_20_35,
        "spec_entropy": spec_entropy,
    }


def _compute_vlf_fraction(
    epochs: mne.Epochs,
    *,
    channel_name: str,
    sample_rate: float,
    fmin_vlf: float,
    fmax_vlf: float,
    lf_n_fft_sec: float,
    eps: float,
) -> np.ndarray:
    psd_lf = epochs.compute_psd(
        method="welch",
        fmin=fmin_vlf,
        fmax=4.0,
        n_fft=int(sample_rate * lf_n_fft_sec),
        n_overlap=0,
        picks=[channel_name],
        average="mean",
        verbose=False,
    )
    psd_lf_3d, freqs_lf = psd_lf.get_data(return_freqs=True)  # (N, 1, F_lf)
    psd_lf_lin = psd_lf_3d[:, 0, :]

    P_vlf = integrate_bandpower(psd_lf_lin, freqs_lf, fmin_vlf, fmax_vlf)
    P_delta_lf = integrate_bandpower(psd_lf_lin, freqs_lf, 0.5, 4.0)
    return P_vlf / np.maximum(P_delta_lf, eps)


def _compute_time_domain_features(
    X: np.ndarray,
    *,
    eps: float,
) -> dict[str, np.ndarray]:
    
    absX = np.abs(X)
    max_abs = absX.max(axis=1)
    rms = np.sqrt((X**2).mean(axis=1))
    std = X.std(axis=1)
    
    median_abs = np.median(absX, axis=1)
    q95_abs = np.percentile(absX, 95, axis=1)

    q_hi = np.percentile(X, ROBUST_PTP_Q_HIGH, axis=1)
    q_lo = np.percentile(X, ROBUST_PTP_Q_LOW, axis=1)
    ptp_robust = q_hi - q_lo

    dx = np.diff(X, axis=1)
    line_length = np.sum(np.abs(dx), axis=1)
    diff_var = np.var(dx, axis=1)
    mean_abs_diff = np.mean(np.abs(dx), axis=1)

    kurt = kurtosis(X, axis=1, fisher=True, bias=False)

    # step/jump (CUSUM)
    X0 = X - X.mean(axis=1, keepdims=True)
    cs = np.cumsum(X0, axis=1)
    max_cusum = np.max(np.abs(cs), axis=1)

    # ZCR
    sgn = np.sign(X)
    sgn[sgn == 0] = 1
    zcr = np.mean(sgn[:, 1:] != sgn[:, :-1], axis=1)

    return {
        "max_abs": max_abs,
        "median_abs": median_abs,
        "q95_abs": q95_abs,
        "rms": rms,
        "std": std,
        "ptp_robust": ptp_robust,
        "line_length": line_length,
        "diff_var": diff_var,
        "mean_abs_diff": mean_abs_diff,
        "kurtosis": kurt,
        "max_cusum": max_cusum,
        "zcr": zcr,
    }


def _compute_subepoch_flatline_features(
    X: np.ndarray,
    *,
    sample_rate: float,
    sub_win_sec: float,
) -> dict[str, np.ndarray]:
    N, T = X.shape
    sub_win = min(int(sub_win_sec * sample_rate), T)
    sub_win = max(sub_win, 1)
    n_sub = T // sub_win

    out = {
        "sub_ptp_max_2s": np.zeros(N, dtype=float),
        "sub_rms_max_2s": np.zeros(N, dtype=float),
        "sub_ptp_min_2s": np.zeros(N, dtype=float),
        "sub_std_min_2s": np.zeros(N, dtype=float),
        "sub_ptp_p10_2s": np.zeros(N, dtype=float),
        "sub_rms_med_2s": np.zeros(N, dtype=float),
        "sub_rms_p10_2s": np.zeros(N, dtype=float),
        "flat_frac_2s": np.zeros(N, dtype=float),
        "diff_rms_min_2s": np.zeros(N, dtype=float),
        "uniq_p10_2s": np.zeros(N, dtype=float),
        "stuck_frac_2s": np.zeros(N, dtype=float),
        "max_block_median_jump": np.zeros(N, dtype=float),
    }

    if n_sub <= 0:
        return out

    Xc = X[:, : n_sub * sub_win].reshape(N, n_sub, sub_win)

    sub_ptp = Xc.max(axis=-1) - Xc.min(axis=-1)  # (N, n_sub)
    sub_rms = np.sqrt((Xc**2).mean(axis=-1))
    sub_std = Xc.std(axis=-1)
    sub_med = np.median(Xc, axis=-1)

    out["sub_ptp_max_2s"] = sub_ptp.max(axis=1)
    out["sub_rms_max_2s"] = sub_rms.max(axis=1)
    out["sub_ptp_min_2s"] = sub_ptp.min(axis=1)
    out["sub_std_min_2s"] = sub_std.min(axis=1)
    out["sub_ptp_p10_2s"] = np.percentile(sub_ptp, 10, axis=1)
    out["sub_rms_med_2s"] = np.median(sub_rms, axis=1)
    out["sub_rms_p10_2s"] = np.percentile(sub_rms, 10, axis=1)
    out["flat_frac_2s"] = np.mean(sub_ptp < FLAT_PTP_THRESHOLD, axis=1)

    if n_sub > 1:
        block_median_jump = np.abs(np.diff(sub_med, axis=1))  # (N, n_sub-1)
        out["max_block_median_jump"] = np.max(block_median_jump, axis=1)

    dXc = np.diff(Xc, axis=-1)
    diff_rms_2s = np.sqrt(np.mean(dXc**2, axis=-1))  # (N, n_sub)
    out["diff_rms_min_2s"] = diff_rms_2s.min(axis=1)

    uniq_2s = np.empty((N, n_sub), dtype=np.int32)
    for e in range(N):
        for w in range(n_sub):
            uniq_2s[e, w] = np.unique(Xc[e, w]).size
    out["uniq_p10_2s"] = np.percentile(uniq_2s, 10, axis=1)

    stuck_win = (
        (diff_rms_2s < DIFF_RMS_THR_V)
        & (sub_ptp < PTP_THR_V)
        & (uniq_2s <= UNIQUE_THR)
    )
    out["stuck_frac_2s"] = stuck_win.mean(axis=1)

    return out


def _compute_drift_features(
    X: np.ndarray,
    *,
    sample_rate: float,
    detrend_win_sec: float,
    eps: float,
) -> dict[str, np.ndarray]:
    X_det = detrend_median_epochs(
        signal=X[:, None, :],
        sample_rate_hz=sample_rate,
        window_seconds=detrend_win_sec,
    )[:, 0, :]

    ptp_raw = X.max(axis=1) - X.min(axis=1)
    ptp_det = X_det.max(axis=1) - X_det.min(axis=1)
    ptp_ratio = ptp_raw / np.maximum(ptp_det, eps)

    return {"ptp_raw": ptp_raw, "ptp_det": ptp_det, "ptp_ratio": ptp_ratio}


def epoch_features_from_single_channel_epochs(
    epochs: mne.Epochs,
    *,
    channel_name: str | None = None,
    fmin: float = F_MIN,
    fmax_signal: float = FMAX_SIGNAL,
    fmax_psd: float = FMAX_PSD,
    fmin_vlf: float = FMIN_VLF,
    fmax_vlf: float = FMAX_VLF,
    n_fft_sec: float = N_FFT_SEC,
    n_overlap_sec: float = N_OVERLAP_SEC,
    lf_n_fft_sec: float = LF_N_FFT_SEC,
    detrend_win_sec: float = DETREND_WIN_SEC,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract epoch-level features from a single-channel MNE Epochs object."""
    sample_rate = float(epochs.info["sfreq"])
    picked_channel = _select_single_channel_name(epochs, channel_name)

    X = epochs.get_data(picks=[picked_channel])[:, 0, :]
    n_epochs, _ = X.shape
    eps = EPS

    psd_lin, freqs = _compute_psd_single_channel(
        epochs,
        channel_name=picked_channel,
        sample_rate=sample_rate,
        fmin=fmin,
        fmax_psd=fmax_psd,
        n_fft_sec=n_fft_sec,
        n_overlap_sec=n_overlap_sec,
    )
    
    harmonic = _compute_harmonic_features(
        psd_lin,
        freqs,
        fmin=fmin,
        fmax_psd=fmax_psd,
    )

    spectral = _compute_spectral_features(
        psd_lin,
        freqs,
        fmin=fmin,
        fmax_signal=fmax_signal,
        fmax_psd=fmax_psd,
        eps=eps,
    )

    fraction_vlf = _compute_vlf_fraction(
        epochs,
        channel_name=picked_channel,
        sample_rate=sample_rate,
        fmin_vlf=fmin_vlf,
        fmax_vlf=fmax_vlf,
        lf_n_fft_sec=lf_n_fft_sec,
        eps=eps,
    )

    time_dom = _compute_time_domain_features(X, eps=eps)

    subepoch = _compute_subepoch_flatline_features(
        X,
        sample_rate=sample_rate,
        sub_win_sec=SUB_WIN_SEC,
    )

    drift = _compute_drift_features(
        X,
        sample_rate=sample_rate,
        detrend_win_sec=detrend_win_sec,
        eps=eps,
    )

    df_feat = pd.DataFrame(
        {
            "epoch": np.arange(n_epochs),
            "channel": picked_channel,
            **time_dom,
            **subepoch,
            **spectral,
            "frac_vlf": fraction_vlf,
            **drift,
            **harmonic,
        }
    )

    return df_feat, freqs, psd_lin


def _compute_harmonic_features(
    psd_lin: np.ndarray,
    freqs: np.ndarray,
    *,
    fmin: float,
    fmax_psd: float,
) -> dict[str, np.ndarray]:
    """
    Compute harmonic-artifact features for a single-channel PSD array.

    Args:
        psd_lin: Linear PSD array of shape (n_epochs, n_freqs).
        freqs: Frequency axis of shape (n_freqs,).
        fmin: Minimum base frequency to consider.
        fmax_psd: Maximum PSD frequency available.

    Returns:
        Dict with harmonic_flag, harmonic_f0, harmonic_score, harmonic_n.
    """
    n_epochs = psd_lin.shape[0]

    harmonic_flag = np.zeros(n_epochs, dtype=bool)
    harmonic_f0 = np.full(n_epochs, np.nan, dtype=float)
    harmonic_score = np.zeros(n_epochs, dtype=float)
    harmonic_n = np.zeros(n_epochs, dtype=int)

    for epoch_index in range(n_epochs):
        flag, f0, score, n_h = harmonic_artifact_simple(
            psd_lin[epoch_index, :],
            freqs,
            fmin=max(4.0, fmin),
            fmax=min(30.0, fmax_psd),
            min_prom_db=8.0,
            tol_hz=0.6,
            min_h2_rel=0.20,
            min_h3_rel=0.10,
            min_total_ratio=0.20,
        )
        harmonic_flag[epoch_index] = flag
        harmonic_f0[epoch_index] = f0
        harmonic_score[epoch_index] = score
        harmonic_n[epoch_index] = n_h

    return {
        "harmonic_flag": harmonic_flag,
        "harmonic_f0": harmonic_f0,
        "harmonic_score": harmonic_score,
        "harmonic_n": harmonic_n,
    }


def detrend_median_epochs(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    window_seconds: float = DEFAULT_DETREND_WINDOW_SEC,
) -> np.ndarray:
    
    """Remove slow trends from epoched signals using a running median filter.

    Computes a running median over a sliding window (in seconds) along the
    time axis and subtracts it from the signal. Useful for removing baseline
    drift while preserving transient activity.

    Args:
        signal: Epoched signal array of shape (n_epochs, n_channels, n_samples).
        sample_rate_hz: Sampling frequency in Hz.
        window_seconds: Median filter window length in seconds.

    Returns:
        Detrended signal with the same shape as `signal`.

    Raises:
        InvalidFilterWindowError: If the computed kernel size is < 1.
        ValueError: If `signal` does not have 3 dimensions.
    """
    if signal.ndim != 3:
        raise ValueError(
            "Expected `signal` to have shape (n_epochs, n_channels, n_samples), "
            f"got {signal.shape}"
        )


    kernel_size = int(window_seconds * sample_rate_hz)
    if kernel_size < 1:
        raise InvalidFilterWindowError(kernel_size)

    # scipy.signal.medfilt requires odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # medfilt filters along each dimension; we only want time-axis filtering
    baseline = medfilt(signal, kernel_size=(1, 1, kernel_size))
    return signal - baseline


def harmonic_artifact_simple(psd_1d, freqs,
                             fmin=4.0, fmax=30.0,
                             min_prom_db=8.0,
                             tol_hz=0.6,
                             min_h2_rel=0.20,
                             min_h3_rel=0.10,
                             min_total_ratio=0.20):
    """
    Stricter harmonic detector:
    requires peaks near f0, 2*f0, 3*f0, and requires harmonics
    to carry enough power relative to the base and total band power.
    """
    out_flag = False
    out_f0 = np.nan
    out_score = 0.0
    out_n = 0

    m = (freqs >= fmin) & (freqs <= fmax)
    f = freqs[m]
    p = psd_1d[m]

    if len(f) < 10 or not np.all(np.isfinite(p)):
        return out_flag, out_f0, out_score, out_n

    p_db = 10 * np.log10(np.maximum(p, 1e-30))
    peaks, props = find_peaks(p_db, prominence=min_prom_db)
    if len(peaks) < 3:
        return out_flag, out_f0, out_score, out_n

    peak_f = f[peaks]
    peak_prom = props["prominences"]
    peak_pow = p[peaks]

    best_score = -np.inf
    best_f0 = np.nan
    best_n = 0
    best_flag = False

    total_pow = np.sum(p)

    for i, f0 in enumerate(peak_f):
        if not (4.0 <= f0 <= 10.0):
            continue

        # fundamental
        p1 = peak_pow[i]
        score = float(peak_prom[i])
        count = 1

        # find 2nd harmonic
        j2 = np.where(np.abs(peak_f - 2 * f0) <= tol_hz)[0]
        if len(j2) == 0:
            continue
        j2 = j2[np.argmax(peak_prom[j2])]
        p2 = peak_pow[j2]
        score += float(peak_prom[j2])
        count += 1

        # find 3rd harmonic
        j3 = np.where(np.abs(peak_f - 3 * f0) <= tol_hz)[0]
        if len(j3) == 0:
            continue
        j3 = j3[np.argmax(peak_prom[j3])]
        p3 = peak_pow[j3]
        score += float(peak_prom[j3])
        count += 1

        # relative harmonic strength
        h2_rel = p2 / max(p1, 1e-30)
        h3_rel = p3 / max(p1, 1e-30)
        total_ratio = (p1 + p2 + p3) / max(total_pow, 1e-30)

        flag = (
            (h2_rel >= min_h2_rel) &
            (h3_rel >= min_h3_rel) &
            (total_ratio >= min_total_ratio)
        )

        if flag and ((count > best_n) or (count == best_n and score > best_score)):
            best_flag = True
            best_n = count
            best_score = score
            best_f0 = f0

    if best_flag:
        out_flag = True
        out_f0 = float(best_f0)
        out_score = float(best_score)
        out_n = int(best_n)

    return out_flag, out_f0, out_score, out_n
