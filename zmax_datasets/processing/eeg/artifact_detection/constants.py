
import numpy as np

'''
Dictionary:
    
THR: Threshold
DIFF: Difference
PTP: Point-to-point
WIN: Window
SEC: Seconds

'''

FLAT_PTP_THRESHOLD = 2e-6,
DIFF_RMS_THR_V = 0.05e-6      # 0.05 µV RMS of first difference (tune)
PTP_THR_V      = 0.30e-6      # 0.30 µV PTP in 2s (tune)
UNIQUE_THR     = 8            # <= 8 unique values in 2s (tune)

ROBUST_PTP_Q_HIGH = 99.5
ROBUST_PTP_Q_LOW = 0.5
SUB_WIN_SEC = 2.0
READMIT_DELTA = 0.65
READMIT_BETA = 0.1
PTP_RATIO=1.4

FRACTION_VLF= 1.5
PTP_RATIO=1.4
ZERO_CROSSING_RATE=0.02

SIGMA= 1.4826
QUIET_RELAXATION_QUANTILES: tuple[float, ...] = (60.0, 70.0, 80.0)
DEFAULT_DETREND_WINDOW_SEC = 2.0

F_MIN = 0.5
FMAX_SIGNAL = 30.0
FMAX_PSD = 35.0
FMIN_VLF = 0.05
FMAX_VLF =0.30
N_FFT_SEC = 8.0
N_OVERLAP_SEC=4.0
LF_N_FFT_SEC = 30.0
DETREND_WIN_SEC =2.0
EPS = np.finfo(np.float32).eps


QUIET_QUANTILE=50.0
TAIL_QUANTILE = 99.9
MIN_QUIET_EPOCHS=30


K1 = 10.0
K2 = 8.0
MIN_KEEP=50
MIN_FRAC_KEEP=0.20
KEEP_Q_FALLBACK=0.8


EPOCH_DURATION = 30
NOTCH_FREQ = 50
PHYS_BAND = (0.3,35.0)
VLF_BAND = (0.05, 35.0)


FEATURES = ["sub_ptp_max_2s", 
            "ptp_robust", 
            "max_abs", 
            "mean_abs_diff", 
            "max_cusum"]

GUARD_COLUMNS = ["frac_delta", "frac_beta", "frac_vlf", "ptp_ratio", "zcr"]

# Percentile rates from the distribution plots 
FEATURE_RATES = {
"max_abs": 0.05,
"ptp_robust": 0.05,
"sub_ptp_max_2s": 0.05,
"mean_abs_diff": 0.02,
"max_cusum": 0.02,
}
