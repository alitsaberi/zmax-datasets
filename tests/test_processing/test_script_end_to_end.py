"""
Created on Mon Dec 15 12:28:05 2025

@author: selaca
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:31:30 2025

@author: selaca
"""

import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pandas.api.types as pdt

from zmax_datasets.settings import ARTIFACT_DETECTION
from zmax_datasets.transforms.eeg import EEGArtifactDetection
from zmax_datasets.utils.data import Data

INT2LAB = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
LAB2INT = {v: k for k, v in INT2LAB.items()}


def read_Zmax_EEG(root):
    """Process only PPG data."""
    eeg_data = {}
    file_name = ["EEG L", "EEG R"]
    for zip_file in extract_zmax_files(root):
        base_name = os.path.basename(zip_file).split(".edf")[0]
        if base_name in file_name:
            if base_name == "EEG L":
                eeg_data[zip_file] = load_and_preprocess_edf(zip_file)
            if base_name == "EEG R":
                eeg_data[zip_file] = load_and_preprocess_edf(zip_file)
    return eeg_data


def extract_zmax_files(root):
    """Find all EDF files in buw_zmax_fulldata/*/* directories."""
    zmax_files = []
    for deviceID in os.scandir(root):
        if deviceID.is_dir() and "buw_zmax_fulldata" in deviceID.name:
            device_path = os.path.join(root, deviceID.name)
            for session_dir in os.scandir(device_path):
                if session_dir.is_dir():
                    session_path = os.path.join(device_path, session_dir.name)
                    for entry in os.scandir(session_path):
                        if entry.is_file() and entry.name.lower().endswith(".edf"):
                            zmax_files.append(os.path.join(session_path, entry.name))
    return zmax_files


def load_and_preprocess_edf(file_path, chunk_size=100_000):
    """
    Safely load EDF to DataFrame chunks.
    Skips files with zero samples or missing data.
    """
    try:
        # Step 1: open without preloading to inspect header safely
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose="ERROR")
    except Exception as e:
        print(f"[ERROR] Cannot read header: {file_path} -> {e}")
        return []

    # Empty file guard
    if getattr(raw, "n_times", 0) == 0:
        print(f"[WARN] EDF has zero samples: {file_path}")
        return []

    # Optional: ensure it actually contains PPG-like channels if you want
    # expected = {"OXY_IR", "PPG", "PLETH"}
    # if not any(k in ch for ch in raw.ch_names for k in expected):
    #     print(f"[WARN] No PPG channels found: {file_path} -> {raw.ch_names}")
    #     return []

    # Step 2: now actually load the data into memory
    try:
        raw.load_data()  # equivalent to preload=True after the fact
    except Exception as e:
        print(f"[ERROR] Failed to preload data: {file_path} -> {e}")
        return []

    # Safety: sampling freq and measurement date
    sfreq = raw.info.get("sfreq", None)
    start_time = raw.info.get("meas_date", None) or datetime.fromtimestamp(
        0
    )  # fallback epoch if None

    # Convert to DataFrame
    try:
        df = raw.to_data_frame()
    except Exception as e:
        print(f"[ERROR] Failed to convert to DataFrame: {file_path} -> {e}")
        return []

    if df.empty:
        print(f"[WARN] DataFrame is empty after conversion: {file_path}")
        return []

    # Build timestamp; df['time'] is seconds from start
    if "time" not in df.columns:
        print(f"[WARN] 'time' column missing in DataFrame: {file_path}")
        return []

    df["timestamp"] = start_time + pd.to_timedelta(df["time"], unit="s")
    df = df.drop(columns=["time"])

    # Split into chunks
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks


def _find_timestamp_col(df: pd.DataFrame):
    # common names: timestamp, time, datetime, ts
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("timestamp", "time", "datetime", "ts"):
            return c
    # fallback: try pandas datetime columns
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    raise ValueError("No timestamp-like column found.")


def _find_signal_col(df: pd.DataFrame, target_label: str):
    # e.g., target_label = "EEG L" or "EEG R"
    for c in df.columns:
        if target_label.lower() in str(c).lower():
            return c
    # fallback: pick the only numeric column if clear
    numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(
        f"No column matching '{target_label}' found and no unique numeric fallback."
    )


def _session_root(path_str: str):
    """
    Return the session directory path (string) ending at .../session_<id>
    Example:
      input:  C:\...\buw_zmax_fulldata_1\session_8\EEG L.edf
      output: C:\...\buw_zmax_fulldata_1\session_8
    """
    p = Path(path_str)
    parts = p.parts
    # find the part that looks like 'session_<digits>'
    for i, comp in enumerate(parts):
        if re.fullmatch(r"session_\d+", comp, flags=re.IGNORECASE):
            return str(Path(*parts[: i + 1]))
    # if not found, use parent
    return str(p.parent)


def combine_sessions(data_by_path: dict, asof_tolerance="10ms"):
    """
    data_by_path: { full_file_path: dataframe }
    Returns dict: { session_root_path: combined_df_with_timestamp_L_EEGL_EEGR }
    """
    # group paths by session
    by_session = {}
    for path, df in data_by_path.items():
        sess = _session_root(path)
        by_session.setdefault(sess, []).append((path, df))

    combined = {}
    for sess, items in by_session.items():
        # find L and R entries
        left_entry = next(((p, df) for p, df in items if "eeg l" in p.lower()), None)
        right_entry = next(((p, df) for p, df in items if "eeg r" in p.lower()), None)

        if left_entry is None and right_entry is None:
            # no EEG L/R here; skip
            continue

        if left_entry is None or right_entry is None:
            # only one side present; still create a single-side dataframe
            p, df = left_entry or right_entry
            ts_col = _find_timestamp_col(df)
            side = "EEG L" if left_entry else "EEG R"
            sig_col = _find_signal_col(df, side)
            out = df.rename(columns={ts_col: "timestamp", sig_col: side})[
                ["timestamp", side]
            ].copy()
            # ensure datetime index (if numeric seconds, convert)
            if not np.issubdtype(out["timestamp"].dtype, np.datetime64):
                # try parse; if numeric seconds, convert via unit="s"
                try:
                    out["timestamp"] = pd.to_datetime(
                        out["timestamp"], utc=True, errors="coerce"
                    )
                except Exception:
                    pass
            combined[sess] = out
            continue

        # Both sides present
        (pL, dfL) = left_entry
        (pR, dfR) = right_entry

        tsL = _find_timestamp_col(dfL)
        tsR = _find_timestamp_col(dfR)
        colL = _find_signal_col(dfL, "EEG L")
        colR = _find_signal_col(dfR, "EEG R")

        L = dfL[[tsL, colL]].rename(columns={tsL: "timestamp", colL: "EEG L"}).copy()
        R = dfR[[tsR, colR]].rename(columns={tsR: "timestamp", colR: "EEG R"}).copy()

        # normalize timestamps to datetime (keeping timezone-agnostic but consistent)
        def _to_dt(s):
            if np.issubdtype(s.dtype, np.datetime64):
                return s
            # try numeric seconds; if large, may be ms
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().all():
                # heuristic: epoch seconds ~1e9, ms ~1e12
                med = s_num.dropna().median()
                unit = "ms" if med > 1e11 else "s"
                return pd.to_datetime(s_num, unit=unit, utc=True)
            # else parse strings
            return pd.to_datetime(s, utc=True, errors="coerce")

        L["timestamp"] = _to_dt_utc(L["timestamp"])
        R["timestamp"] = _to_dt_utc(R["timestamp"])

        # asof-join R onto L by nearest timestamp, keep timestamp from L
        L_sorted = L.sort_values("timestamp")
        R_sorted = R.sort_values("timestamp").rename(
            columns={"timestamp": "timestamp_R"}
        )

        merged = pd.merge_asof(
            L_sorted,
            R_sorted,
            left_on="timestamp",
            right_on="timestamp_R",
            direction="nearest",
            tolerance=pd.Timedelta(asof_tolerance),
        ).drop(columns=["timestamp_R"])

        combined[sess] = merged[["timestamp", "EEG L", "EEG R"]]

    return combined


def _to_dt_utc(s: pd.Series) -> pd.Series:
    """Return a tz-aware (UTC) datetime series from mixed inputs."""
    # Already datetime?
    if pdt.is_datetime64_any_dtype(s):
        # If tz-naive → localize to UTC; if tz-aware → convert to UTC
        try:
            tz = s.dt.tz
        except Exception:
            tz = None
        return s.dt.tz_localize("UTC") if tz is None else s.dt.tz_convert("UTC")

    # Numeric epoch? (heuristic: ms vs s)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().all():
        med = s_num.median()
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(s_num, unit=unit, utc=True)

    # Fallback: parse strings
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_data(user_id, pre_id, session_id, root):
    zmax_eeg_data = read_Zmax_EEG(root)

    matches = [f for f in zmax_eeg_data.keys() if f"buw_zmax_fulldata_{pre_id}" in f]
    filtered_dict = {k: v for k, v in zmax_eeg_data.items() if k in matches}
    eeg_dfs = {k: pd.concat(v) for k, v in filtered_dict.items()}
    result = combine_sessions(eeg_dfs, asof_tolerance="20ms")
    eeg = result[
        f"C:\\Users\\selaca\\Desktop\\data_example_folder\\{user_id}\\buw_zmax_fulldata_{pre_id}\\session_{session_id}"
    ]

    span_s = (eeg["timestamp"].max() - eeg["timestamp"].min()).total_seconds()

    # check if it spans at least 30 seconds
    if span_s < 30:
        print(f"Skipping: data too short ({span_s:.2f} s)")
        return {}, []
    else:
        print(f"Data OK: {span_s:.2f} s long")
        return eeg


############## LOAD TEST DATA #############
sf = ARTIFACT_DETECTION["sampling_frequency"]  # 256

user_id = "HBU0C768628895A922A4"
pre_id = "1"
session_id = "3"
root = os.path.join(r"C:\Users\selaca\Desktop\data_example_folder", user_id)

eeg = load_data(user_id, pre_id, session_id, root)

df = eeg.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# Convert microvolts → volts
t = (df.index - df.index[0]).total_seconds().to_numpy()
left = df["EEG L"].to_numpy()
right = df["EEG R"].to_numpy()

############### RUN PIPELINE #############
# Build Data object
data = Data(
    array=np.column_stack([left, right]),
    sample_rate=sf,
    channel_names=["EEG_L", "EEG_R"],
    timestamps=t,
)

################# STAND ALONE TEST ##############
from zmax_datasets.processing.eeg_artifact_detect import (
    run_full_zmax_artifact_pipeline_from_data,
)

results = run_full_zmax_artifact_pipeline_from_data(data=data, sf=256, plot_eeg=True)
fig = results["fig_eeg"]
slider = results["slider_eeg"]

usability_df = results["usability_df"]
print(usability_df.head())


################# TRANSFORM CALL TEST #########
rb_transform = EEGArtifactDetection(plot=False)
rb_data = rb_transform(data)

print(rb_data.shape)  # (n_epochs, 3)
print(rb_data.channel_names)  # ['usability_rb', 'usability_rb_bin', 'artifact_rb']
