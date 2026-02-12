
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def plot_eeg(
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

    for ax, sig, lbl in zip(axes, signals, labels, strict=False):
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
            zip(lines, signals, axes, info_texts, strict=False)
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

