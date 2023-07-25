import numpy as np
import matplotlib.pyplot as plt


def plot_spike_ft_diagnostics(sweep, window=[0.4, 0.45]):
    mosaic = "aaabb\naaabb\ncccbb"
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 4), constrained_layout=True)

    # plot sweep
    axes["a"].plot(sweep.t, sweep.v, color="k")
    axes["a"].set_ylabel("Voltage (mV)")
    axes["a"].axvline(window[0], color="grey", alpha=0.5)
    axes["a"].axvline(window[1], color="grey", alpha=0.5, label="window")

    axes["b"].plot(sweep.t, sweep.v, color="k")
    axes["b"].set_ylabel("Voltage (mV)")
    axes["b"].set_xlabel("Time (s)")
    axes["b"].set_xlim(window)

    axes["c"].plot(sweep.t, sweep.i, color="k")
    axes["c"].axvline(window[0], color="grey", alpha=0.5)
    axes["c"].axvline(window[1], color="grey", alpha=0.5, label="window")
    axes["c"].set_yticks([0, np.max(sweep.i) + np.min(sweep.i)])
    axes["c"].set_ylabel("Current (pA)")
    axes["c"].set_xlabel("Time (s)")

    # plot spike features
    spike_fts = sweep._spikes_df
    fts = [
        "peak",
        "trough",
        "threshold",
        "upstroke",
        "downstroke",
        "fast_trough",
        "slow_trough",
        "adp",
    ]
    if spike_fts.size:
        for x in ["a", "b"]:
            for ft in fts:
                axes[x].scatter(
                    spike_fts[f"{ft}_t"],
                    spike_fts[f"{ft}_v"],
                    s=10,
                    label=ft,
                )
            for l, f1, f2 in zip(
                ["ahp", "adp"], ["fast_trough", "adp"], ["threshold", "fast_trough"]
            ):
                axes[x].vlines(
                    0.5 * (spike_fts[f"{f1}_t"] + spike_fts[f"{f2}_t"]),
                    spike_fts[f"{f1}_v"],
                    spike_fts[f"{f2}_v"],
                    ls="--",
                    lw=1,
                    label=l,
                )

    axes["b"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig, axes
