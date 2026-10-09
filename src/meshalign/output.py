"""Histogram and CSV writers for surface-error results. File writers only."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .config import UM_PER_MM


def save_surf_histogram(errors_um, max_abs_err_um, hist_path):
    """Vertex-wise signed-error histogram.

    Shows the mean signed error (bias) as a dashed line and the reported Surface
    DFT metric (MAE = mean |error|) as a text box.
    """
    mae_um = float(np.mean(np.abs(errors_um)))
    mean_signed_um = float(np.mean(errors_um))

    fig, ax = plt.subplots()
    ax.hist(errors_um, bins=100)
    ax.set_xlabel("Error (µm)")
    ax.set_ylabel("Count")
    ax.set_title("Vertex-wise error distribution (µm)")
    ax.set_xlim(-max_abs_err_um, max_abs_err_um)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="major", width=1.4)
    ax.tick_params(axis="both", which="minor", width=0.8)
    for tick in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        tick.set_fontweight("bold")
    ax.axvline(0, color="#444", lw=1.5)
    ax.axvline(mean_signed_um, color="#C44E52", lw=1.5, linestyle="--",
               label=f"Bias = {mean_signed_um:+.1f} µm")
    ax.legend(fontsize=10, loc="upper left")
    ax.text(0.97, 0.95, f"Surface DFT (MAE)\n{mae_um:.0f} µm",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#333", linewidth=1.2))
    fig.tight_layout()
    fig.savefig(hist_path, dpi=300)
    plt.close(fig)


def save_dft_csvs(out_dir, prefix, dft_table, errors_mm, delta_xyz_mm, save_per_sample):
    """Write the per-sample dumps + the directional DFT summary CSV.

    ``save_per_sample=False`` skips the large per-vertex ``_surf_errors_mm`` and
    ``_surf_dft_xyz_um`` .npy/.csv; the summary ``_surf_dft_um.csv`` is always written.
    """
    if save_per_sample and delta_xyz_mm is None:
        delta_xyz_mm = np.zeros((len(errors_mm), 3))
    if save_per_sample:
        np.save(os.path.join(out_dir, f"{prefix}_surf_errors_mm.npy"), errors_mm)
        np.savetxt(
            os.path.join(out_dir, f"{prefix}_surf_errors_mm.csv"),
            errors_mm, delimiter=",", header="signed_error_mm", comments="",
        )
        delta_xyz_um = delta_xyz_mm * UM_PER_MM
        np.save(os.path.join(out_dir, f"{prefix}_surf_dft_xyz_um.npy"), delta_xyz_um)
        np.savetxt(
            os.path.join(out_dir, f"{prefix}_surf_dft_xyz_um.csv"),
            delta_xyz_um, delimiter=",", header="x_error_um,y_error_um,z_error_um", comments="",
        )

    dft_csv_path = os.path.join(out_dir, f"{prefix}_surf_dft_um.csv")
    dft_array = np.array(
        [[row["direction"], row["mean_signed_dft_um"], row["mean_abs_dft_um"], row["std_dft_um"]]
         for row in dft_table],
        dtype=object,
    )
    np.savetxt(
        dft_csv_path, dft_array, delimiter=",", fmt="%s",
        header="direction,mean_signed_dft_um,mean_abs_directional_error_um,std_directional_error_um",
        comments="",
    )
    print(f"Directional DFT table saved to: {dft_csv_path}")


def save_error_outputs(ref_mesh, errors_mm, delta_xyz_mm, max_abs_err_mm, dft_table,
                       out_dir, prefix, histogram=False, save_per_sample=True):
    """Write the DFT summary CSV(s) (always) and the histogram PNG (on request)."""
    os.makedirs(out_dir, exist_ok=True)

    errors_um = errors_mm * UM_PER_MM
    max_abs_err_um = max_abs_err_mm * UM_PER_MM

    mae_mm = np.mean(np.abs(errors_mm))
    mae_um = mae_mm * UM_PER_MM

    print("\nSigned surface-normal error stats:")
    print(f"  min:          {errors_mm.min():.6f} mm ({errors_um.min():.2f} µm)")
    print(f"  max:          {errors_mm.max():.6f} mm ({errors_um.max():.2f} µm)")
    print(f"  mean |error|: {mae_mm:.6f} mm ({mae_um:.2f} µm)")
    print(
        f"  95% |error|:  {np.percentile(np.abs(errors_mm), 95):.6f} mm "
        f"({np.percentile(np.abs(errors_um), 95):.2f} µm)"
    )

    print("\nDirectional DFT from average surface displacement components:")
    for row in dft_table:
        print(
            f"  {row['direction']}: "
            f"{row['mean_signed_dft_um']:.2f} µm "
            f"(mean |{row['direction']} error| = {row['mean_abs_dft_um']:.2f} µm)"
        )

    print(f"\nSaving outputs to: {out_dir}")

    save_dft_csvs(out_dir, prefix, dft_table, errors_mm, delta_xyz_mm, save_per_sample)

    if histogram:
        hist_path = os.path.join(out_dir, f"{prefix}_surf_histogram.png")
        save_surf_histogram(errors_um, max_abs_err_um, hist_path)
        print(f"Histogram saved to: {hist_path}")
