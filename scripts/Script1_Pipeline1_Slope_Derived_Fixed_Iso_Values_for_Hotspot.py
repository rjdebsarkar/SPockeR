import argparse
import re
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

FIELDS = ["stacking", "hydrophobic", "apbs"]

FIELD_COLORS = {
    "stacking": "green",
    "hydrophobic": "goldenrod",
    "apbs": "red",
}

FIELD_MARK_THRESHOLDS = {
    "stacking": 1.0,
    "hydrophobic": 5.0,
}

ELE_SEARCH_UP_TO = -17.5
ELE_MAX_NEGATIVE_VALUE = -0.05
ELE_TARGET_LEFT_RANK = 1
ELE_MIN_PROM_FRAC = 0.01
ELE_MIN_ABS_PROM = 0.008
ELE_MIN_PEAK_COUNT = 100
ELE_MAX_PEAK_COUNT = 10000

# -------------------------------------------------------------------
# Used ONLY by the additional fallback when the primary ELE range
# [ELE_SEARCH_UP_TO, ELE_MAX_NEGATIVE_VALUE] yields no result.
# -------------------------------------------------------------------
ELE_FALLBACK_COUNT_MIN = 100
ELE_FALLBACK_COUNT_MAX = 1000


def parse_name(path):
    name = path.name

    m = re.match(r"([^.]+)\.(.+)\.mrc$", name)
    if not m:
        return None, None

    pdb = m.group(1)
    raw_field = m.group(2).lower()

    if "stacking" in raw_field:
        field = "stacking"
    elif "hydrophobic" in raw_field:
        field = "hydrophobic"
    elif (
        raw_field == "apbs" or
        "electrostatic" in raw_field or
        raw_field.endswith(".ele") or
        ".ele." in raw_field or
        raw_field == "ele"
    ):
        field = "apbs"
    else:
        aliases = {
            "electrostatic": "apbs",
            "ele": "apbs",
            "apbs": "apbs",
            "stacking": "stacking",
            "hydrophobic": "hydrophobic",
        }
        field = aliases.get(raw_field)

    return pdb, field


def load_values(path, field):
    with mrcfile.open(path, mode="r") as mrc:
        vol = np.asarray(mrc.data, dtype=np.float32).ravel()

    vol = vol[np.isfinite(vol)]
    if field in ("stacking", "hydrophobic"):
        vol = vol[vol > 0]
    elif field == "apbs":
        vol = vol[vol < ELE_MAX_NEGATIVE_VALUE]
    return vol


def load_values_apbs_full(path):
    """
    Load ALL finite negative APBS values with no lower-bound restriction.
    Used only by the additional fallback when the primary ELE range is absent.
    """
    with mrcfile.open(path, mode="r") as mrc:
        vol = np.asarray(mrc.data, dtype=np.float32).ravel()
    vol = vol[np.isfinite(vol)]
    vol = vol[vol < 0]          # all negative values, no ELE_MAX_NEGATIVE_VALUE cut
    return vol


def choose_bins(values, field):
    n = len(values)
    if n < 100:
        return 80
    if field == "apbs":
        return 240
    return 180


def smooth_histogram(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    log_counts = np.log10(counts + 1.0)

    if len(log_counts) < 7:
        return None

    win = min(21, len(log_counts) if len(log_counts) % 2 == 1 else len(log_counts) - 1)
    if win < 5:
        return None

    smooth = savgol_filter(log_counts, window_length=win, polyorder=3)
    deriv1 = np.gradient(smooth, centers)
    return counts, centers, log_counts, smooth, deriv1


def pick_global_minimum(centers, deriv1, threshold_x):
    mask = np.isfinite(centers) & np.isfinite(deriv1) & (centers >= threshold_x)
    if not np.any(mask):
        return None

    c = centers[mask]
    d = deriv1[mask]
    idx = int(np.argmin(d))

    return {
        "x": float(c[idx]),
        "metric": float(d[idx]),
        "method": "slope_minimum",
    }


def fallback_ele_midpoint_from_count_band(c, h, s):
    band = (h >= ELE_MIN_PEAK_COUNT) & (h <= ELE_MAX_PEAK_COUNT)
    if np.any(band):
        idxs = np.where(band)[0]
        mid_idx = idxs[len(idxs) // 2]
        return {
            "x": float(c[mid_idx]),
            "metric": float(s[mid_idx]),
            "raw_count": float(h[mid_idx]),
            "method": "ele_midpoint_of_count_band_fallback",
            "peak_rank": np.nan,
            "n_candidate_peaks": 0,
            "peak_prominence": np.nan,
        }

    idx = int(np.argmax(s))
    return {
        "x": float(c[idx]),
        "metric": float(s[idx]),
        "raw_count": float(h[idx]),
        "method": "ele_globalmax_fallback_no_count_band",
        "peak_rank": np.nan,
        "n_candidate_peaks": 0,
        "peak_prominence": np.nan,
    }


def pick_ele_peak(centers, counts, smooth):
    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    smooth = np.asarray(smooth, dtype=float)

    mask = (
        np.isfinite(centers) &
        np.isfinite(counts) &
        np.isfinite(smooth) &
        (centers <= ELE_SEARCH_UP_TO)
    )
    if np.count_nonzero(mask) < 7:
        return None

    c = centers[mask]
    h = counts[mask]
    s = smooth[mask]

    order = np.argsort(c)
    c = c[order]
    h = h[order]
    s = s[order]

    dynamic_range = float(np.max(s) - np.min(s))
    if dynamic_range <= 0:
        return fallback_ele_midpoint_from_count_band(c, h, s)

    prominence = max(ELE_MIN_ABS_PROM, ELE_MIN_PROM_FRAC * dynamic_range)
    distance = max(1, len(s) // 40)

    peaks, props = find_peaks(s, prominence=prominence, distance=distance)

    if len(peaks) == 0:
        peaks, props = find_peaks(s, distance=distance)

    if len(peaks) == 0:
        return fallback_ele_midpoint_from_count_band(c, h, s)

    peak_positions = c[peaks]
    peak_heights = s[peaks]
    peak_counts = h[peaks]
    peak_prominences = props.get("prominences", np.full(len(peaks), np.nan))

    valid = np.isfinite(peak_positions) & np.isfinite(peak_heights) & np.isfinite(peak_counts)
    peak_positions = peak_positions[valid]
    peak_heights = peak_heights[valid]
    peak_counts = peak_counts[valid]
    peak_prominences = peak_prominences[valid]

    if len(peak_positions) == 0:
        return fallback_ele_midpoint_from_count_band(c, h, s)

    order_peaks = np.argsort(peak_positions)
    peak_positions = peak_positions[order_peaks]
    peak_heights = peak_heights[order_peaks]
    peak_counts = peak_counts[order_peaks]
    peak_prominences = peak_prominences[order_peaks]

    n_peaks = len(peak_positions)
    start_idx = min(max(ELE_TARGET_LEFT_RANK - 1, 0), n_peaks - 1)

    chosen_idx = None
    for i in range(start_idx, n_peaks):
        if ELE_MIN_PEAK_COUNT <= peak_counts[i] <= ELE_MAX_PEAK_COUNT:
            chosen_idx = i
            break

    if chosen_idx is not None:
        method = "ele_left_rank_with_count_band_filter"
        peak_rank = chosen_idx + 1
        return {
            "x": float(peak_positions[chosen_idx]),
            "metric": float(peak_heights[chosen_idx]),
            "raw_count": float(peak_counts[chosen_idx]),
            "method": method,
            "peak_rank": int(peak_rank),
            "n_candidate_peaks": int(n_peaks),
            "peak_prominence": float(peak_prominences[chosen_idx]) if np.isfinite(peak_prominences[chosen_idx]) else np.nan,
        }

    fallback = fallback_ele_midpoint_from_count_band(c, h, s)
    fallback["n_candidate_peaks"] = int(n_peaks)
    return fallback


# -------------------------------------------------------------------
# ADDITIONAL FALLBACK FUNCTION (new — do not modify anything above)
#
# Called ONLY when pick_ele_peak() returned None, meaning the primary
# ELE range [ELE_SEARCH_UP_TO, ELE_MAX_NEGATIVE_VALUE] = [-17.5, -0.05]
# contains no usable data for this PDB (e.g. 2KD4).
#
# Builds a histogram from the full negative APBS distribution (all
# values < 0, without the ELE_MAX_NEGATIVE_VALUE / ELE_SEARCH_UP_TO
# restriction) and picks the field value at the middle index of all
# histogram bins whose raw count falls in [100, 1000].
# -------------------------------------------------------------------
def pick_ele_fallback_from_full_distribution(values_full):
    """
    values_full : 1-D numpy array of ALL finite negative APBS voxel
                  values for this PDB (no range restriction).
    Returns a 'chosen' dict compatible with the rest of the pipeline,
    or None if there are fewer than 20 values.
    """
    vol = np.asarray(values_full, dtype=float)
    vol = vol[np.isfinite(vol) & (vol < 0)]
    if len(vol) < 20:
        return None

    bins = choose_bins(vol, "apbs")
    res = smooth_histogram(vol, bins)
    if res is None:
        return None

    counts, centers, log_counts, smooth, deriv1 = res

    c = np.asarray(centers, dtype=float)
    h = np.asarray(counts, dtype=float)
    s = np.asarray(smooth, dtype=float)

    # Pick midpoint of the 100–1000 count band across the full distribution
    band = (h >= ELE_FALLBACK_COUNT_MIN) & (h <= ELE_FALLBACK_COUNT_MAX)
    if np.any(band):
        idxs = np.where(band)[0]
        mid_idx = idxs[len(idxs) // 2]
        return {
            "x": float(c[mid_idx]),
            "metric": float(s[mid_idx]),
            "raw_count": float(h[mid_idx]),
            "method": "ele_full_dist_midpoint_100_1000_fallback",
            "peak_rank": np.nan,
            "n_candidate_peaks": 0,
            "peak_prominence": np.nan,
        }

    # Last resort: global smoothed maximum of the full distribution
    idx = int(np.argmax(s))
    return {
        "x": float(c[idx]),
        "metric": float(s[idx]),
        "raw_count": float(h[idx]),
        "method": "ele_full_dist_globalmax_last_resort",
        "peak_rank": np.nan,
        "n_candidate_peaks": 0,
        "peak_prominence": np.nan,
    }


def analyze_field(values, field, apbs_path=None):
    if len(values) < 20:
        return None

    bins = choose_bins(values, field)
    result = smooth_histogram(values, bins)
    if result is None:
        return None

    counts, centers, log_counts, smooth, deriv1 = result

    if field in ("stacking", "hydrophobic"):
        chosen = pick_global_minimum(centers, deriv1, FIELD_MARK_THRESHOLDS[field])
        threshold_x = FIELD_MARK_THRESHOLDS[field]
    else:
        chosen = pick_ele_peak(centers, counts, smooth)
        threshold_x = ELE_SEARCH_UP_TO

        # -----------------------------------------------------------------
        # ADDITIONAL CONDITION (new):
        # If the primary algorithm found nothing within the defined ELE
        # range [ELE_SEARCH_UP_TO, ELE_MAX_NEGATIVE_VALUE], load the full
        # negative APBS distribution for this PDB and pick the midpoint of
        # the 100–1000 count band as the threshold value.
        # This block is entered ONLY when chosen is None (primary failed);
        # it does not alter behaviour for any PDB where primary succeeds.
        # -----------------------------------------------------------------
        if chosen is None and apbs_path is not None:
            values_full = load_values_apbs_full(apbs_path)
            chosen = pick_ele_fallback_from_full_distribution(values_full)

    return {
        "counts": counts,
        "centers": centers,
        "log_counts": log_counts,
        "smooth": smooth,
        "deriv1": deriv1,
        "chosen": chosen,
        "threshold_x": threshold_x,
        "n": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
    }


# =====================================================================
# SINGLE-PDB DRIVER (generalized — replaces the old multi-PDB BASE loop)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script1 (Pipeline 1): Slope-derived / fixed iso-values for hotspot detection, for a single PDB."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the Fields_Pipeline1_<PDB_ID> folder containing the .mrc field files.",
    )
    parser.add_argument(
        "--pdb_id",
        required=True,
        help="PDB identifier (must match the prefix used in the .mrc filenames), e.g. 1AJU_fixed.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output folder where results for this PDB will be saved. "
             "Default: Analysis_Pipeline1_<pdb_id> in the current working directory.",
    )
    args = parser.parse_args()

    pdb = args.pdb_id
    pdb_dir = Path(args.input_dir)

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"Analysis_Pipeline1_{pdb}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_mrc = sorted(pdb_dir.glob("*.mrc"))

    files = {}
    for f in all_mrc:
        parsed_pdb, field = parse_name(f)
        if parsed_pdb is None or field not in FIELDS:
            continue
        if parsed_pdb.upper() != pdb.upper():
            continue
        if field not in files:
            files[field] = f

    if not files:
        print(f"ERROR: No matching .mrc field files found for PDB '{pdb}' in {pdb_dir}")
        return

    summary_rows = []
    iso_rows = []

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()

    panel_order = [
        ("stacking", "distribution"),
        ("stacking", "derivative"),
        ("hydrophobic", "distribution"),
        ("hydrophobic", "derivative"),
        ("apbs", "distribution"),
        (None, None),
    ]

    selected = {
        "pdb": pdb,
        "stacking_iso": np.nan,
        "hydrophobic_iso": np.nan,
        "apbs_iso": np.nan,
        "apbs_method": None,
        "apbs_peak_rank": np.nan,
        "apbs_n_candidate_peaks": np.nan,
        "apbs_raw_count": np.nan,
    }

    cached = {}
    for field in FIELDS:
        if field in files:
            values = load_values(files[field], field)
            # Pass the apbs file path so that analyze_field can trigger
            # the additional fallback if the primary ELE range is absent.
            apbs_path = files[field] if field == "apbs" else None
            cached[field] = analyze_field(values, field, apbs_path=apbs_path)
        else:
            cached[field] = None

    for ax, panel in zip(axes, panel_order):
        field, mode = panel

        if field is None:
            ax.axis("off")
            continue

        result = cached[field]
        if result is None:
            ax.set_title(f"{pdb} {field} ({mode}, insufficient data)")
            ax.axis("off")
            continue

        color = FIELD_COLORS[field]
        chosen = result["chosen"]

        if mode == "distribution":
            ax.plot(result["centers"], result["counts"] + 1, color=color, lw=2.0, label=field)
            ax.plot(result["centers"], 10 ** result["smooth"], color="gray", lw=1.3, ls="--", label="smoothed")
            ax.axvline(result["threshold_x"], color="purple", ls=":", lw=1.3, label=f"guide = {result['threshold_x']:.2f}")

            if chosen is not None:
                x0 = chosen["x"]
                y0 = np.interp(x0, result["centers"], result["counts"] + 1)
                ax.axvline(x0, color="red", ls="--", lw=1.3)
                ax.scatter([x0], [y0], color="red", s=45, zorder=5)
                ax.text(x0, y0, f"{x0:.3f}", fontsize=10, color="red", ha="left", va="bottom")

                if field == "stacking":
                    selected["stacking_iso"] = x0
                elif field == "hydrophobic":
                    selected["hydrophobic_iso"] = x0
                elif field == "apbs":
                    selected["apbs_iso"] = x0
                    selected["apbs_method"] = chosen.get("method")
                    selected["apbs_peak_rank"] = chosen.get("peak_rank", np.nan)
                    selected["apbs_n_candidate_peaks"] = chosen.get("n_candidate_peaks", np.nan)
                    selected["apbs_raw_count"] = chosen.get("raw_count", np.nan)

            ax.set_title(f"{pdb} {field} distribution")
            ax.set_xlabel("Field value")
            ax.set_ylabel("Count (+1)")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

        elif mode == "derivative":
            ax.plot(result["centers"], result["deriv1"], color="royalblue", lw=2.0, label="1st derivative")
            ax.axhline(0.0, color="gray", ls=":", lw=1.0)
            ax.axvline(result["threshold_x"], color="purple", ls=":", lw=1.3, label=f"guide = {result['threshold_x']:.2f}")

            if chosen is not None:
                x0 = chosen["x"]
                y0 = np.interp(x0, result["centers"], result["deriv1"])
                ax.axvline(x0, color="red", ls="--", lw=1.3)
                ax.scatter([x0], [y0], color="red", s=45, zorder=5)
                ax.text(x0, y0, f"{x0:.3f}", fontsize=10, color="red", ha="left", va="top")

            ax.set_title(f"{pdb} {field} derivative")
            ax.set_xlabel("Field value")
            ax.set_ylabel("d/dx log-count")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{pdb}_stk_hp_ele_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for field in FIELDS:
        result = cached[field]
        if result is None:
            continue

        row = {
            "pdb": pdb,
            "field": field,
            "n_voxels": result["n"],
            "min": result["min"],
            "max": result["max"],
            "mean": result["mean"],
            "std": result["std"],
            "q95": result["q95"],
            "q99": result["q99"],
            "guide_threshold_x": result["threshold_x"],
            "selected_iso": np.nan if result["chosen"] is None else result["chosen"]["x"],
            "selection_method": None if result["chosen"] is None else result["chosen"]["method"],
            "selection_metric": np.nan if result["chosen"] is None else result["chosen"]["metric"],
            "peak_rank": np.nan if result["chosen"] is None else result["chosen"].get("peak_rank", np.nan),
            "n_candidate_peaks": np.nan if result["chosen"] is None else result["chosen"].get("n_candidate_peaks", np.nan),
            "peak_prominence": np.nan if result["chosen"] is None else result["chosen"].get("peak_prominence", np.nan),
            "raw_count_at_peak": np.nan if result["chosen"] is None else result["chosen"].get("raw_count", np.nan),
            "source_mrc": str(files[field]),
        }
        summary_rows.append(row)

    iso_rows.append(selected)

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values(["pdb", "field"]).to_csv(
            out_dir / f"{pdb}_stk_hp_ele_distribution_summary.csv",
            index=False
        )

    pd.DataFrame([selected]).to_csv(
        out_dir / f"{pdb}_stk_hp_ele_selected_isovalues.csv",
        index=False
    )

    print(f"Script1 complete for {pdb}. Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
