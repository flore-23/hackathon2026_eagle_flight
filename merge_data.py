"""
merge_data.py — Merge GPS and 3D orientation CSVs for eagle flight visualization.

Strategy:
  - GPS: linearly interpolated to fill every timestamp (no gaps)
  - Orientation: only populated where real sensor data exists, NaN elsewhere
  - Output: unified CSV with a `has_orientation` boolean for timeline coloring

Supports resolutions: 1hz (per-second means), 5hz, 20hz (expanded sub-samples)

Usage:
    python merge_data.py --gps Data/madi_loc_day1.csv \
                         --orientation Data/processed_angle_day1.csv \
                         --output Data/merged_day1.csv \
                         --resolution 1hz
"""

import argparse
import csv
import pandas as pd
import numpy as np
from pathlib import Path


def load_gps(path: str) -> pd.DataFrame:
    """Load GPS CSV, keep relevant columns, parse timestamps."""
    df = pd.read_csv(path)
    df = df.rename(columns={
        "individual_local_identifier": "bird_id",
        "height_above_ellipsoid": "altitude",
        "long": "lon",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    cols = ["bird_id", "timestamp", "lat", "lon", "altitude", "ground_speed", "heading"]
    df = df[cols].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_orientation(path: str, resolution: str = "1hz") -> pd.DataFrame:
    """
    Load orientation CSV and expand to the requested resolution.

    At 1hz  -> one row per second using roll_mean / pitch_mean / yaw_mean.
    At 5hz  -> parse 20 Hz sub-sample arrays, keep every 4th sample.
    At 20hz -> parse all sub-samples.

    Note: the geometry column contains unquoted ``c(NA, NA)`` which adds a
    stray comma.  We fix this by merging the two split fields back together
    whenever a data row has one more field than the header.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_hdr = len(header)

        # Find the geometry column index so we know where to re-join
        geom_idx = header.index("geometry") if "geometry" in header else None

        fixed_rows = []
        for row in reader:
            if len(row) == n_hdr + 1 and geom_idx is not None:
                row = row[:geom_idx] + [row[geom_idx] + "," + row[geom_idx + 1]] + row[geom_idx + 2:]
            fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header)

    # First column is 'floor_date(timestamp, "second")' — the whole-second timestamp
    floor_col = df.columns[0]
    df["second_ts"] = pd.to_datetime(df[floor_col])
    df = df.sort_values("second_ts").reset_index(drop=True)

    if resolution == "1hz":
        return pd.DataFrame({
            "timestamp": df["second_ts"],
            "roll":  df["roll_mean"].astype(float),
            "pitch": df["pitch_mean"].astype(float),
            "yaw":   df["yaw_mean"].astype(float),
            "bird_id": df["individual_local_identifier"],
        })

    # --- 5 Hz or 20 Hz: expand the per-row sub-sample arrays ---
    step = 4 if resolution == "5hz" else 1
    freq = 20  # IMU sampling frequency (always 20 for this dataset)

    records = []
    for _, row in df.iterrows():
        base_time = row["second_ts"]

        # Sub-sample arrays are space-separated floats in the _deg columns
        roll_vals  = [float(x) for x in str(row["roll_deg"]).split()]
        pitch_vals = [float(x) for x in str(row["pitch_deg"]).split()]
        yaw_vals   = [float(x) for x in str(row["yaw_deg"]).split()]

        n = min(len(roll_vals), len(pitch_vals), len(yaw_vals))

        for i in range(0, n, step):
            t = base_time + pd.Timedelta(seconds=i / freq)
            records.append({
                "timestamp": t,
                "roll":  roll_vals[i],
                "pitch": pitch_vals[i],
                "yaw":   yaw_vals[i],
                "bird_id": row["individual_local_identifier"],
            })

    return pd.DataFrame(records)


def interpolate_gps(gps_df: pd.DataFrame, target_timestamps: pd.Series) -> pd.DataFrame:
    """
    Linearly interpolate GPS values onto every target timestamp.

    Interpolates lat, lon, altitude, ground_speed, heading over time.
    Timestamps outside the GPS range are forward/back-filled from the
    nearest known GPS fix.
    """
    gps = gps_df.set_index("timestamp").sort_index()
    gps = gps[~gps.index.duplicated(keep="last")]

    target_idx = pd.DatetimeIndex(target_timestamps).sort_values().drop_duplicates()

    # Merge original GPS index with targets so interpolation has anchor points
    combined_idx = gps.index.union(target_idx)
    gps_reindexed = gps.reindex(combined_idx)

    # Interpolate numeric columns using time-aware linear interpolation
    numeric_cols = ["lat", "lon", "altitude", "ground_speed", "heading"]
    for col in numeric_cols:
        gps_reindexed[col] = gps_reindexed[col].interpolate(method="time")
        # Fill any edge NaNs (timestamps before first / after last GPS fix)
        gps_reindexed[col] = gps_reindexed[col].ffill().bfill()

    gps_reindexed["bird_id"] = gps_reindexed["bird_id"].ffill().bfill()

    result = gps_reindexed.loc[target_idx].reset_index()
    result = result.rename(columns={"index": "timestamp"})
    return result


def merge(gps_path: str, orientation_path: str, output_path: str, resolution: str = "1hz"):
    """Merge GPS + orientation into a single timeline CSV."""

    print(f"Loading GPS data from {gps_path} ...")
    gps_df = load_gps(gps_path)
    print(f"  {len(gps_df):,} rows  |  {gps_df['timestamp'].min()} -> {gps_df['timestamp'].max()}")

    print(f"Loading orientation data from {orientation_path} at {resolution} ...")
    orient_df = load_orientation(orientation_path, resolution)
    print(f"  {len(orient_df):,} rows  |  {orient_df['timestamp'].min()} -> {orient_df['timestamp'].max()}")

    # ---- unified timeline = union of all timestamps from both sources ----
    all_ts = pd.concat([gps_df["timestamp"], orient_df["timestamp"]])
    all_ts = all_ts.drop_duplicates().sort_values().reset_index(drop=True)
    print(f"Unified timeline: {len(all_ts):,} timestamps")

    # ---- GPS: interpolate onto every timestamp ----
    print("Interpolating GPS ...")
    merged = interpolate_gps(gps_df, all_ts)

    # ---- Orientation: left-join (NaN where no sensor data) ----
    orient_join = orient_df[["timestamp", "roll", "pitch", "yaw"]].copy()
    orient_join = orient_join.drop_duplicates(subset="timestamp", keep="last")
    merged = merged.merge(orient_join, on="timestamp", how="left")

    # ---- has_orientation flag for timeline coloring ----
    merged["has_orientation"] = merged["roll"].notna()

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # ---- save ----
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    n_orient = merged["has_orientation"].sum()
    print(f"Saved {len(merged):,} rows to {out}  "
          f"({n_orient:,} with orientation, {len(merged) - n_orient:,} GPS-only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge GPS and orientation CSVs")
    parser.add_argument("--gps", required=True, help="Path to GPS CSV")
    parser.add_argument("--orientation", required=True, help="Path to orientation CSV")
    parser.add_argument("--output", required=True, help="Output merged CSV path")
    parser.add_argument("--resolution", default="1hz", choices=["1hz", "5hz", "20hz"],
                        help="Orientation resolution (default: 1hz)")
    args = parser.parse_args()

    merge(args.gps, args.orientation, args.output, args.resolution)
