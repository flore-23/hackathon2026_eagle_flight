"""
merge_data.py — Merge GPS + 3D orientation + IMU acceleration for the eagle
flight visualizer, with a 15-state error-state Kalman filter (ESEKF) that fuses
GPS and IMU during IMU bursts.

Strategy:
  - GPS: linearly interpolated to fill every timestamp (no gaps).
  - Orientation: only populated where real sensor data exists, NaN elsewhere.
  - Acceleration: same — only populated within IMU bursts, NaN elsewhere.
  - During each contiguous IMU burst, an ESEKF fuses the IMU stream (omega +
    specific force) with the burst's GPS samples to produce smooth estimates of
    velocity, body-frame angular rate, and gravity-removed body-frame
    acceleration. Outside bursts, those fields are NaN.

Output CSV columns:
  timestamp, bird_id, lat, lon, altitude, ground_speed, heading,
  roll, pitch, yaw, has_orientation,
  vx_body, vy_body, vz_body,        # m/s,  body frame (x=forward, y=right, z=down)
  omega_x, omega_y, omega_z,        # deg/s, body frame
  ax_body, ay_body, az_body         # m/s², body frame, gravity removed

Usage:
    python merge_data.py \
        --gps Data/madi_loc_day1.csv \
        --orientation Data/processed_angle_day1.csv \
        --acceleration Data/processed_acc_day1.csv \
        --output Data/merged_day1_20hz.csv \
        --resolution 20hz
"""

import argparse
import csv
import numpy as np
import pandas as pd
from pathlib import Path

from ekf import (
    G_NED,
    StrapdownEKF,
    lla_to_ned,
    ned_to_lla,
    omega_from_quat_pair,
    quat_from_rpy,
    quat_to_rotmat,
    quat_to_rpy,
)


# ============================================================
#  CSV loaders
# ============================================================

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


def _read_csv_with_geometry_fix(path: str):
    """Movebank exports leave an unquoted ``c(NA, NA)`` in the geometry column
    which adds a stray comma; merge the two split fields back together when a
    row has one more field than the header."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_hdr = len(header)
        geom_idx = header.index("geometry") if "geometry" in header else None
        rows = []
        for row in reader:
            if len(row) == n_hdr + 1 and geom_idx is not None:
                row = row[:geom_idx] + [row[geom_idx] + "," + row[geom_idx + 1]] + row[geom_idx + 2:]
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def load_orientation(path: str, resolution: str = "1hz") -> pd.DataFrame:
    """Load orientation CSV and expand to the requested resolution.

    1hz  → roll_mean / pitch_mean / yaw_mean per second.
    5hz  → parse 20 Hz sub-sample arrays, keep every 4th sample.
    20hz → parse all sub-samples.
    """
    df = _read_csv_with_geometry_fix(path)
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

    step = 4 if resolution == "5hz" else 1
    freq = 20

    records = []
    for _, row in df.iterrows():
        base_time = row["second_ts"]
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


def load_acceleration(path: str, resolution: str = "20hz") -> pd.DataFrame:
    """Parse processed_acc CSV → per-sample body-frame acceleration in m/s².

    Each row is one IMU burst with `eobs_acceleration_g` containing
    space-separated XYZ triplets at 20 Hz. Returns one row per sample.
    """
    df = _read_csv_with_geometry_fix(path)
    # Some rows have fractional seconds and some don't — let pandas infer per row.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)

    G = 9.81
    freq = 20
    step = {"1hz": 20, "5hz": 4, "20hz": 1}.get(resolution, 1)

    records = []
    for _, row in df.iterrows():
        base_time = row["timestamp"]
        s = str(row.get("eobs_acceleration_g", ""))
        if not s.strip() or s == "nan":
            continue
        vals = s.split()
        n_triplets = len(vals) // 3
        for i in range(0, n_triplets, step):
            ax_g = float(vals[3 * i + 0])
            ay_g = float(vals[3 * i + 1])
            az_g = float(vals[3 * i + 2])
            records.append({
                "timestamp": base_time + pd.Timedelta(seconds=i / freq),
                "ax_body":   ax_g * G,
                "ay_body":   ay_g * G,
                "az_body":   az_g * G,
            })
    return pd.DataFrame(records)


def interpolate_gps(gps_df: pd.DataFrame, target_timestamps: pd.Series) -> pd.DataFrame:
    """Linearly interpolate GPS values onto every target timestamp."""
    gps = gps_df.set_index("timestamp").sort_index()
    gps = gps[~gps.index.duplicated(keep="last")]
    target_idx = pd.DatetimeIndex(target_timestamps).sort_values().drop_duplicates()
    combined_idx = gps.index.union(target_idx)
    gps_reindexed = gps.reindex(combined_idx)
    for col in ["lat", "lon", "altitude", "ground_speed", "heading"]:
        gps_reindexed[col] = gps_reindexed[col].interpolate(method="time")
        gps_reindexed[col] = gps_reindexed[col].ffill().bfill()
    gps_reindexed["bird_id"] = gps_reindexed["bird_id"].ffill().bfill()
    result = gps_reindexed.loc[target_idx].reset_index()
    result = result.rename(columns={"index": "timestamp"})
    return result


# ============================================================
#  Burst-by-burst fusion driver  (uses ekf.StrapdownEKF)
# ============================================================

def _geo_bearing(lat1, lon1, lat2, lon2):
    """Great-circle bearing from (lat1, lon1) to (lat2, lon2), degrees [0, 360)."""
    R = np.pi / 180
    f1 = lat1 * R; f2 = lat2 * R; dl = (lon2 - lon1) * R
    y = np.sin(dl) * np.cos(f2)
    x = np.cos(f1) * np.sin(f2) - np.sin(f1) * np.cos(f2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _geo_dist_m(lat1, lon1, lat2, lon2):
    """Haversine distance between two lat/lon points, in metres."""
    R = np.pi / 180
    dlat = (lat2 - lat1) * R
    dlon = (lon2 - lon1) * R
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * R) * np.cos(lat2 * R) * np.sin(dlon / 2) ** 2)
    return 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _estimate_initial_course(gps_df, t_start, t_end, n_samples=5, min_speed=2.0):
    """Compass heading at the start of a burst, averaged (circular mean) over
    the first few moving GPS samples within the burst window.

    Returns degrees in [0, 360), or None if no usable GPS pairs exist.
    """
    win = gps_df[(gps_df["timestamp"] >= t_start) & (gps_df["timestamp"] <= t_end)]
    win = win[win["ground_speed"] > min_speed].reset_index(drop=True)
    if len(win) < 2:
        return None
    bearings = []
    for i in range(min(n_samples, len(win) - 1)):
        bearings.append(_geo_bearing(
            win["lat"].iat[i], win["lon"].iat[i],
            win["lat"].iat[i + 1], win["lon"].iat[i + 1],
        ))
    if not bearings:
        return None
    rad = np.radians(bearings)
    return (np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360.0) % 360.0


def fuse_imu_bursts(merged_df: pd.DataFrame, gps_df: pd.DataFrame) -> pd.DataFrame:
    """Run the ESEKF over each contiguous IMU burst in `merged_df` and
    populate the new fused columns. Outside bursts every fused field is NaN.

    Requires `merged_df` to already contain (per row): roll/pitch/yaw (deg),
    has_orientation (bool), ax_body/ay_body/az_body (m/s²), lat/lon/altitude.
    """
    new_cols = [
        "vx_body", "vy_body", "vz_body",
        "omega_x", "omega_y", "omega_z",
        "ax_body_inertial", "ay_body_inertial", "az_body_inertial",
    ]
    for c in new_cols:
        merged_df[c] = np.nan

    has_orient = merged_df["has_orientation"].fillna(False).values.astype(bool)
    n_rows = len(merged_df)

    # Locate contiguous bursts
    bursts = []
    i = 0
    while i < n_rows:
        if has_orient[i]:
            j = i
            while j < n_rows and has_orient[j]:
                j += 1
            bursts.append((i, j))
            i = j
        else:
            i += 1
    if not bursts:
        print("No IMU bursts found — nothing to fuse.")
        return merged_df

    print(f"Found {len(bursts)} IMU burst(s); running EKF on each…")

    timestamps = merged_df["timestamp"].values
    roll_arr   = merged_df["roll"].values
    pitch_arr  = merged_df["pitch"].values
    yaw_arr    = merged_df["yaw"].values
    ax_arr     = merged_df["ax_body"].values
    ay_arr     = merged_df["ay_body"].values
    az_arr     = merged_df["az_body"].values
    lat_arr    = merged_df["lat"].values
    lon_arr    = merged_df["lon"].values
    alt_arr    = merged_df["altitude"].values

    gps_ts = gps_df["timestamp"].values
    gps_lat = gps_df["lat"].values
    gps_lon = gps_df["lon"].values
    gps_alt = gps_df["altitude"].values

    n_skipped_no_accel = 0

    for (s, e) in bursts:
        burst_n = e - s
        if burst_n < 2:
            continue

        # Origin for local NED = first row's GPS position
        lat0, lon0, alt0 = lat_arr[s], lon_arr[s], alt_arr[s]

        # Per-burst yaw calibration: the e-obs IMU integrates yaw from the
        # gyro alone (no usable magnetometer on a bird-mounted unit), so its
        # absolute value is unrelated to compass north. We anchor it by the
        # GPS-derived course at the start of the burst, then trust the IMU
        # yaw rate for everything else within the burst.
        t_start_burst = pd.Timestamp(timestamps[s])
        t_end_burst   = pd.Timestamp(timestamps[e - 1])
        course0 = _estimate_initial_course(gps_df, t_start_burst, t_end_burst)
        if course0 is not None:
            yaw_offset_deg = ((course0 - yaw_arr[s] + 180.0) % 360.0) - 180.0
        else:
            yaw_offset_deg = 0.0

        # Pre-build attitude quaternions for the whole burst (used both for
        # initial state and for finite-diff omega input).
        q_seq = np.empty((burst_n, 4))
        for k in range(burst_n):
            r = np.radians(roll_arr[s + k])
            p = np.radians(pitch_arr[s + k])
            y = np.radians(yaw_arr[s + k] + yaw_offset_deg)
            q_seq[k] = quat_from_rpy(r, p, y)

        # Initial velocity from a 1-second GPS finite difference around burst start
        t0 = pd.Timestamp(timestamps[s])
        prev_idx = np.searchsorted(gps_ts, np.datetime64(t0 - pd.Timedelta(seconds=1)), side="left")
        next_idx = np.searchsorted(gps_ts, np.datetime64(t0 + pd.Timedelta(seconds=1)), side="right") - 1
        if prev_idx < len(gps_ts) and next_idx > prev_idx:
            p_prev = lla_to_ned(gps_lat[prev_idx], gps_lon[prev_idx], gps_alt[prev_idx], lat0, lon0, alt0)
            p_next = lla_to_ned(gps_lat[next_idx], gps_lon[next_idx], gps_alt[next_idx], lat0, lon0, alt0)
            dt0 = (pd.Timestamp(gps_ts[next_idx]) - pd.Timestamp(gps_ts[prev_idx])).total_seconds()
            v0 = (p_next - p_prev) / dt0 if dt0 > 0 else np.zeros(3)
        else:
            v0 = np.zeros(3)

        ekf = StrapdownEKF(q0=q_seq[0], p0=np.zeros(3), v0=v0)

        # GPS samples that fall within this burst's time window
        burst_start_ts = pd.Timestamp(timestamps[s])
        burst_end_ts   = pd.Timestamp(timestamps[e - 1])
        gps_mask = (gps_df["timestamp"] >= burst_start_ts) & (gps_df["timestamp"] <= burst_end_ts)
        burst_gps = gps_df.loc[gps_mask].reset_index(drop=True)
        burst_gps_ts = burst_gps["timestamp"].values
        burst_gps_used = np.zeros(len(burst_gps), dtype=bool)

        # GPS Doppler velocity per sample: GPS receivers compute ground_speed
        # and heading from carrier-phase Doppler shift, which is far more
        # accurate (~0.1 m/s) than finite-differencing positions. Feeding it as
        # a direct velocity observation kills the position-only sawtooth.
        # Vertical comes from a central altitude difference (loose σ).
        gps_velocities = []      # (timestamp, v_NED, sigma_NED)
        for gi in range(len(burst_gps)):
            ts = pd.Timestamp(burst_gps_ts[gi])
            gs_v = burst_gps["ground_speed"].iat[gi]
            hd_v = burst_gps["heading"].iat[gi]
            if not (np.isfinite(gs_v) and np.isfinite(hd_v)) or gs_v < 0.5:
                continue   # Doppler unreliable at near-zero speed
            hd_rad = np.radians(hd_v)
            v_n = gs_v * np.cos(hd_rad)
            v_e = gs_v * np.sin(hd_rad)
            if 0 < gi < len(burst_gps) - 1:
                dt_v = (pd.Timestamp(burst_gps_ts[gi + 1])
                        - pd.Timestamp(burst_gps_ts[gi - 1])).total_seconds()
                v_d = (-(burst_gps["altitude"].iat[gi + 1]
                          - burst_gps["altitude"].iat[gi - 1]) / dt_v
                       if dt_v > 0 else 0.0)
            else:
                v_d = 0.0
            gps_velocities.append((ts,
                                   np.array([v_n, v_e, v_d]),
                                   np.array([0.3, 0.3, 3.0])))   # m/s
        gps_velocity_used = np.zeros(len(gps_velocities), dtype=bool)

        # Course observations: bearing of each consecutive GPS pair, applied
        # at the midpoint timestamp. Speed-gated so we don't feed noise when
        # the bird is hovering. σ scales with how far the GPS pair travelled
        # (≈ position-noise / distance).
        courses = []
        SIGMA_POS = 5.0   # GPS horizontal noise (m) — matches the EKF prior
        for gi in range(len(burst_gps) - 1):
            t1 = pd.Timestamp(burst_gps_ts[gi])
            t2 = pd.Timestamp(burst_gps_ts[gi + 1])
            dt_gps = (t2 - t1).total_seconds()
            if dt_gps <= 0:
                continue
            dist_m = _geo_dist_m(
                burst_gps["lat"].iat[gi],     burst_gps["lon"].iat[gi],
                burst_gps["lat"].iat[gi + 1], burst_gps["lon"].iat[gi + 1])
            speed = dist_m / dt_gps
            if speed < 2.0 or dist_m < 1.0:
                continue
            course_rad = np.radians(_geo_bearing(
                burst_gps["lat"].iat[gi],     burst_gps["lon"].iat[gi],
                burst_gps["lat"].iat[gi + 1], burst_gps["lon"].iat[gi + 1]))
            sigma = max(0.05, min(0.6, SIGMA_POS / dist_m))   # rad
            mid_ts = t1 + (t2 - t1) / 2
            courses.append((mid_ts, course_rad, sigma))
        course_used = np.zeros(len(courses), dtype=bool)

        # Step through the burst at the merged-timeline cadence (~20 Hz)
        for k in range(burst_n):
            row_idx = s + k

            if k > 0:
                dt = (pd.Timestamp(timestamps[row_idx]) - pd.Timestamp(timestamps[row_idx - 1])).total_seconds()
                if dt <= 0 or dt > 1.0:
                    dt = 0.05  # safety fallback

                # ω from finite-diff of the data's attitude
                if k < burst_n:
                    omega_meas = omega_from_quat_pair(q_seq[k - 1], q_seq[k], dt)
                else:
                    omega_meas = np.zeros(3)

                # Accel from data; if NaN (accel file gap), pretend 0 with high noise
                ax = ax_arr[row_idx]; ay = ay_arr[row_idx]; az = az_arr[row_idx]
                if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(az)):
                    n_skipped_no_accel += 1
                    f_meas = -G_NED  # fall back to "stationary" specific force
                else:
                    f_meas = np.array([ax, ay, az])

                ekf.predict(omega_meas, f_meas, dt)

                # GPS update if a real GPS sample falls within this 50 ms window
                row_ts = pd.Timestamp(timestamps[row_idx])
                window = pd.Timedelta(seconds=dt / 2)
                for gi in range(len(burst_gps)):
                    if burst_gps_used[gi]:
                        continue
                    if abs(pd.Timestamp(burst_gps_ts[gi]) - row_ts) <= window:
                        z = lla_to_ned(burst_gps["lat"].iat[gi],
                                       burst_gps["lon"].iat[gi],
                                       burst_gps["altitude"].iat[gi],
                                       lat0, lon0, alt0)
                        ekf.update_gps(z)
                        burst_gps_used[gi] = True
                        break

                # Course observation if a midpoint falls in this row's window
                for ci in range(len(courses)):
                    if course_used[ci]:
                        continue
                    if abs(courses[ci][0] - row_ts) <= window:
                        ekf.update_course(courses[ci][1], sigma_course=courses[ci][2])
                        course_used[ci] = True
                        break

                # Doppler velocity update at each GPS sample
                for vi in range(len(gps_velocities)):
                    if gps_velocity_used[vi]:
                        continue
                    if abs(gps_velocities[vi][0] - row_ts) <= window:
                        ekf.update_velocity(gps_velocities[vi][1], gps_velocities[vi][2])
                        gps_velocity_used[vi] = True
                        break

            # Output: rotate filtered velocity into body frame; gravity-removed
            # body-frame inertial accel; bias-compensated body-frame omega.
            R = quat_to_rotmat(ekf.q)
            v_body = R.T @ ekf.v
            f_comp = (np.array([ax_arr[row_idx], ay_arr[row_idx], az_arr[row_idx]])
                      - ekf.b_a) if (np.isfinite(ax_arr[row_idx])
                                      and np.isfinite(ay_arr[row_idx])
                                      and np.isfinite(az_arr[row_idx])) else None
            if f_comp is not None:
                a_body_inertial = f_comp + R.T @ G_NED
            else:
                a_body_inertial = np.array([np.nan, np.nan, np.nan])

            if k > 0:
                dt_now = (pd.Timestamp(timestamps[row_idx])
                          - pd.Timestamp(timestamps[row_idx - 1])).total_seconds()
                if dt_now <= 0 or dt_now > 1.0:
                    dt_now = 0.05
                omega_body = omega_from_quat_pair(q_seq[k - 1], q_seq[k], dt_now) - ekf.b_g
            else:
                omega_body = np.zeros(3)

            # ---- EKF-derived outputs (overwrite existing GPS-/IMU-data values
            #      during bursts so every fused field comes from the filter) ----
            # Position: filter NED → lat/lon/alt
            lat_f, lon_f, alt_f = ned_to_lla(ekf.p[0], ekf.p[1], ekf.p[2], lat0, lon0, alt0)
            # Attitude: filter quaternion → roll/pitch/yaw (deg)
            roll_f, pitch_f, yaw_f = quat_to_rpy(ekf.q)
            # Horizontal speed and compass heading from filter NED velocity
            vN, vE = ekf.v[0], ekf.v[1]
            gs_f = float(np.hypot(vN, vE))
            hdg_f = (np.degrees(np.arctan2(vE, vN)) + 360.0) % 360.0

            cols = merged_df.columns
            merged_df.iat[row_idx, cols.get_loc("lat")]              = lat_f
            merged_df.iat[row_idx, cols.get_loc("lon")]              = lon_f
            merged_df.iat[row_idx, cols.get_loc("altitude")]         = alt_f
            merged_df.iat[row_idx, cols.get_loc("ground_speed")]     = gs_f
            merged_df.iat[row_idx, cols.get_loc("heading")]          = hdg_f
            merged_df.iat[row_idx, cols.get_loc("roll")]             = np.degrees(roll_f)
            merged_df.iat[row_idx, cols.get_loc("pitch")]            = np.degrees(pitch_f)
            merged_df.iat[row_idx, cols.get_loc("yaw")]              = np.degrees(yaw_f)
            merged_df.iat[row_idx, cols.get_loc("vx_body")]          = v_body[0]
            merged_df.iat[row_idx, cols.get_loc("vy_body")]          = v_body[1]
            merged_df.iat[row_idx, cols.get_loc("vz_body")]          = v_body[2]
            merged_df.iat[row_idx, cols.get_loc("omega_x")]          = np.degrees(omega_body[0])
            merged_df.iat[row_idx, cols.get_loc("omega_y")]          = np.degrees(omega_body[1])
            merged_df.iat[row_idx, cols.get_loc("omega_z")]          = np.degrees(omega_body[2])
            merged_df.iat[row_idx, cols.get_loc("ax_body_inertial")] = a_body_inertial[0]
            merged_df.iat[row_idx, cols.get_loc("ay_body_inertial")] = a_body_inertial[1]
            merged_df.iat[row_idx, cols.get_loc("az_body_inertial")] = a_body_inertial[2]

    # Rename inertial-accel columns back to ax_body/ay_body/az_body for output
    # (the raw IMU specific-force columns are kept under ax_body_raw/etc.)
    merged_df.rename(columns={
        "ax_body": "ax_body_raw",
        "ay_body": "ay_body_raw",
        "az_body": "az_body_raw",
        "ax_body_inertial": "ax_body",
        "ay_body_inertial": "ay_body",
        "az_body_inertial": "az_body",
    }, inplace=True)

    if n_skipped_no_accel:
        print(f"  ({n_skipped_no_accel} predict steps ran with missing accel — "
              "filled with rest-state assumption)")
    return merged_df


# ============================================================
#  Merge entry point
# ============================================================

def merge(gps_path: str,
          orientation_path: str,
          acceleration_path: str,
          output_path: str,
          resolution: str = "20hz"):
    """Merge GPS + orientation + acceleration into one CSV, then run the EKF."""

    print(f"Loading GPS data from {gps_path} ...")
    gps_df = load_gps(gps_path)
    print(f"  {len(gps_df):,} rows  |  {gps_df['timestamp'].min()} → {gps_df['timestamp'].max()}")

    print(f"Loading orientation data from {orientation_path} at {resolution} ...")
    orient_df = load_orientation(orientation_path, resolution)
    print(f"  {len(orient_df):,} rows  |  {orient_df['timestamp'].min()} → {orient_df['timestamp'].max()}")

    print(f"Loading acceleration data from {acceleration_path} at {resolution} ...")
    accel_df = load_acceleration(acceleration_path, resolution)
    if not accel_df.empty:
        print(f"  {len(accel_df):,} rows  |  {accel_df['timestamp'].min()} → {accel_df['timestamp'].max()}")
    else:
        print("  (no acceleration samples parsed)")

    # Unified timeline = GPS ∪ orientation timestamps (acceleration is ALIGNED
    # to the orientation grid below, NOT unioned, so its slightly-offset
    # sub-second timestamps don't fragment has_orientation runs).
    all_ts = pd.concat([gps_df["timestamp"], orient_df["timestamp"]])
    all_ts = all_ts.drop_duplicates().sort_values().reset_index(drop=True)
    print(f"Unified timeline: {len(all_ts):,} timestamps")

    # GPS interpolated everywhere
    print("Interpolating GPS …")
    merged = interpolate_gps(gps_df, all_ts)

    # Left-join orientation
    orient_join = orient_df[["timestamp", "roll", "pitch", "yaw"]].drop_duplicates(
        subset="timestamp", keep="last")
    merged = merged.merge(orient_join, on="timestamp", how="left")
    merged["has_orientation"] = merged["roll"].notna()
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Snap acceleration to the merged-timeline grid via nearest-neighbour
    # within ±25 ms (= half a 20 Hz sample period). The accel sampler runs at
    # 20 Hz too but starts at a different sub-second offset than orientation,
    # so an exact-timestamp join misses every sample.
    if not accel_df.empty:
        a = accel_df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(
            merged, a, on="timestamp",
            direction="nearest", tolerance=pd.Timedelta(milliseconds=25),
        )
    else:
        for c in ("ax_body", "ay_body", "az_body"):
            merged[c] = np.nan

    # Run the EKF
    print("Running ESEKF over IMU bursts …")
    merged = fuse_imu_bursts(merged, gps_df)

    # Drop the raw specific-force columns from the output (kept as
    # ax_body_raw/etc. internally; remove for a clean public schema).
    drop_cols = [c for c in ("ax_body_raw", "ay_body_raw", "az_body_raw") if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    n_orient = int(merged["has_orientation"].sum())
    n_fused  = int(merged["vx_body"].notna().sum())
    print(f"Saved {len(merged):,} rows to {out}")
    print(f"  has_orientation : {n_orient:,}")
    print(f"  EKF-fused rows  : {n_fused:,}  (the rest are NaN for fused fields)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge GPS + orientation + acceleration with EKF fusion")
    parser.add_argument("--gps",          required=True, help="Path to GPS CSV")
    parser.add_argument("--orientation",  required=True, help="Path to orientation CSV")
    parser.add_argument("--acceleration", required=True, help="Path to acceleration CSV")
    parser.add_argument("--output",       required=True, help="Output merged CSV path")
    parser.add_argument("--resolution",   default="20hz", choices=["1hz", "5hz", "20hz"],
                        help="Output resolution (default: 20hz)")
    args = parser.parse_args()
    merge(args.gps, args.orientation, args.acceleration, args.output, args.resolution)
