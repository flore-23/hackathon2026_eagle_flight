"""
server.py — FastAPI backend for Eagle Flight Visualization.

Endpoints:
  POST /api/upload    Accept GPS + orientation CSVs, merge at 20 Hz, return flight JSON.
  GET  /              Serve the single-page CesiumJS app.
"""

import io
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from merge_data import load_gps, load_orientation, interpolate_gps

app = FastAPI(title="Eagle Flight")

@app.get("/config.js", response_class=__import__("fastapi").responses.Response)
def config_js():
    token = os.getenv("CESIUM_ION_TOKEN", "")
    return __import__("fastapi").responses.Response(
        content=f'window.CESIUM_ION_TOKEN = "{token}";',
        media_type="application/javascript",
    )
app.add_middleware(GZipMiddleware, minimum_size=1000)

RESOLUTION = "20hz"
# Position/speed/heading only need ~1 Hz (GPS native rate during bursts).
# CesiumJS interpolates between keyframes, so 20 Hz adds no real information.
POSITION_SUBSAMPLE = 20  # keep every 20th row for position data


def _safe_round(x, ndigits: int):
    """Round to ndigits, but emit None for NaN/Inf so JSON encoding works."""
    fx = float(x)
    if not np.isfinite(fx):
        return None
    return round(fx, ndigits)


def build_flight_json(merged: pd.DataFrame) -> dict:
    """Convert a merged DataFrame into a compact JSON payload for the frontend.

    Returns
    -------
    dict with keys:
      bird_id, start_time, end_time, epoch,
      positions   – flat array [t, lon, lat, alt, …] (seconds since epoch),
      speeds      – flat array [t, speed, …],
      headings    – flat array [t, heading, …],
      orientations – {times: [...], roll: [...], pitch: [...], yaw: [...]},
      orientation_intervals – [[start, end], …] in seconds since epoch.
    """
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    epoch = merged["timestamp"].iloc[0]
    # Seconds since epoch for every row
    t_sec = (merged["timestamp"] - epoch).dt.total_seconds().values

    # --- positions / speeds / headings: subsample to ~1 Hz ---
    # GPS was originally 1 Hz; the 20 Hz interpolation adds no real info for
    # position.  CesiumJS will smoothly interpolate between these keyframes.
    step = POSITION_SUBSAMPLE
    idx = list(range(0, len(t_sec), step))
    if idx[-1] != len(t_sec) - 1:
        idx.append(len(t_sec) - 1)  # always include the last point

    positions = []
    speeds = []
    headings = []
    for i in idx:
        t = round(t_sec[i], 3)
        positions.extend([t, _safe_round(merged["lon"].iat[i], 6),
                             _safe_round(merged["lat"].iat[i], 6),
                             _safe_round(merged["altitude"].iat[i], 1)])
        speeds.extend([t, _safe_round(merged["ground_speed"].iat[i], 2)])
        headings.extend([t, _safe_round(merged["heading"].iat[i], 2)])

    # --- orientations: only rows with data ---
    orient_mask = merged["has_orientation"].values
    orient_t = t_sec[orient_mask]
    orient_roll = merged.loc[orient_mask, "roll"].values
    orient_pitch = merged.loc[orient_mask, "pitch"].values
    orient_yaw = merged.loc[orient_mask, "yaw"].values

    orientations = {
        "times": [_safe_round(x, 3) for x in orient_t],
        "roll":  [_safe_round(x, 3) for x in orient_roll],
        "pitch": [_safe_round(x, 3) for x in orient_pitch],
        "yaw":   [_safe_round(x, 3) for x in orient_yaw],
    }

    # EKF-fused fields share the orientation timestamps (they're populated
    # only during IMU bursts). Forward whatever the merged CSV contains.
    for col in ("vx_body", "vy_body", "vz_body",
                "omega_x", "omega_y", "omega_z",
                "ax_body", "ay_body", "az_body"):
        if col in merged.columns:
            vals = merged.loc[orient_mask, col].values
            orientations[col] = [_safe_round(x, 3) for x in vals]

    # --- orientation intervals (contiguous blocks of has_orientation=True) ---
    intervals = []
    in_block = False
    block_start = 0.0
    for i, has in enumerate(orient_mask):
        if has and not in_block:
            block_start = t_sec[i]
            in_block = True
        elif not has and in_block:
            intervals.append([round(block_start, 3), round(t_sec[i - 1], 3)])
            in_block = False
    if in_block:
        intervals.append([round(block_start, 3), round(t_sec[len(t_sec) - 1], 3)])

    return {
        "bird_id": str(merged["bird_id"].iloc[0]),
        "start_time": epoch.isoformat() + "Z",
        "end_time": merged["timestamp"].iloc[-1].isoformat() + "Z",
        "epoch": epoch.isoformat() + "Z",
        "positions": positions,
        "speeds": speeds,
        "headings": headings,
        "orientations": orientations,
        "orientation_intervals": intervals,
    }


@app.post("/api/upload")
async def upload_files(
    gps_file: UploadFile = File(...),
    orientation_file: UploadFile = File(...),
):
    """Accept two CSV uploads, merge them, return flight JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        gps_path = Path(tmp) / "gps.csv"
        orient_path = Path(tmp) / "orient.csv"

        gps_path.write_bytes(await gps_file.read())
        orient_path.write_bytes(await orientation_file.read())

        gps_df = load_gps(str(gps_path))
        orient_df = load_orientation(str(orient_path), RESOLUTION)

        # Unified timeline
        all_ts = pd.concat([gps_df["timestamp"], orient_df["timestamp"]])
        all_ts = all_ts.drop_duplicates().sort_values().reset_index(drop=True)

        # Interpolate GPS
        merged = interpolate_gps(gps_df, all_ts)

        # Join orientation
        orient_join = orient_df[["timestamp", "roll", "pitch", "yaw"]].copy()
        orient_join = orient_join.drop_duplicates(subset="timestamp", keep="last")
        merged = merged.merge(orient_join, on="timestamp", how="left")
        merged["has_orientation"] = merged["roll"].notna()
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        payload = build_flight_json(merged)

    return JSONResponse(content=payload)


# ----- also serve pre-merged demo data if it exists -----
DEMO_DATA_PATH = Path(__file__).resolve().parent / "Data" / "merged_day1_20hz.csv"


@app.get("/api/demo")
async def demo_flight():
    """Return flight JSON from the pre-merged demo CSV."""
    if not DEMO_DATA_PATH.exists():
        return JSONResponse({"error": "No demo data found"}, status_code=404)
    merged = pd.read_csv(DEMO_DATA_PATH)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    return JSONResponse(content=build_flight_json(merged))


# ----- static files (must be last) -----
app.mount("/", StaticFiles(directory="static", html=True), name="static")
