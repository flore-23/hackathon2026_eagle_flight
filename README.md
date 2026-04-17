# Eagle Flight Visualizer

3D visualization of eagle GPS tracks and IMU orientation data.

- **Left pane** — CesiumJS 3D globe, camera locked behind the eagle, real terrain.
- **Right pane** — Leaflet 2D trajectory map (clickable to jump) / Cesium 3D view toggle.
- **Timeline** — orange segments = periods with full 3D orientation data.

---

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n eagle_flight python=3.11 -y
conda activate eagle_flight
pip install -r requirements.txt
```

### 2. Cesium Ion token (needed for real terrain)

Get a free token at [ion.cesium.com/signup](https://ion.cesium.com/signup/), then create a `.env` file in the project root:

```env
CESIUM_ION_TOKEN=your_token_here
```

> `.env` is gitignored and never committed.
> Without a token the app still works with flat terrain and OpenStreetMap tiles.

---

## Running

```bash
conda activate eagle_flight
uvicorn server:app --host 0.0.0.0 --port 8050 --reload
```

Open **http://localhost:8050** in your browser.

---

## Usage

### Option A — Upload files directly in the app

1. Click **Load Flight** and upload:
   - **GPS CSV** — e.g. `madi_loc_day1.csv`
   - **Orientation CSV** — e.g. `processed_angle_day1.csv`
2. The server merges them at 20 Hz automatically (~10 s).

### Option B — Pre-generate demo data (faster startup)

Run once to produce the merged file:

```bash
conda activate eagle_flight
python merge_data.py \
  --gps Data/madi_loc_day1.csv \
  --orientation Data/processed_angle_day1.csv \
  --output Data/merged_day1_20hz.csv \
  --resolution 20hz
```

Then click **Load Demo Data** in the app.

---

## Controls

| Action | Result |
| --- | --- |
| Click trajectory on 2D map | Jump to that point in time |
| Click anywhere on timeline | Seek to that time |
| Drag timeline scrubber | Scrub through flight |
| Play / Pause button | Start or pause playback |
| Speed selector | Change playback speed (1× – 300×) |
| 2D Map / 3D View toggle | Switch right-pane mode |

---

## Data format

The app expects two CSV files exported from Movebank / e-obs loggers:

| File | Key columns used |
| --- | --- |
| GPS | `individual_local_identifier`, `timestamp`, `lat`, `long`, `height_above_ellipsoid`, `ground_speed`, `heading` |
| Orientation | `floor_date(timestamp,"second")`, `roll_deg`, `pitch_deg`, `yaw_deg`, `roll_mean`, `pitch_mean`, `yaw_mean` |
