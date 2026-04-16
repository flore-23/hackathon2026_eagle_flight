import pandas as pd
import pyvista as pv
import time
from pathlib import Path
from tkinter import Tk, filedialog

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

DEFAULT_DATA = REPO_DIR / "Data" / "processed_angle_day1.csv"
DEFAULT_MODEL = BASE_DIR / "18741_Eagle_with_talons_spread_v1.obj"

print("SCRIPT:", BASE_DIR)
print("DATA :", DEFAULT_DATA)
print("MODEL:", DEFAULT_MODEL)
print("MODEL EXISTS:", DEFAULT_MODEL.exists())

root = Tk()
root.withdraw()

print("Select your CSV file (Cancel to use default)")
data_path = filedialog.askopenfilename(
    title="Select CSV file",
    filetypes=[("CSV files", "*.csv")]
)

if not data_path:
    data_path = DEFAULT_DATA

print("Using data file:", data_path)

df = pd.read_csv(data_path, index_col=False)
eagle = pv.read(DEFAULT_MODEL)


eagle.scale([1, 1, 1], inplace=True)

def parse_angle_series(cell):
    if pd.isna(cell):
        return []
    return [float(x) for x in str(cell).split()]

plotter = pv.Plotter(window_size=[1000, 700])
plotter.enable_trackball_style()

floor = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=250, j_size=250)
plotter.add_mesh(floor, opacity=0.5)

eagle_actor = plotter.add_mesh(eagle, color="white")
text_actor = plotter.add_text("", position="upper_left", font_size=12)

plotter.show(auto_close=False, interactive_update=True)

try:
    for sec_idx, (_, row) in enumerate(df.iterrows()):
        rolls = parse_angle_series(row["roll_deg"])
        pitchs = parse_angle_series(row["pitch_deg"])
        yaws = parse_angle_series(row["yaw_deg"])

        n = min(len(rolls), len(pitchs), len(yaws))
        if n == 0:
            continue

        for i in range(n):
            eagle_actor.SetOrientation(rolls[i], pitchs[i], yaws[i])

            current_time = sec_idx + i / max(n, 1)
            legend_text = (
                f"Time: {current_time:.2f} s\n"
                f"Roll: {rolls[i]:.2f} deg\n"
                f"Pitch: {pitchs[i]:.2f} deg\n"
                f"Yaw: {yaws[i]:.2f} deg"
            )
            text_actor.SetText(0, legend_text)

            plotter.update()
            time.sleep(0.05)

except Exception:
    pass

finally:
    try:
        plotter.close()
    except Exception:
        pass
