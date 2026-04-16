import pandas as pd
import pyvista as pv
df = pd.read_csv("Data/processed_acc_day1.csv")
print(df.columns.tolist())
print(df.head())
plotter = pv.Plotter(window_size=[1000, 700])
floor = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
plotter.add_mesh(floor, opacity=0.5)
# -------------------------
# ADD PLACEHOLDER "BIRD"
# -------------------------
bird = pv.Cube(center=(0, 0, 1),
               x_length=1.5,
               y_length=0.4,
               z_length=0.2)

plotter.add_mesh(bird, show_edges=True)

# -------------------------
# ADD AXES (important for orientation)
# -------------------------
plotter.add_axes()
plotter.show_grid()

# -------------------------
# TEMPORARY LEGEND VALUES
# (we will replace with real data)
# -------------------------
row = df.iloc[0]  # first row for now


time_val = row.get("time", 0)
roll_val = row.get("roll", 0)
pitch_val = row.get("pitch", 0)
yaw_val = row.get("yaw", 0)
altitude_val = row.get("altitude", 1)

legend_text = (
    f"Time: {time_val:.2f} s\n"
    f"Roll: {roll_val:.2f}°\n"
    f"Pitch: {pitch_val:.2f}°\n"
    f"Yaw: {yaw_val:.2f}°\n"
    f"Altitude: {altitude_val:.2f} m"
)

plotter.add_text(legend_text,
                 position="upper_left",
                 font_size=12)

# -------------------------
# SHOW WINDOW
# -------------------------
plotter.show()





