import pandas as pd
import time 
import pyvista as pv

df = pd.read_csv("Data/processed_angle_day1.csv", index_col=False)
print(df.head())

def parse_angle_series(cell):
    if pd.isna(cell):
        return []
    return [float(x) for x in str(cell).split()]

plotter = pv.Plotter(window_size=[1000, 700])
plotter.enable_trackball_style()
floor = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=10, j_size=10)
plotter.add_mesh(floor, opacity=0.5)

#bird = pv.Cube(center=(0, 0, 1), x_length=1.5, y_length=0.4, z_length=0.2)
eagle = pv.read("3D Visualization/18741_Eagle_with_talons_spread_v1.obj")

#bird_actor = plotter.add_mesh(bird, show_edges=True)
eagle_actor = plotter.add_mesh(eagle, color="white")

text_actor = plotter.add_text("", position="upper_left", font_size=12)
plotter.show(auto_close=False, interactive_update=True)

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


"""
for i in range(len(df)):
    print(i, flush=True)
    row = df.iloc[i]
    time_val = i

    pitch_mean = row["pitch_mean"]
    roll_mean = row["roll_mean"]
    yaw_mean = row["yaw_mean"]

    pitch_min = row["pitch_min"]
    pitch_max = row["pitch_max"]

    yaw_min = row["yaw_min"]
    yaw_max = row["yaw_max"]

    roll_min = row["roll_min"]
    roll_max = row["roll_max"]

    cumulative_roll = row["cumulative_roll"]
    cumulative_pitch = row["cumulative_pitch"]
    cumulative_yaw = row["cumulative_yaw"]

    legend_text = (
        f"Time: {time_val} s\n"
        f"Pitch: {pitch_mean:.2f}° ({pitch_min:.2f} → {pitch_max:.2f})\n"
        f"Roll: {roll_mean:.2f}° ({roll_min:.2f} → {roll_max:.2f})\n"
        f"Yaw: {yaw_mean:.2f}° ({yaw_min:.2f} → {yaw_max:.2f})\n"
    )
    text_actor.SetText(0, legend_text)

    pitch_series = parse_angle_series(row["pitch_deg"])
    roll_series = parse_angle_series(row["roll_deg"])

    yaw_series = parse_angle_series(row["yaw_deg"])

    n = min(len(pitch_series), len(roll_series))

    for j in range(n):
        pitch_val = pitch_series[j]
        roll_val = roll_series[j]
        yaw_val = yaw_series[j]

        bird_actor.SetOrientation(pitch_val, yaw_val, roll_val)

        plotter.update()
        time.sleep(0.05)  

plotter.close()
"""




