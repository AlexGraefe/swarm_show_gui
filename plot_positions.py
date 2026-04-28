import pandas as pd
import matplotlib.pyplot as plt
import re

csv_path = "karate/takeoff_phase.csv"

df = pd.read_csv(csv_path)
row = df.iloc[0]

# Parse drone count from column names
drone_ids = sorted({int(re.match(r"[xyz](\d+)", col).group(1)) for col in df.columns})

xs = [row[f"x{i}"] for i in drone_ids]
ys = [row[f"y{i}"] for i in drone_ids]

fig, ax = plt.subplots()
ax.scatter(xs, ys, c="blue", s=40)

for i, drone_id in enumerate(drone_ids):
    ax.text(xs[i], ys[i], str(drone_id), fontsize=7)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Drone positions — row 0 ({len(drone_ids)} drones)")
plt.tight_layout()
plt.show()
