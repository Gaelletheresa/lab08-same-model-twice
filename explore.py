from pathlib import Path

import matplotlib.pyplot as plt

from bikes_data import make_rentals

frame = make_rentals()
print(frame.head())

Path("reports").mkdir(exist_ok=True)
hourly_mean = frame.groupby("hour")["rentals"].mean()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(frame["hour"], frame["rentals"], s=6, alpha=0.3, label="one row")
ax.plot(hourly_mean.index, hourly_mean.values, color="red", label="mean per hour")
ax.set_xlabel("Hour of the day")
ax.set_ylabel("Bikes rented")
ax.set_title("Rentals against the hour")
ax.legend()
fig.tight_layout()
fig.savefig("reports/rentals_by_hour.png", dpi=150)
