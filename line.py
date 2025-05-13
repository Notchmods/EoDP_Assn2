import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./table.csv")

""" Speed Zone """
df["SPEED_ZONE"] = df["SPEED_ZONE"].replace({75: 80})
grouped = (
    df.groupby(["VEHICLE_CATEGORY", "SPEED_ZONE"])
    .agg(mean_injury=("AVERAGE_INJ_LEVEL", "mean"), count=("AVERAGE_INJ_LEVEL", "count"))
    .reset_index()
)

filtered = grouped[grouped["count"] >= 30]

plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered, x="SPEED_ZONE", y="mean_injury", hue="VEHICLE_CATEGORY")
plt.title("Mean Injury Level by SPEED_ZONE and Vehicle Type (Filtered: count ≥ 30)")
plt.ylabel("Mean AVERAGE_INJ_LEVEL")
plt.xlabel("SPEED_ZONE")
plt.tight_layout()
plt.show()
