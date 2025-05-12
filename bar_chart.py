import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./table.csv")

grouped = df.groupby(["VEHICLE_CATEGORY", "LIGHT_LEVEL", "AVERAGE_INJ_LEVEL"]).size().reset_index(name="count")

grouped["total"] = grouped.groupby(["VEHICLE_CATEGORY", "LIGHT_LEVEL"])["count"].transform("sum")
grouped["proportion"] = grouped["count"] / grouped["total"]

g = sns.catplot(data=grouped, x="LIGHT_LEVEL", y="proportion", hue="AVERAGE_INJ_LEVEL", kind="bar", col="VEHICLE_CATEGORY", col_wrap=4,palette="YlOrRd", height=4, aspect=1)

g.set_titles("{col_name}")
g.set_axis_labels("LIGHT_LEVEL", "Proportion")
g._legend.set_title("INJURY LEVEL")
plt.tight_layout()
plt.show()

