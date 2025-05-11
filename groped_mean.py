import pandas as pd
import pre
data = pd.read_csv("./table.csv")

weather_priority = [
    "Raining", "Snowing", "Fog", "Strong winds", 
    "Smoke", "Dust", "Clear"
]

def group_mean_analysis(df, dependent, independent):
    grouped_means = df.groupby(dependent)[independent].agg(["mean", "count"]).reset_index()
    grouped_means = grouped_means[grouped_means["count"] > 500]
    print(grouped_means)
    return
    
group_mean_analysis(data, ["VEHICLE_CATEGORY", "LIGHT_LEVEL", "ROAD_GEOMETRY_DESC"], "AVERAGE_INJ_LEVEL")
print("\n")
group_mean_analysis(data, ["VEHICLE_CATEGORY", "MAIN_ATMOSPH_COND"], "AVERAGE_INJ_LEVEL")