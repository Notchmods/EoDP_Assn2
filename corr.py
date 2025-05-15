from pre import everything_df
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

""" 
Research Question:  What are the differences in traffic accident outcomes among various 
vehicle types under different road and environmental conditions? 

Correlation Analysis: Identify which factors("LIGHT_LEVEL", "ROAD_TYPE", "RAIN/SNOW", "FOG/SMOKE/DUST", "CLEAR") 
are most associated with severe accidents. 
"""

v_path = "vehicle.csv"
a_path = "accident.csv"
p_path = "person.csv"
atmo_path = "atmospheric_cond.csv"
everything = everything_df(v_path, a_path, atmo_path, p_path)
vehicle_types = [
    "Car",
    "Station Wagon",
    "Utility",
    "Motorcycle",
    "Bicycle",
    "Light Commercial Vehicle",
    "Heavy Truck",
    "Public Transport"
]

#call pearson_corr(everything, ["SPEED_ZONE", "AVERAGE_INJ_LEVEL"])
def pearson_corr(df, variables):
    if len(variables) != 2:
        print("Only 2 variables")
        return
    
    corr_df = pd.DataFrame(columns=["VEHICLE_CATEGORY", "PEARSON_CORR"])
    for v in vehicle_types:
        new_df = df[(df["VEHICLE_CATEGORY"] == v)]
        num_df = new_df[variables]
        corrMatrix = num_df.corr()
        corrValues = corrMatrix[variables[0]][corrMatrix[variables[0]] != 1]
        temp_df = pd.DataFrame({
            "VEHICLE_CATEGORY": f'{v}',
            "PEARSON_CORR": corrValues
        })
        corr_df = pd.concat([corr_df, temp_df], ignore_index=True)
    print(f"Pearson Correlation between {variables[0]} and {variables[1]}")
    print(corr_df)
    return

def make_lineGraph(df, independent, filename):
    grouped = (df.groupby(["VEHICLE_CATEGORY", "SPEED_ZONE"])
    .agg(mean_injury=(independent, "mean"), count=(independent, "count"))
    .reset_index())
    
    filtered = grouped[grouped["count"] >= 30]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=filtered, x="SPEED_ZONE", y="mean_injury", hue="VEHICLE_CATEGORY")
    plt.title(f"Mean {independent} by SPEED_ZONE and Vehicle Type (Filtered: count ≥ 30)")
    plt.ylabel(f"Mean {independent}")
    plt.xlabel("SPEED_ZONE")
    plt.tight_layout()
    plt.savefig(filename)
    return

def mutual_info(df, dependent, independent):
    mi_df = pd.DataFrame(columns=["VEHICLE_CATEGORY", "CONDITIONS", "MUTUAL_INFORMATION"])
    for v in vehicle_types:
        new_df = df[df["VEHICLE_CATEGORY"] == v]
        x_vals = new_df[dependent]
        y = new_df[independent]
        x_encoded = OrdinalEncoder().fit_transform(x_vals)
        #Finding the mutual information between the independent and dependent values.
        mut_inf = mutual_info_classif(x_encoded, y, discrete_features=True)
        temp_df = pd.DataFrame({
        'VEHICLE_CATEGORY': [f'{v}']*len(dependent),
        'CONDITIONS': x_vals.columns,
        'MUTUAL_INFORMATION': mut_inf})
        mi_df = pd.concat([mi_df, temp_df], ignore_index=True)
    mi_df = mi_df.sort_values(by='MUTUAL_INFORMATION', ascending=False)
    return mi_df

def make_barGraph(df, independent, filename):
    g = sns.catplot(data=df, x="CONDITIONS", y="MUTUAL_INFORMATION", hue="VEHICLE_CATEGORY", kind="bar", height=5, aspect=2, 
        palette="YlOrRd")
    plt.title(f"MI between {independent} and different CONDITIONS")
    g.set_axis_labels("CONDITIONS", "MUTUAL_INFORMATION")
    plt.tight_layout()
    plt.savefig(filename)
    return


ailMI = mutual_info(everything, ["LIGHT_LEVEL", "ROAD_TYPE", "RAIN/SNOW", "FOG/SMOKE/DUST", "CLEAR"], "AVERAGE_INJ_LEVEL")
vdlMI = mutual_info(everything, ["LIGHT_LEVEL", "ROAD_TYPE", "RAIN/SNOW", "FOG/SMOKE/DUST", "CLEAR"], "VEHICLE_DAMAGE_LEVEL")
make_barGraph(ailMI,'AVERAGE_INJ_LEVEL', 'AILMutInfo.png')
make_barGraph(vdlMI, 'VEHICLE_DMG_LEVEL', 'VDLMutInfo.png')


pearson_corr(everything, ["SPEED_ZONE", "AVERAGE_INJ_LEVEL"])
pearson_corr(everything, ["SPEED_ZONE", "VEHICLE_DAMAGE_LEVEL"])
make_lineGraph(everything, "VEHICLE_DAMAGE_LEVEL", "VDLPearson.png")
make_lineGraph(everything, "AVERAGE_INJ_LEVEL", "AILPearson.png")

