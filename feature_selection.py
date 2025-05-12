import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
import pre

v_path = "./vehicle.csv"
a_path = "./accident.csv"
p_path = "./person.csv"
atmo_path = "./atmospheric_cond.csv"
# df = pre.everything_df(v_path, a_path, atmo_path, p_path)
df = pd.read_csv("./table.csv")

damage_level_dict = {
    0: "No damage",
    1: "Minor",
    2: "Moderate (driveable vehicle)",
    3: "Moderate (unit towed away)",
    4: "Major (unit towed away)",
    5: "Extensive (unrepairable)"
}

df["VEHICLE_DAMAGE_DESC"] = df["VEHICLE_DAMAGE_LEVEL"].map(damage_level_dict)

injury_level_dict = {
    3: "FATALITY",
    2: "SERIOUS_INJURY",
    1: "OTHER_INJURY",
    0: "NO_INJURY"
}

df["INJURY_LEVEL_DESC"] = df["AVERAGE_INJ_LEVEL"].map(injury_level_dict)


def num_independent(df, independent, dependent):
    if not pd.api.types.is_numeric_dtype(df[independent]):
        return False
    return df[independent].corr(df[dependent])

def cate_independent(df, independent, dependent):
    already_encoded = [col for col in independent if df[col].nunique() <= 2 and set(df[col].unique()) <= {0, 1}]
    to_encode = [col for col in independent if col not in already_encoded]

    x_cat = df[to_encode].astype(str).fillna("Missing")
    x_encoded = OrdinalEncoder().fit_transform(x_cat)
    df_encoded = pd.DataFrame(x_encoded, columns=to_encode, index=df.index)

    x_all = pd.concat([df_encoded, df[already_encoded]], axis=1)

    y = df[dependent]
    mi_scores = mutual_info_classif(x_all, y, discrete_features=True)

    print(f"\nMutual Information with dependent: {dependent}")
    for feature, score in zip(x_all.columns, mi_scores):
        print(f"{feature}: {score:.4f}")

print("Pearson Correlation:")
print("Independent: LIGHT_LEVEL Dependent: VEHICLE_DAMAGE_LEVEL", num_independent(df, "LIGHT_LEVEL", "VEHICLE_DAMAGE_LEVEL"))
print("Independent: LIGHT_LEVEL Dependent: AVERAGE_INJ_LEVEL", num_independent(df, "LIGHT_LEVEL", "AVERAGE_INJ_LEVEL"))

print("\nAVERAGE_INJ_LEVEL: (Overall)")
cate_independent(df, ["VEHICLE_CATEGORY", "ACCIDENT_TYPE", "LIGHT_LEVEL", 
                      "ROAD_TYPE", "ROAD_GEOMETRY_DESC", "ROAD_SURFACE_TYPE_DESC",
                      "ATMOSPH_COND_DESC_Clear", "ATMOSPH_COND_DESC_Dust", "ATMOSPH_COND_DESC_Fog",
                      "ATMOSPH_COND_DESC_Raining", "ATMOSPH_COND_DESC_Smoke", "ATMOSPH_COND_DESC_Snowing", 
                      "ATMOSPH_COND_DESC_Strong winds"], "INJURY_LEVEL_DESC")
print("VEHICLE_DAMAGE_LEVEL: (Overall)")
cate_independent(df, ["VEHICLE_CATEGORY", "ACCIDENT_TYPE", "LIGHT_LEVEL", 
                      "ROAD_TYPE", "ROAD_GEOMETRY_DESC", "ROAD_SURFACE_TYPE_DESC",
                      "ATMOSPH_COND_DESC_Clear", "ATMOSPH_COND_DESC_Dust", "ATMOSPH_COND_DESC_Fog",
                      "ATMOSPH_COND_DESC_Raining", "ATMOSPH_COND_DESC_Smoke", "ATMOSPH_COND_DESC_Snowing", 
                      "ATMOSPH_COND_DESC_Strong winds"], "VEHICLE_DAMAGE_DESC")

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

# Pearson Correlation for SPEED_ZONE
for v_type in vehicle_types:
    subset = df[df["VEHICLE_CATEGORY"] == v_type]
    print(num_independent(subset, "SPEED_ZONE", "AVERAGE_INJ_LEVEL"))
    print(num_independent(subset, "SPEED_ZONE", "VEHICLE_DAMAGE_LEVEL"))

for v_type in vehicle_types:
    subset = df[df["VEHICLE_CATEGORY"] == v_type]
    print(f"\n===== {v_type} =====")
    print("AVERAGE_INJ_LEVEL: (Overall)")
    cate_independent(subset, ["VEHICLE_CATEGORY", "ACCIDENT_TYPE", "LIGHT_LEVEL", 
                              "ROAD_TYPE", "ROAD_GEOMETRY_DESC", "ROAD_SURFACE_TYPE_DESC",
                              "ATMOSPH_COND_DESC_Clear", "ATMOSPH_COND_DESC_Dust", "ATMOSPH_COND_DESC_Fog",
                              "ATMOSPH_COND_DESC_Raining", "ATMOSPH_COND_DESC_Smoke", "ATMOSPH_COND_DESC_Snowing", 
                              "ATMOSPH_COND_DESC_Strong winds"], "INJURY_LEVEL_DESC")
    print("VEHICLE_DAMAGE_LEVEL: (Overall)")
    cate_independent(subset, ["VEHICLE_CATEGORY", "ACCIDENT_TYPE", "LIGHT_LEVEL", 
                              "ROAD_TYPE", "ROAD_GEOMETRY_DESC", "ROAD_SURFACE_TYPE_DESC",
                              "ATMOSPH_COND_DESC_Clear", "ATMOSPH_COND_DESC_Dust", "ATMOSPH_COND_DESC_Fog",
                              "ATMOSPH_COND_DESC_Raining", "ATMOSPH_COND_DESC_Smoke", "ATMOSPH_COND_DESC_Snowing", 
                              "ATMOSPH_COND_DESC_Strong winds"], "VEHICLE_DAMAGE_DESC")
