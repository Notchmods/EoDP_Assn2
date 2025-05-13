"""
Read it before you start your part:

Question: What are the differences in traffic accident outcomes among various vehicle types 
          under different road and environmental conditions?

- Please download data from https://discover.data.vic.gov.au/dataset/victoria-road-crash-data
  In this assignment, we are currently using these data file: 
    Accident, 
    Vehicle, 
    Atmospheric Condition, 
    Person
  If you reckon any other files could be included, pls discuss with all group members.

- For accident environments:
    We mainly focus on the following factor:
        - road geometry
        - light level
        - road surface
        - Atmospheric Condition

- For accident outcome:
    We mainly focus on the following factor:
        - vehicle damage level
        - number of people (driver or passenger) injured from the accident
            Notes: There are counts for diffrent level of injury. 
        - average injury level across driver and all passengers

Some Questions I'm not sure:
- number of people (driver or passenger) injured VS. average injury level
    There two are basically same things. I think you only need to use one of them to reflect the 
    safety of the car under a certain condition. Considering which one to use, I think just choose
    the one better/easier for your task. The reason I kept both is just providing your another 
    option (hopefully is helpful to you)

- How should we measure the outcome of an accident?
    Just my opinion, we could analyze how structurally vulnerable or robust the vehicle is (i.e., how 
    easily it gets damaged or the structural integrity). Another use of damage level could be a 
    factor to reflect the servity of the accidents. I think the correlation between damage level and 
    injury under different condition could tell us something.

- Frequency:
    I didn't do anything with the frequency counts (sorry for that). It should be very easy. I think 
    it's better to have that as well. It could answer some questions like "Which type of vehicle is 
    more dangerous when there is no light?"

A few possible problems we could solve:
- How does road geometry influence accident severity across different vehicle types?
- Do different light conditions (e.g., day, night, dusk) impact injury severity differently across vehicle types?
- Which vehicle types are more prone to severe outcomes on poor road surfaces (e.g., gravel, unpaved)?
- Under adverse weather (e.g., heavy rain, fog), which vehicle types experience the highest increase in fatality rate?
- Are rear-seat passengers more vulnerable in certain vehicle types (e.g., Light Commercial vs. Passenger Cars)?
- Are some vehicle types consistently more dangerous for front-seat passengers?
- Does higher vehicle damage always correlate with higher injury levels? Does this vary by vehicle type?
- Are there vehicles with high structural damage but surprisingly low passenger injuries—or the opposite?
- Are there specific combinations of conditions (e.g., night + wet road + motorcycle) that result in extreme injury outcomes?
- Are some vehicle types generally safer regardless of road and environmental conditions?
- Does the number of occupants in a vehicle influence the injury outcome? For example, are rear-seat injuries worse when more people are seated?
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re

reordered_INJ_dict = {
    1: 3,
    2: 2,
    3: 1,
    4: 0
}

inj_name_dict = {
    3: "FATALITY_COUNT",
    2: "SERIOUS_INJ_COUNT",
    1: "OTHER_INJ_COUNT",
    0: "NO_INJ_COUNT",
}


level_of_damage_dict = {
    1: "Minor",
    2: "Moderate (driveable vehicle)",
    3: "Moderate (unit towed away)",
    4: "Major (unit towed away)",
    5: "Extensive (unrepairable)",
    6: "No damage"
}

front = ["LF", "CF", "PL", "D"]
UK_position = ["NA", "NK"]

unknown_type = ["Unknown", "Not Applicable", "Not Known"]

Unwanted_type = ["Electric Device", "Horse (ridden or drawn)",
                  "Plant machinery and Agricultural equipment",
                  "Parked trailers", "Other Vehicle"]

model_to_type = {}
body_to_type = {}

accident_type_dict = {0: "Pedestrian on foot in toy/pram",
    1: "Vehicles from adjacent directions (intersections only)",
    2: "Vehicles from opposing directions",
    3: "Vehicles from same direction",
    4: "Manoeuvring",
    5: "Overtaking",
    6: "On path",
    7: "Off path on straight",
    8: "Off path on curve",
    9: "Passenger and miscellaneous"}

light_code = {
    "Day": 1,
    "Dusk/dawn": 2,
    "Dark street lights on": 3,
    "Dark street lights off": 4,
    "Dark no street lights": 5,
    "Dark street lights unknown": 6,
    "Unknown": 9
}

reordered_light_dict = {
    light_code["Day"]: 3,
    light_code["Dusk/dawn"]: 2,
    light_code["Dark street lights on"]: 1,
    light_code["Dark street lights off"]: 0,
    light_code["Dark no street lights"]: 0
}

weather_priority_map = {
    "Raining": 0,
    "Snowing": 1,
    "Fog": 2,
    "Smoke": 3,
    "Dust": 4,
    "Strong winds": 5,
    "Clear": 6
}

def simplify_geometry(desc):
    if desc in ["Not at intersection", "Dead end", "Private property", "Road closure"]:
        return "Not at intersection"
    elif desc in ["Cross intersection", "T intersection"]:
        return "Standard intersection"
    else:
        return "Complex intersection"

# For testing:
v_path = "./vehicle.csv"
a_path = "./accident.csv"
p_path = "./person.csv"
atmo_path = "./atmospheric_cond.csv"

def encoded_column(df, columns):
    """
    Used for replace the column in df, "column_name" with one hot encoding columns
    For example:
    environment = encoded_column(environment, ["ROAD_SURFACE_TYPE_DESC", "LIGHT_LEVEL"])
    The column "ROAD_SURFACE_TYPE_DESC" would be on hot encoded
    """
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    encoder.fit(df[columns])
    encoded_df = encoder.transform(df[columns])
    df = df.drop(columns=columns)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def time_to_light_condition(time):
    """
    Given a time string in "HH:MM" format, return the corresponding light condition:
    "Day", "Dusk", or "Dark"
    """

    # Transform to mins
    hour = time.hour
    minute = time.minute
    total_minutes = hour * 60 + minute

    if 7 * 60 <= total_minutes < 18 * 60:
        return light_code["Day"]
    elif 18 * 60 <= total_minutes < 20 * 60 or 5 * 60 <= total_minutes < 7 * 60:
        return light_code["Dusk/dawn"]
    else:
        return light_code["Dark street lights unknown"]

def resolve_unknown_light(row, df):
    light = row["LIGHT_CONDITION"]

    if light == light_code["Unknown"]:
        light = time_to_light_condition(row["ACCIDENT_TIME"])

    if light == light_code["Dark street lights unknown"]:
        temp = df[(df["NODE_ID"] == row["NODE_ID"]) 
                   & (df["LIGHT_CONDITION"].isin([light_code["Dark street lights on"], 
                                                  light_code["Dark street lights off"]]))]
        if not temp.empty:
            most_common = temp["LIGHT_CONDITION"].value_counts().idxmax()
            return most_common
    return light

def resolve_unknown_surface(row, df):
    surface = row["ROAD_SURFACE_TYPE_DESC"]
    if pd.isna(surface) or surface == "Not known":
        df = df[df["NODE_ID"] == row["NODE_ID"]]
        freq = df["ROAD_SURFACE_TYPE_DESC"].value_counts()
        if freq.index[0] != "Unknown":
            return freq.index[0]
        elif len(freq) > 1:
            return freq.index[1]
        else:
            return freq.index[0]
    return surface
    
def environment_df(vehicle_path, accident_path, atomosphere_path):
    """
    This function will take address of vehicle.csv and accident.csv as input, 
    remove all rows with null (or unknown) value, 
    and return a df about all environment factor, which has the following columns:

        - ACCIDENT_NO

        - ROAD_GEOMETRY_DESC: Dtype of road structure where the accident occurred.
            Contained value types:
                - "Not at intersection"
                - "Standard intersection"
                - "Complex intersection"

        - LIGHT_LEVEL: Indicates the lighting at the time of the accident, 
                       encoded numerically, higher->lighter.
            Contained value types:
                {3: "Day", 
                 2: "Dusk/dawn", 
                 1: "Dark - street lights on",
                 0: "Dark - no street lights"}

        - ROAD_TYPE: type of road where the accident occurred.
            Contained value types:
                ['Arterial Other', 'Local Road', 'Arterial Highway', 'Freeway', 'Non Arterial']

        - ROAD_SURFACE_TYPE_DESC: Describes the type of road surface at the site of the accident.
            Contained value types:
                ['Paved', 'Gravel', 'Unpaved']

        - ATMOSPH_COND: Weather and atmospheric conditions at the time of	
            Contained value types:
                [Clear, Raining, Snowing, Fog, Smoke, Dust, Strong winds]
]

    """
    # Read from accident:
    accident = pd.read_csv(accident_path, usecols=["ACCIDENT_NO", "ACCIDENT_TIME", "NODE_ID", "LIGHT_CONDITION", "ROAD_GEOMETRY_DESC", "RMA", "SPEED_ZONE"])
    
    # Remove all Unknown ROAD_GEOMETRY_DESC and Unknown or other speed zone
    accident = accident[(accident["ROAD_GEOMETRY_DESC"] != "Unknown") & (~accident["SPEED_ZONE"].isin([777, 888, 999]))]
    # due to insufficient size of speed zone 75, merge 75 to 80
    accident["SPEED_ZONE"] = accident["SPEED_ZONE"].replace({75: 80})
    # Simplify ROAD_GEOMETRY_DESC
    accident["ROAD_GEOMETRY_DESC"] = accident["ROAD_GEOMETRY_DESC"].apply(simplify_geometry)

    # replace unknown light level with predicted value based on location and time as much as we can
    accident["ACCIDENT_TIME"] = pd.to_datetime(accident["ACCIDENT_TIME"], format="%H:%M:%S")
    accident["LIGHT_CONDITION"] = accident.apply(lambda row: resolve_unknown_light(row, accident), axis=1)
    accident = accident[accident["LIGHT_CONDITION"] != light_code["Dark street lights unknown"]]
    # level encoding LIGHT_CONDITION
    accident["LIGHT_CONDITION"] = accident["LIGHT_CONDITION"].map(reordered_light_dict)
    accident = accident.rename(columns={"RMA": "ROAD_TYPE", "LIGHT_CONDITION": "LIGHT_LEVEL"})

    accident = accident.drop(columns=["ACCIDENT_TIME"])

    
    # Read surface type:
    vehicle = pd.read_csv(vehicle_path, usecols=["ACCIDENT_NO", "ROAD_SURFACE_TYPE_DESC"])
    # remove all Null value
    vehicle = vehicle.drop_duplicates()
    # merge with accident (need to use NODE_ID to predict unknown surface type)
    environment = pd.merge(accident, vehicle, on = "ACCIDENT_NO")
    environment["ROAD_SURFACE_TYPE_DESC"] = environment.apply(lambda row: resolve_unknown_surface(row, environment), axis=1)
    environment = environment[(environment["ROAD_SURFACE_TYPE_DESC"] != "Not known")]
    environment = environment.drop(columns=["NODE_ID"])

    # Read weather:
    atmosphere = pd.read_csv(atomosphere_path, usecols=["ACCIDENT_NO", "ATMOSPH_COND_DESC"])
    # Remove Null value
    atmosphere = atmosphere[atmosphere["ATMOSPH_COND_DESC"] != "Not known"]
    # Select Main  (as we have multiple ATMOSPH_COND for the same accident)
    main_weather = (atmosphere.groupby("ACCIDENT_NO")["ATMOSPH_COND_DESC"]
                    .apply(lambda x: sorted(set(x), key=lambda w: weather_priority_map.get(w))[0])
                    .reset_index(name="MAIN_ATMOSPH_COND"))
    # One hot encoding (there are multiple ATMOSPH_COND for an accident)
    atmosphere = encoded_column(atmosphere, ["ATMOSPH_COND_DESC"])
    atmosphere = atmosphere.groupby("ACCIDENT_NO").max().reset_index()
    atmosphere["RAIN/SNOW"] = (atmosphere["ATMOSPH_COND_DESC_Raining"].astype(int) | atmosphere["ATMOSPH_COND_DESC_Snowing"].astype(int))
    atmosphere["FOG/SMOKE/DUST"] = (atmosphere["ATMOSPH_COND_DESC_Fog"].astype(int)
                                                                 | atmosphere["ATMOSPH_COND_DESC_Smoke"].astype(int) 
                                                                 | atmosphere["ATMOSPH_COND_DESC_Dust"].astype(int))
    atmosphere["CLEAR"] = atmosphere["ATMOSPH_COND_DESC_Clear"].astype(int)
    atmosphere = atmosphere.drop(columns=[col for col in atmosphere.columns if re.match(r"^ATMOSPH_COND_DESC_", col)])

    environment = pd.merge(environment, main_weather, on = "ACCIDENT_NO")
    environment = pd.merge(environment, atmosphere, on = "ACCIDENT_NO")
    environment = environment.dropna()
    # For viewing what each column is
    # print(environment.groupby(["LIGHT_CONDITION"]).size())
    # print(environment["ROAD_GEOMETRY_DESC"].unique())
    # print(environment["ROAD_TYPE"].unique())
    # print(environment["ROAD_SURFACE_TYPE_DESC"].unique())
    # environment.to_csv("environment.csv", index=False)
    return environment

def classify_vehicle_type(vtype):
    if vtype in ['Car', 'Taxi']:
        return 'Car'
    elif vtype in ['Light Commercial Vehicle (Rigid) <= 4.5 Tonnes GVM', 'Panel Van']:
        return 'Light Commercial Vehicle'
    elif vtype in ['Motor Cycle', 'Moped', 'Motor Scooter', 'Quad Bike']:
        return 'Motorcycle'
    elif vtype in ['Tram', 'Bus/Coach', 'Mini Bus(9-13 seats)', 'Train']:
        return 'Public Transport'
    elif vtype in ['Prime Mover Only', 'Prime Mover B-Double', 'Prime Mover B-Triple',
                   'Prime Mover - Single Trailer', 'Prime Mover (No of Trailers Unknown)']:
        return 'Heavy Truck'
    elif vtype in ['Heavy Vehicle (Rigid) > 4.5 Tonnes', 'Rigid Truck(Weight Unknown)']:
        return 'Heavy Truck'
    else:
        return vtype

def resolve_unknown_type(row, df):
    v_type = row["VEHICLE_TYPE_DESC"]

    if pd.isna(v_type) or v_type in unknown_type:
        model = row["VEHICLE_MODEL"]
        body = row["VEHICLE_BODY_STYLE"]

        # Predict type via model
        if not pd.isna(model):
            if model not in model_to_type.keys():
                subset = df[(df['VEHICLE_MODEL'] == model) & (~df['VEHICLE_TYPE_DESC'].isin(unknown_type))]
                if not subset.empty:
                    model_to_type[model] = subset['VEHICLE_TYPE_DESC'].value_counts().idxmax()
                else:
                    model_to_type[model] = "Unknown"
            v_type = model_to_type[model]

        # Predict type via body
        if v_type in unknown_type and not pd.isna(body):
            if model not in body_to_type.keys():
                subset = df[(df['VEHICLE_BODY_STYLE'] == body) & (~df['VEHICLE_TYPE_DESC'].isin(unknown_type))]
                if not subset.empty:
                    body_to_type[model] = subset['VEHICLE_TYPE_DESC'].value_counts().idxmax()
                else:
                    body_to_type[model] = "Unknown"
            v_type = body_to_type[model]

    return v_type

def vehicle_df(vehicle_path):
    """
    Notes: Don't use the output dataframe from this function. Use the outcome one.

    This function will take address of vehicle.csv as input, 
    remove unwanted types of vehicles, 
    and return a df about all vehicle, which has the following columns:
    - ACCIDENT_NO
    - VEHICLE_ID
    - VEHICLE_CATEGORY:
        1. 'Car'
        2. 'Station Wagon'
        3. 'Utility'
        4. 'Light Commercial Vehicle'
        5. 'Motorcycle': ['Motor Cycle', 'Moped', 'Motor Scooter', 'Quad Bike']
        6. 'Public Transport': ['Tram', 'Bus/Coach', 'Mini Bus(9-13 seats)', 'Train']
        7. 'Heavy Truck': ['Prime Mover Only', 'Prime Mover B-Double', 'Prime Mover B-Triple',
                   'Prime Mover - Single Trailer', 'Prime Mover (No of Trailers Unknown)', 
                   'Parked trailers', 'Heavy Vehicle (Rigid) > 4.5 Tonnes', 'Rigid Truck(Weight Unknown)']
        8. 'Bicycle'
    
    - VEHICLE_DAMAGE_LEVEL: Indicates the level of damage to the vehicle.
        Contained value types (after mapping):
            0: "No damage",
            1: "Minor",
            2: "Moderate (driveable vehicle)",
            3: "Moderate (unit towed away)",
            4: "Major (unit towed away)",
            5: "Extensive (unrepairable)"
    """
    vehicle = pd.read_csv(vehicle_path, usecols=["ACCIDENT_NO", "VEHICLE_ID", "VEHICLE_MODEL", "VEHICLE_BODY_STYLE", "VEHICLE_TYPE_DESC", "LEVEL_OF_DAMAGE"])
    # For viewing what each column is
    vehicle = vehicle[(~vehicle["VEHICLE_TYPE_DESC"].isin(Unwanted_type)) & (vehicle["LEVEL_OF_DAMAGE"] != 9)]
    # Predict unknown type as much as we can
    vehicle["VEHICLE_TYPE_DESC"] = vehicle.apply(lambda row: resolve_unknown_type(row, vehicle), axis=1)
    vehicle = vehicle[~vehicle["VEHICLE_TYPE_DESC"].isin(unknown_type)]
    vehicle = vehicle.rename(columns={"LEVEL_OF_DAMAGE": "VEHICLE_DAMAGE_LEVEL"})
    vehicle["VEHICLE_DAMAGE_LEVEL"] = vehicle["VEHICLE_DAMAGE_LEVEL"].replace(6, 0)
    # After predicting we can delete the columns only use for prediction
    vehicle.drop(columns=["VEHICLE_MODEL","VEHICLE_BODY_STYLE"], inplace=True)

    # Group similar vehicle types
    vehicle["VEHICLE_CATEGORY"] = vehicle["VEHICLE_TYPE_DESC"].apply(classify_vehicle_type)
    # For viewing:
    vehicle = vehicle.drop("VEHICLE_TYPE_DESC", axis=1)
    # print(vehicle["VEHICLE_CATEGORY"].unique())
    # print(vehicle[vehicle["VEHICLE_ID"].isna()])
    # print(vehicle.groupby(["VEHICLE_CATEGORY"]).size())
    return vehicle

def outcome_df(vehicle_path, person_path, accident_path):
    """
    This function merges and processes accident outcome data to produce a DataFrame 
    that combines vehicle damage and injury severity information. It removes incomplete 
    or irrelevant data and computes injury severity statistics.

    The resulting DataFrame includes the following columns:

        - ACCIDENT_NO

        - VEHICLE_ID
        
        - VEHICLE_DAMAGE_LEVEL: Indicates the level of damage to the vehicle.
        Contained value types (after mapping):
            0: "No damage",
            1: "Minor",
            2: "Moderate (driveable vehicle)",
            3: "Moderate (unit towed away)",
            4: "Major (unit towed away)",
            5: "Extensive (unrepairable)"

        - VEHICLE_CATEGORY:
            1. 'Car'
            2. 'Station Wagon'
            3. 'Utility'
            4. 'Panel Van'
            5. 'Light Commercial Vehicle'
            6. 'Motorcycle': ['Motor Cycle', 'Moped', 'Motor Scooter', 'Quad Bike']
            7. 'Public Transport': ['Tram', 'Bus/Coach', 'Mini Bus(9-13 seats)', 'Train']
            8. 'Prime Mover': ['Prime Mover Only', 'Prime Mover B-Double', 'Prime Mover B-Triple',
                       'Prime Mover - Single Trailer', 'Prime Mover (No of Trailers Unknown)', 
                       'Parked trailers']
            9. 'Heavy Truck': ['Heavy Vehicle (Rigid) > 4.5 Tonnes', 'Rigid Truck(Weight Unknown)']
            10. 'Bicycle'

        - NO_INJ_COUNT

        - OTHER_INJ_COUNT

        - SERIOUS_INJ_COUNT

        - FATALITY_COUNT

        - AVERAGE_INJ_LEVEL:
            3: "FATALITY",
            2: "SERIOUS_INJURY",
            1: "OTHER_INJURY",
            0: "NO_INJURY"

        - REAR_AVG_INJURY: Average injury level of the rear passenger

        - FRONT_AVG_INJURY: Average injury level of the front passenger / Driver
    """
    

    vehicle = vehicle_df(vehicle_path)
    person = pd.read_csv(person_path, usecols=["ACCIDENT_NO", "VEHICLE_ID", "INJ_LEVEL"])
    
    # Column: Accident Type
    accident = pd.read_csv(accident_path, usecols=["ACCIDENT_NO", "DCA_CODE"])
    accident["DCA_CODE"] = ((accident["DCA_CODE"] // 10) % 10).map(accident_type_dict)
    accident = accident.rename(columns={"DCA_CODE": "ACCIDENT_TYPE"})

    # delete all rows with VEHICLE_ID is null, they might be pedestrians (Not interested)
    person = person[~person["VEHICLE_ID"].isna()]
    person["INJ_LEVEL"] = person["INJ_LEVEL"].map(reordered_INJ_dict)

    # count injuried people
    injury_count= person.groupby(["ACCIDENT_NO", "VEHICLE_ID"])["INJ_LEVEL"].value_counts().unstack(fill_value=0).reset_index()
    injury_count = injury_count.rename(columns=inj_name_dict)

    # Find average injury level
    avg_injury_level = person.groupby(["ACCIDENT_NO", "VEHICLE_ID"])["INJ_LEVEL"].mean().reset_index(name="AVERAGE_INJ_LEVEL")
    avg_injury_level["AVERAGE_INJ_LEVEL"] = avg_injury_level["AVERAGE_INJ_LEVEL"].round().astype(int)

    # Merge together
    vehicle = pd.merge(vehicle, injury_count, on=["ACCIDENT_NO", "VEHICLE_ID"])
    vehicle = pd.merge(vehicle, avg_injury_level, on=["ACCIDENT_NO", "VEHICLE_ID"])
    vehicle = vehicle.drop_duplicates()
    vehicle = pd.merge(vehicle, accident, on=["ACCIDENT_NO"])
    # Viewing table:
    # vehicle.to_csv("my_vehicle.csv", index=False)
    return vehicle

def everything_df(vehicle_path, accident_path, atomosphere_path, person_path):
    environment = environment_df(vehicle_path, accident_path, atomosphere_path)
    outcome = outcome_df(vehicle_path, person_path, accident_path)
    # make sure you merge in this way if you merged by your self, not pd.merge(environment, outcome, ...)
    everything = pd.merge(outcome, environment, on=["ACCIDENT_NO"])
    #Add speed zones to table.csv
    speed_zones=pd.read_csv(accident_path)
    everything["SPEED_ZONE"]=speed_zones["SPEED_ZONE"]
    # For viewing
    everything.to_csv("table.csv", index=False)
    return everything

# For testing:
# outcome_df(v_path, a_path, p_path)
# vehicle_df(v_path)
# environment_df(v_path, a_path, atmo_path)
# everything_df(v_path, a_path, atmo_path, p_path)
