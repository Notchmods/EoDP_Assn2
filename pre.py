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
        - road type (eg. freeway)
        - Atmospheric Condition

- For accident outcome:
    We mainly focus on the following factor:
        - Accident type
        - vehicle damage level
        - number of people (driver or passenger) injured from the accident
            Notes: There are counts for diffrent level of injury. 
        - average injury level across driver and all passengers
        - average injury level for the front seat and rear seat
            Notes: if you find these two columns containing null value, it may because:
                1. This type of vehicle does not have rear seat. 
                    Only   ['Car', 
                            'Station Wagon', 
                            'Taxi', 'Utility', 
                            'Panel Van',
                            'Utility', 
                            'Panel Van', 
                            'Light Commercial Vehicle']
                    has front seats and rear seats
                2. If it is null in the rear column, it could because no rear-seat passenger 
                   (DIFFERENT FROM NOT INJURY. if no one injury, it should be 0)

Some Questions I'm not sure:
- number of people (driver or passenger) injured VS. average injury level
    There two are basically same things. I think you only need to use one of them to reflect the 
    safety of the car under a certain condition. Considering which one to use, I think just choose
    the one better/easier for your task. The reason I kept both is just providing your another 
    option (hopefully is helpful to you)

- How should we measure the outcome of an accident?
    Just my opinion, we could analyse how structurally vulnerable or robust the vehicle is (i.e., how 
    easily it gets damaged or the structural integrity). Another use of damage level could be a 
    factor to reflect the servity of the accidents. I think the correlation between damage level and 
    injury under different condition could tell us something.

A few possible problems we could solve:
Damage Level:
- How do road surface, weather, and light level influence vehicle damage level across different vehicle types?

Injury Rate:
- Which type of vehicle is the safest (lowest injury level) overall? Under a specific weather, road type, light level?
  (comparing the correlation between Injury (level or number) and damage level)
- What is the most deadly combination of road surface, light, weather, and road type for each vehicle type?

Accident Type:
- Under a specific type of road surface, weather, light level, road level, what type of accident is the most frequently 
  happened on each type of vehicle?

Frequency:
- Under what combinations of vehicle type + road/environmental conditions do accidents occur most frequently?
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

reordered_light_dict = {
    1: 3,
    2: 2,
    3: 1,
    4: 0,
    5: 0
}

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

Unknown_light = [6,9]

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

Unwanted_type = ["Unknown", "Not Applicable", "Not Known",
                  "Electric Device", "Horse (ridden or drawn)",
                  "Plant machinery and Agricultural equipment",
                  "Parked trailers", "Other Vehicle"]

rear_seat_type = ['Car', 'Station Wagon', 'Taxi', 'Utility', 'Panel Van'
                'Utility', 'Panel Van', 'Light Commercial Vehicle']

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

# For testing:
# v_path = "./vehicle.csv"
# a_path = "./accident.csv"
# p_path = "./person.csv"
# atmo_path = "./atmospheric_cond.csv"

def encoded_column(df, column_name):
    """
    Used for replace the column in df, "column_name" with one hot encoding columns
    For example:
    environment = encoded_column(environment, "ROAD_SURFACE_TYPE_DESC")
    The column "ROAD_SURFACE_TYPE_DESC" would be on hot encoded
    """
    # Fit the encoder once on training data
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    encoder.fit(df[[column_name]])
    encoded_df = encoder.transform(df[[column_name]])
    df.drop(column_name, axis=1, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def environment_df(vehicle_path, accident_path, atomosphere_path):
    """
    This function will take address of vehicle.csv and accident.csv as input, 
    remove all rows with null (or unknown) value, 
    and return a df about all environment factor, which has the following columns:

        - ACCIDENT_NO

        - ROAD_GEOMETRY_DESC: Dtype of road structure where the accident occurred.
            Contained value types:
                ['T intersection', 'Not at intersection', 'Cross intersection',
                 'Multiple intersection', 'Y intersection', 'Dead end',
                 'Private property', 'Road closure']

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
    accident = pd.read_csv(accident_path, usecols=["ACCIDENT_NO", "LIGHT_CONDITION", "ROAD_GEOMETRY_DESC", "RMA"])
    # remove all Null value
    accident = accident.dropna()
    accident = accident[(accident["ROAD_GEOMETRY_DESC"] != "Unknown") & (~accident["LIGHT_CONDITION"].isin(Unknown_light))]
    accident["LIGHT_CONDITION"] = accident["LIGHT_CONDITION"].map(reordered_light_dict)
    accident = accident.rename(columns={"RMA": "ROAD_TYPE", "LIGHT_CONDITION": "LIGHT_LEVEL"})

    vehicle = pd.read_csv(vehicle_path, usecols=["ACCIDENT_NO", "ROAD_SURFACE_TYPE_DESC"])
    # remove all Null value
    vehicle = vehicle.drop_duplicates().dropna()
    vehicle = vehicle[vehicle["ROAD_SURFACE_TYPE_DESC"] != "Not known"]

    atmosphere = pd.read_csv(atomosphere_path, usecols=["ACCIDENT_NO", "ATMOSPH_COND_DESC"])
    atmosphere = atmosphere[atmosphere["ATMOSPH_COND_DESC"] != "Not known"]
    atmosphere = atmosphere.rename(columns={"ATMOSPH_COND_DESC": "ATMOSPH_COND"})
    environment = pd.merge(accident, vehicle, on = "ACCIDENT_NO")
    environment = pd.merge(environment, atmosphere, on = "ACCIDENT_NO")
    environment = environment.dropna()
    # For viewing what each column is
    # print(environment.groupby(["LIGHT_CONDITION"]).size())
    # print(environment["ROAD_GEOMETRY_DESC"].unique())
    # print(environment["ROAD_TYPE"].unique())
    # print(environment["ROAD_SURFACE_TYPE_DESC"].unique())
    # environment.to_csv("environment.csv", index=False)
    return environment

def classify_vehicle_size(vtype):
    if vtype in ['Light Commercial Vehicle (Rigid) <= 4.5 Tonnes GVM']:
        return 'Light Commercial Vehicle'
    elif vtype in ['Motor Cycle', 'Moped', 'Motor Scooter', 'Quad Bike']:
        return 'Motorcycle'
    elif vtype in ['Tram', 'Bus/Coach', 'Mini Bus(9-13 seats)', 'Train']:
        return 'Public Transport'
    elif vtype in ['Prime Mover Only', 'Prime Mover B-Double', 'Prime Mover B-Triple',
                   'Prime Mover - Single Trailer', 'Prime Mover (No of Trailers Unknown)', 
                   'Parked trailers']:
        return 'Prime Mover'
    elif vtype in ['Heavy Vehicle (Rigid) > 4.5 Tonnes', 'Rigid Truck(Weight Unknown)']:
        return 'Heavy Truck'
    else:
        return vtype

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
        3. 'Taxi'
        4. 'Utility'
        5. 'Panel Van'
        6. 'Utility'
        7. 'Panel Van'
        8. 'Light Commercial Vehicle'
        3. 'Motorcycle': ['Motor Cycle', 'Moped', 'Motor Scooter', 'Quad Bike']
        4. 'Public Transport': ['Tram', 'Bus/Coach', 'Mini Bus(9-13 seats)', 'Train']
        5. 'Prime Mover': ['Prime Mover Only', 'Prime Mover B-Double', 'Prime Mover B-Triple',
                   'Prime Mover - Single Trailer', 'Prime Mover (No of Trailers Unknown)', 
                   'Parked trailers']
        6. 'Heavy Truck': ['Heavy Vehicle (Rigid) > 4.5 Tonnes', 'Rigid Truck(Weight Unknown)']
        7. 'Bicycle'
    
    - VEHICLE_DAMAGE_LEVEL: Indicates the level of damage to the vehicle.
        Contained value types (after mapping):
            0: "No damage",
            1: "Minor",
            2: "Moderate (driveable vehicle)",
            3: "Moderate (unit towed away)",
            4: "Major (unit towed away)",
            5: "Extensive (unrepairable)"
    """
    vehicle = pd.read_csv(vehicle_path, usecols=["ACCIDENT_NO", "VEHICLE_ID", "VEHICLE_TYPE_DESC", "LEVEL_OF_DAMAGE"])
    # For viewing what each column is
    vehicle = vehicle[(~vehicle["VEHICLE_TYPE_DESC"].isin(Unwanted_type)) & (vehicle["LEVEL_OF_DAMAGE"] != 9)]
    vehicle = vehicle.rename(columns={"LEVEL_OF_DAMAGE": "VEHICLE_DAMAGE_LEVEL"})
    vehicle["VEHICLE_DAMAGE_LEVEL"] = vehicle["VEHICLE_DAMAGE_LEVEL"].replace(6, 0)
    vehicle["VEHICLE_CATEGORY"] = vehicle["VEHICLE_TYPE_DESC"].apply(classify_vehicle_size)
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
            3. 'Taxi'
            4. 'Utility'
            5. 'Panel Van'
            6. 'Utility'
            7. 'Panel Van'
            8. 'Light Commercial Vehicle'
            3. 'Motorcycle': ['Motor Cycle', 'Moped', 'Motor Scooter', 'Quad Bike']
            4. 'Public Transport': ['Tram', 'Bus/Coach', 'Mini Bus(9-13 seats)', 'Train']
            5. 'Prime Mover': ['Prime Mover Only', 'Prime Mover B-Double', 'Prime Mover B-Triple',
                       'Prime Mover - Single Trailer', 'Prime Mover (No of Trailers Unknown)', 
                       'Parked trailers']
            6. 'Heavy Truck': ['Heavy Vehicle (Rigid) > 4.5 Tonnes', 'Rigid Truck(Weight Unknown)']
            7. 'Bicycle'

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

        - ACCIDENT_TYPE:
            "Pedestrian on foot in toy/pram",
            "Vehicles from adjacent directions (intersections only)",
            "Vehicles from opposing directions",
            "Vehicles from same direction",
            "Manoeuvring",
            "Overtaking",
            "On path",
            "Off path on straight",
            "Off path on curve",
            "Passenger and miscellaneous"
    """
    

    vehicle = vehicle_df(vehicle_path)
    person = pd.read_csv(person_path, usecols=["ACCIDENT_NO", "VEHICLE_ID", "SEATING_POSITION", "INJ_LEVEL"])
    
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

    # Find injury level
    avg_injury_level = person.groupby(["ACCIDENT_NO", "VEHICLE_ID"])["INJ_LEVEL"].mean().reset_index(name="AVERAGE_INJ_LEVEL")
    person["FRONT"] = person["SEATING_POSITION"].isin(front)
    person.drop(columns=["SEATING_POSITION"])
    vehicle = pd.merge(vehicle, person, on=["ACCIDENT_NO", "VEHICLE_ID"])
    vehicle_by_seat = vehicle[vehicle["VEHICLE_CATEGORY"].isin(rear_seat_type)]
    avg_injury_by_seat = vehicle_by_seat.groupby(["ACCIDENT_NO", "VEHICLE_ID", "FRONT"])["INJ_LEVEL"].mean().unstack().reset_index()
    avg_injury_by_seat = avg_injury_by_seat.rename(columns={True: "FRONT_AVG_INJURY", False: "REAR_AVG_INJURY"})

    vehicle = pd.merge(vehicle, injury_count, on=["ACCIDENT_NO", "VEHICLE_ID"])
    vehicle = pd.merge(vehicle, avg_injury_level, on=["ACCIDENT_NO", "VEHICLE_ID"])
    vehicle = pd.merge(vehicle, avg_injury_by_seat, on=["ACCIDENT_NO", "VEHICLE_ID"])
    vehicle = vehicle.drop(["SEATING_POSITION", "INJ_LEVEL", "FRONT"], axis=1)
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
    everything.to_csv("table.csv", index=False)
    return everything

# For testing:
# outcome_df(v_path, a_path, p_path)
# vehicle_df(v_path)
# environment_df(v_path, a_path, atmo_path)
# everything_df(v_path, a_path, atmo_path, p_path)