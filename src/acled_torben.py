import pandas as pd
from src.constants import EXTERNAL_PATH

event_types = [
    "Strategic developments",
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Protests",
    "Riots",
]
sub_event_types = [
    "Remote explosive/landmine/IED",
    "Armed clash",
    "Shelling/artillery/missile attack",
    "Air/drone strike",
    "Attack" "Non-state actor overtakes territory",
    "Grenade",
    "Abduction/forced disappearance",
    "Government regains territory",
    "Sexual violence",
    "Suicide bomb",
]

xlsx = pd.read_excel(EXTERNAL_PATH / "ACLED__Ukraine_Black_Sea_2020_2023_Feb09.xlsx")
xlsx = xlsx[xlsx["COUNTRY"] == "Ukraine"]  # only Ukraine
xlsx["EVENT_DATE"] = pd.to_datetime(xlsx["EVENT_DATE"])
date_of_war = "2022-04-01"
xlsx = xlsx[xlsx["EVENT_DATE"] >= pd.to_datetime(date_of_war)]
print("num fatalaties", xlsx["FATALITIES"].sum())

specific_event_types = [event_types[1], event_types[2], event_types[3]]
xlsx = xlsx[xlsx["EVENT_TYPE"].isin(specific_event_types)]

specific_event_types = [
    # sub_event_types[0],
    # sub_event_types[1],
    sub_event_types[2],
    sub_event_types[3],
    # sub_event_types[4],
    # sub_event_types[5],
    # sub_event_types[9],
]

xlsx = xlsx[xlsx["SUB_EVENT_TYPE"].isin(specific_event_types)]

print("event_types", xlsx["EVENT_TYPE"].unique())
print("subn event_types", xlsx["SUB_EVENT_TYPE"].unique())

print("following cities are affected")
print(xlsx["EVENT_ID_CNTY"].unique(), len(xlsx["EVENT_ID_CNTY"].unique()))
print(xlsx["ADMIN1"].unique(), len(xlsx["ADMIN1"].unique()))
print(xlsx["ADMIN2"].unique(), len(xlsx["ADMIN2"].unique()))
print(xlsx["ADMIN3"].unique(), len(xlsx["ADMIN3"].unique()))
