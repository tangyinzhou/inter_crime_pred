import json
import numpy as np

catgory_list = [
    "THEFT",
    "BURGLARY",
    "BATTERY",
    "CRIMINAL TRESPASS",
    "ROBBERY",
    "CRIMINAL DAMAGE",
    "MOTOR VEHICLE THEFT",
    "ASSAULT",
    "DECEPTIVE PRACTICE",
    "CRIM SEXUAL ASSAULT",
    "PUBLIC PEACE VIOLATION",
    "WEAPONS VIOLATION",
    "OFFENSE INVOLVING CHILDREN",
    "PROSTITUTION",
    "INTERFERENCE WITH PUBLIC OFFICER",
    "INTIMIDATION",
    "NARCOTICS",
    "ARSON",
    "SEX OFFENSE",
    "KIDNAPPING",
    "STALKING",
    "CRIMINAL SEXUAL ASSAULT",
    "HOMICIDE",
    "OTHER NARCOTIC VIOLATION",
    "LIQUOR LAW VIOLATION",
    "GAMBLING",
    "OBSCENITY",
    "PUBLIC INDECENCY",
    "RITUALISM",
]
with open("inter_crime_pred/data/NYC/dataset.json", "r") as f:
    data = json.load(f)
train_data = []
for year in range(2006, 2021):
    for month in range(1, 13):
        month_data = data["{0}-{1}".format(year, month)]
        month_list = []
        for cat, cat_data in month_data.items():
            cat_list = []
            for area, area_data in cat_data.items():
                cat_list.append(area_data)
            month_list.append(cat_list)
        month_list = np.array(month_list).T.tolist()
        train_data.append(month_list)
train_data = np.array(train_data)
mean = np.mean(train_data)
std = np.std(train_data)
normalized_data = (train_data - mean) / std
mean_to_recover = mean
std_to_recover = std
save = {"mean": mean_to_recover, "std": std_to_recover}
with open("inter_crime_pred/data/NYC/normalize_param.json", "w") as f:
    json.dump(save, f)

pd_dict = {}
for year_index, year in enumerate(range(2006, 2021)):
    for month_index, month in enumerate(range(1, 13)):
        pd_dict["{0}-{1}".format(year, month)] = {}
        for cat_index, cat in enumerate(catgory_list):
            pd_dict["{0}-{1}".format(year, month)][cat] = {}
            for area in range(262):
                pd_dict["{0}-{1}".format(year, month)][cat][area] = normalized_data[
                    year_index * 12 + month_index, area, cat_index
                ]
with open(
    "/home/tangyinzhou/inter_crime_pred/data/NYC/dataset_normalized.json", "w"
) as f:
    json.dump(pd_dict, f)
