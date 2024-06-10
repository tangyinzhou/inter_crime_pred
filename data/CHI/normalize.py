import json
import numpy as np

catgory_list = [
    "THEFT",
    "DECEPTIVE PRACTICE",
    "CRIMINAL DAMAGE",
    "CRIMINAL SEXUAL ASSAULT",
    "BURGLARY",
    "HOMICIDE",
    "ASSAULT",
    "ROBBERY",
    "OFFENSE INVOLVING CHILDREN",
    "BATTERY",
    "CRIM SEXUAL ASSAULT",
    "SEX OFFENSE",
    "MOTOR VEHICLE THEFT",
    "CRIMINAL TRESPASS",
    "NARCOTICS",
    "PROSTITUTION",
    "WEAPONS VIOLATION",
    "ARSON",
    "KIDNAPPING",
    "PUBLIC PEACE VIOLATION",
    "GAMBLING",
    "LIQUOR LAW VIOLATION",
    "RITUALISM",
    "INTIMIDATION",
    "INTERFERENCE WITH PUBLIC OFFICER",
    "OTHER NARCOTIC VIOLATION",
    "STALKING",
    "PUBLIC INDECENCY",
    "OBSCENITY",
    "DOMESTIC VIOLENCE",
]
with open("inter_crime_pred/data/CHI/dataset.json", "r") as f:
    data = json.load(f)
train_data = []
for year in range(2001, 2021):
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
with open("inter_crime_pred/data/CHI/normalize_param.json", "w") as f:
    json.dump(save, f)

pd_dict = {}
for year_index, year in enumerate(range(2001, 2021)):
    for month_index, month in enumerate(range(1, 13)):
        pd_dict["{0}-{1}".format(year, month)] = {}
        for cat_index, cat in enumerate(catgory_list):
            pd_dict["{0}-{1}".format(year, month)][cat] = {}
            for area in range(77):
                pd_dict["{0}-{1}".format(year, month)][cat][area] = normalized_data[
                    year_index * 12 + month_index, area, cat_index
                ]
with open(
    "/home/tangyinzhou/inter_crime_pred/data/CHI/dataset_normalized.json", "w"
) as f:
    json.dump(pd_dict, f)
