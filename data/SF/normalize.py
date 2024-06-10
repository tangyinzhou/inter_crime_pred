import json
import numpy as np

catgory_list = [
    "SUSPICIOUS OCC",
    "DRUG/NARCOTIC",
    "LARCENY/THEFT",
    "OTHER OFFENSES",
    "FRAUD",
    "ASSAULT",
    "BURGLARY",
    "ROBBERY",
    "NON-CRIMINAL",
    "KIDNAPPING",
    "PROSTITUTION",
    "RECOVERED VEHICLE",
    "TRESPASS",
    "VANDALISM",
    "MISSING PERSON",
    "DRUNKENNESS",
    "VEHICLE THEFT",
    "WARRANTS",
    "DISORDERLY CONDUCT",
    "FORGERY/COUNTERFEITING",
    "WEAPON LAWS",
    "SEX OFFENSES, FORCIBLE",
    "STOLEN PROPERTY",
    "SECONDARY CODES",
    "DRIVING UNDER THE INFLUENCE",
    "EMBEZZLEMENT",
    "ARSON",
    "LIQUOR LAWS",
    "SUICIDE",
    "BRIBERY",
    "BAD CHECKS",
    "GAMBLING",
    "LOITERING",
    "EXTORTION",
    "PORNOGRAPHY/OBSCENE MAT",
    "SEX OFFENSES, NON FORCIBLE",
]
with open("inter_crime_pred/data/SF/dataset.json", "r") as f:
    data = json.load(f)
train_data = []
for year in range(2006, 2018):
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
with open("inter_crime_pred/data/SF/normalize_param.json", "w") as f:
    json.dump(save, f)

pd_dict = {}
for year_index, year in enumerate(range(2006, 2018)):
    for month_index, month in enumerate(range(1, 13)):
        pd_dict["{0}-{1}".format(year, month)] = {}
        for cat_index, cat in enumerate(catgory_list):
            pd_dict["{0}-{1}".format(year, month)][cat] = {}
            for area in range(41):
                pd_dict["{0}-{1}".format(year, month)][cat][area] = normalized_data[
                    year_index * 12 + month_index, area, cat_index
                ]
with open(
    "/home/tangyinzhou/inter_crime_pred/data/SF/dataset_normalized.json", "w"
) as f:
    json.dump(pd_dict, f)
