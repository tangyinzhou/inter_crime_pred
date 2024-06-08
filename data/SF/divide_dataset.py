import pandas as pd
import geopandas as gpd
from tqdm import *
import json

pd_dict={}
gdf = gpd.read_file(
    "/home/tangyinzhou/inter_crime_pred/data/SF/community.geojson"
)  # 替换为你的GeoJSON文件路径

cat_list = ['SUSPICIOUS OCC', 'DRUG/NARCOTIC', 'LARCENY/THEFT', 'OTHER OFFENSES', 'FRAUD', 'ASSAULT', 'BURGLARY', 'ROBBERY', 'NON-CRIMINAL', 'KIDNAPPING', 'PROSTITUTION', 'RECOVERED VEHICLE', 'TRESPASS', 'VANDALISM', 'MISSING PERSON', 'DRUNKENNESS', 'VEHICLE THEFT', 'WARRANTS', 'DISORDERLY CONDUCT', 'FORGERY/COUNTERFEITING', 'WEAPON LAWS', 'SEX OFFENSES, FORCIBLE', 'STOLEN PROPERTY', 'SECONDARY CODES', 'DRIVING UNDER THE INFLUENCE', 'EMBEZZLEMENT', 'ARSON', 'LIQUOR LAWS', 'SUICIDE', 'BRIBERY', 'BAD CHECKS', 'GAMBLING', 'LOITERING', 'EXTORTION', 'PORNOGRAPHY/OBSCENE MAT', 'SEX OFFENSES, NON FORCIBLE']
for year in trange(2006, 2018):
    data = pd.read_csv('/home/tangyinzhou/inter_crime_pred/data/SF/crime_{0}.csv'.format(year))
    data['datetime'] = pd.to_datetime(data['time'], format='%Y-%m-%d-%H:%M')
    for month in range(1,13):
        pd_dict['{0}-{1}'.format(year, month)] = {}
        month_data = data[data['datetime'].dt.month == month]
        for cat in cat_list:
            cat_data = month_data[month_data['category'] == cat]
            pd_dict['{0}-{1}'.format(year, month)][cat] = {}
            for area in range(len(gdf)):
                area_data = cat_data[cat_data['area'] == area]
                if not area_data.empty:
                    pd_dict['{0}-{1}'.format(year, month)][cat][area] = len(area_data)
                else:
                    pd_dict['{0}-{1}'.format(year, month)][cat][area] = 0
with open('/home/tangyinzhou/inter_crime_pred/data/SF/dataset.json', 'w') as f:
    json.dump(pd_dict, f)