import pandas as pd
import geopandas as gpd
from tqdm import *
import json

pd_dict={}
gdf = gpd.read_file(
    "/home/tangyinzhou/inter_crime_pred/data/NYC/community.geojson"
)  # 替换为你的GeoJSON文件路径

cat_list = ['THEFT', 'BURGLARY', 'BATTERY', 'CRIMINAL TRESPASS', 'ROBBERY', 'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT', 'ASSAULT', 'DECEPTIVE PRACTICE', 'CRIM SEXUAL ASSAULT', 'PUBLIC PEACE VIOLATION', 'WEAPONS VIOLATION', 'OFFENSE INVOLVING CHILDREN', 'PROSTITUTION', 'INTERFERENCE WITH PUBLIC OFFICER', 'INTIMIDATION', 'NARCOTICS', 'ARSON', 'SEX OFFENSE', 'KIDNAPPING', 'STALKING', 'CRIMINAL SEXUAL ASSAULT', 'HOMICIDE', 'OTHER NARCOTIC VIOLATION', 'LIQUOR LAW VIOLATION', 'GAMBLING', 'OBSCENITY', 'PUBLIC INDECENCY', 'RITUALISM']
for year in trange(2006, 2021):
    data = pd.read_csv('/home/tangyinzhou/inter_crime_pred/data/NYC/crime_{0}.csv'.format(year))
    data['datetime'] = pd.to_datetime(data['time'], format='%Y-%m-%d')
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
with open('/home/tangyinzhou/inter_crime_pred/data/NYC/dataset.json', 'w') as f:
    json.dump(pd_dict, f)