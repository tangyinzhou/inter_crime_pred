import pandas as pd
import geopandas as gpd
from tqdm import *
import json

pd_dict={}
gdf = gpd.read_file(
    "/home/tangyinzhou/inter_crime_pred/data/CHI/community.geojson"
)  # 替换为你的GeoJSON文件路径

cat_list = ['THEFT', 'DECEPTIVE PRACTICE', 'CRIMINAL DAMAGE', 'CRIMINAL SEXUAL ASSAULT', 'BURGLARY', 'HOMICIDE', 'ASSAULT', 'ROBBERY', 'OFFENSE INVOLVING CHILDREN', 'BATTERY', 'CRIM SEXUAL ASSAULT', 'SEX OFFENSE', 'MOTOR VEHICLE THEFT', 'CRIMINAL TRESPASS', 'NARCOTICS', 'PROSTITUTION', 'WEAPONS VIOLATION', 'ARSON', 'KIDNAPPING', 'PUBLIC PEACE VIOLATION', 'GAMBLING', 'LIQUOR LAW VIOLATION', 'RITUALISM', 'INTIMIDATION', 'INTERFERENCE WITH PUBLIC OFFICER', 'OTHER NARCOTIC VIOLATION', 'STALKING', 'PUBLIC INDECENCY', 'OBSCENITY', 'DOMESTIC VIOLENCE']
for year in trange(2001, 2021):
    data = pd.read_csv('/home/tangyinzhou/inter_crime_pred/data/CHI/crime_{0}.csv'.format(year))
    data['datetime'] = pd.to_datetime(data['time'], format='%Y-%m-%d-%H:%M:%S')
    crime_type = data['category'].unique()
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
with open('/home/tangyinzhou/inter_crime_pred/data/CHI/dataset.json', 'w') as f:
    json.dump(pd_dict, f)