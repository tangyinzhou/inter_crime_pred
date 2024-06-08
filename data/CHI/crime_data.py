# import pandas as pd
# from datetime import datetime


# def convert_datetime(date_str):
#     # 使用strptime将字符串转换为datetime对象
#     dt = datetime.strptime(date_str, "%m/%d/%Y %I:%M:%S %p")
#     # 格式化为新的日期时间格式
#     return dt.strftime("%Y-%m-%d-%H:%M:%S")


# data = pd.read_csv(
#     "/home/tangyinzhou/inter_crime_prediction/data/CHI/crime_1.csv"
# )
# print(1)
# new_data = pd.DataFrame(
#     {
#         "time": data["time"],
#         "category": data["category"],
#         "lat": data["lat"],
#         "lon": data["lon"],
#     }
# )
# print(2)
# # new_data = new_data.dropna()
# # new_data = new_data[~(new_data['category'] == 'OTHER OFFENSE')]
# new_data.to_csv(
#     "/home/tangyinzhou/inter_crime_prediction/data/CHI/crime_1.csv", index=False
# )
# exit(0)
# data = pd.read_csv("/home/tangyinzhou/inter_crime_prediction/data/CHI/crime_2.csv")
# data = data.dropna()
# data.to_csv("/home/tangyinzhou/inter_crime_prediction/data/CHI/crime_3.csv")

import pandas as pd
from tqdm import *

data = pd.read_csv("/home/tangyinzhou/inter_crime_prediction/data/CHI/crime_2.csv")
for year in trange(2001, 2021):
    year = str(year)
    year_data = data[data['time'].str.slice(start=0, stop=4) == year]
    year_data.to_csv('/home/tangyinzhou/inter_crime_prediction/data/CHI/crime_{0}.csv'.format(year), index = False)