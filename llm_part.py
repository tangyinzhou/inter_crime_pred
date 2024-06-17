import os
import json
from openai import OpenAI
import time
import httpx
import numpy as np
import pandas as pd
from util import *


os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"


def get_agent_action_simple(
    session,
    model_name="gpt-3.5",
    temperature=1.0,
    max_tokens=200,
    OPENAI_KEY="",
):

    # 模型API设置
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        model_name_map = {
            "gpt-3.5": "gpt-3.5-turbo-0125",
            "gpt-4": "gpt-4-0125-preview",
        }
        model_name = model_name_map[model_name]
        # client = OpenAI(
        #     http_client=httpx.Client(proxy="http://127.0.0.1:10190"), api_key=OPENAI_KEY
        # )
        client = OpenAI(api_key=OPENAI_KEY)
    MAX_RETRIES = 5

    WAIT_TIME = 1
    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=session,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if i < MAX_RETRIES - 1:
                time.sleep(WAIT_TIME)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "OpenAI API Error."


def get_prompt_area(
    use_dataset, area_data, llm_prompt_weight, gnn_pred, index, city, year, city_dict
):
    # feature_per[index], llm_prompt_weight, gnn_pred_per[index]
    community_name = pd.read_csv(
        "./data/{0}/index2name.csv".format(use_dataset)
    )
    community_dict = {}
    for index, row in community_name.iterrows():
        key = int(row["index"])
        name = row["name"]
        if key not in community_dict:
            community_dict[key] = name
    crime_data = area_data[0].T  # (cat,his_len)
    crime_data_list = list(crime_data)
    crime_data_proc = [process_ser(crime_item) for crime_item in crime_data_list]
    weight_dict = {
        0: "Please disregard the T-GCN prediction entirely",
        0.1: "Please barely take into account the T-GCN prediction",
        0.2: "Please minimally consider the T-GCN prediction",
        0.3: "Please slightly consider the T-GCN prediction",
        0.4: "Please give some consideration to the T-GCN prediction",
        0.5: "Please moderately consider the T-GCN prediction",
        0.6: "Please largely consider the T-GCN prediction",
        0.7: "Please significantly consider the T-GCN prediction",
        0.8: "Please highly consider the T-GCN prediction",
        0.9: "Please fully consider the T-GCN prediction",
    }
    degree = weight_dict[llm_prompt_weight]
    if city == "CHI":
        crime_list = [
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
    if city == "NYC":
        crime_list = [
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
    if city == "SF":
        crime_list = [
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
    community = community_dict[index]
    list_length = len(crime_list)
    gnn_pred_type_his = str(
        {key: value for key, value in zip(crime_list, crime_data_proc)}
    )
    gnn_pred_type = str({key: value for key, value in zip(crime_list, gnn_pred)})
    prompt_head = f"The crime counts of each kind of crime in past 11 months of {community} are as follows:\n{gnn_pred_type_his}"
    prompt_tail = f"""I will give you the T-GCN prediction of crime count of each kind of crime in next month of {community}.
    {degree}, consider the crime counts of each kind of crime in past 11 months of {community} and features of {community}, and give the prediction of crime count of each kind of crime
    in next month of {community} without producing any additional text. Do not say anything like 'The predicted crime count for {community} for the next month is:', 
    just return a list of length {list_length},each value represents for a prediction of count for a certain crime type.The order of the crime types is the same as the T-GCN prediction output:[the prediction count for {crime_list[0]},the prediction count for {crime_list[1]},...,the prediction count for {crime_list[-1]}].
    \n The prediction of T-GCN:{gnn_pred_type}\nYour prediction:\n
    """

    total_population = city_dict[city][year][community]["total_population"]
    Average_family_income = city_dict[city][year][community]["income"]
    female_ratio = city_dict[city][year][community]["female_ratio"]
    white_ratio = city_dict[city][year][community]["white_ratio"]
    black_ratio = city_dict[city][year][community]["Black or African American ratio"]
    asian_ratio = city_dict[city][year][community]["Asian ratio"]
    Latino_ratio = city_dict[city][year][community]["Hispanic or Latino Origin ratio"]
    poverty_ratio = city_dict[city][year][community]["poverty ratio"]
    young_ratio = city_dict[city][year][community][
        "population below 35 years old ratio"
    ]
    unemployment_ratio = city_dict[city][year][community]["unemploymen ratio"]
    enrollment_ratio = city_dict[city][year][community]["school enrollment ratio"]

    prompt = f"""Now you arrive at {community}.Features of this community are as follows:\n
    Total population:{total_population}\nAverage family income last year:{Average_family_income}\nFemale ratio:{female_ratio}
    White ratio:{white_ratio}\nBlack or African American ratio:{black_ratio}\nAsian ratio:{asian_ratio}\nHispanic or Latino origin ratio:{Latino_ratio}\n
    Poverty ratio:{poverty_ratio}\nPopulation below 35 years old ratio:{young_ratio}\nUnemployment ratio:{unemployment_ratio}\nSchool enrollment ratio:{enrollment_ratio}\n
    """
    return prompt_head + prompt + prompt_tail


def extract_output(output):
    ###TODO:依据测试输出填写处理输出的函数
    output = eval(output)
    return output


##feature_per（地区数，batch_size,历史11个月，犯罪类别）
# gnn_pred(地区，batch_size,GNN预测未来的1个月，犯罪类别)


def llm_area_pred_func(
    use_dataset,
    area_data: np.array,
    llm_prompt_weight,
    gnn_pred,
    index,
    city,
    year,
    city_dict,
    openai_key,
):  # model为gpt-3.5-turbo, temperature=0, max_tokens=200
    prompt = get_prompt_area(
        use_dataset,
        area_data,
        llm_prompt_weight,
        gnn_pred,
        index,
        city,
        year,
        city_dict,
    )
    # (area_data, llm_prompt_weight, gnn_pred, index, city, year, city_dict)
    output = get_agent_action_simple(
        session=[
            {
                "role": "system",
                "content": "you are a helpful assistant that performs crime prediction of each community you arrive.The user will provide the description of the community, crime counts of each kind of crime in past 11 months,\
                            T-GCN prediction of crime counts for each kind of crime in next month.You will predict the crime counts for each kind of crime in next month.You should consider the ",
            },
            {"role": "user", "content": prompt},
        ],
        model_name="gpt-3.5",
        temperature=0,
        max_tokens=200,
        OPENAI_KEY=openai_key,
    )
    addtoken(len(prompt.split()))
    addtoken(len(output.split()))
    result = extract_output(output)
    return result
