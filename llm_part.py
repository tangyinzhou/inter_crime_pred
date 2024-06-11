import os
import json
from openai import OpenAI
import time
import httpx
import numpy as np
import pandas as pd
from util import process_ser

def get_agent_action_simple(session, model_name="gpt-3.5",temperature=1.0, max_tokens=200):
    # 模型API设置
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        model_name_map = {
            "gpt-3.5": "gpt-3.5-turbo-0125",
            "gpt-4": "gpt-4-0125-preview"
        }
        model_name = model_name_map[model_name]
        client = OpenAI(
            http_client=httpx.Client(proxy="http://127.0.0.1:10190"),
            api_key='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'
        )
        
    elif "meta-llama" in model_name or "mistralai" in model_name:
        client = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key="DMtmh2gv9CgObKohHmb7iVxgXSonNAj6",
            http_client=httpx.Client(proxies="http://127.0.0.1:10190"),
        )
    elif "deepseek-chat" in model_name:
        client = OpenAI(
            api_key="sk-97d864b57efb45cd8c430603dea4580a",
            base_url="https://api.deepseek.com/v1"
        )
    else:
        model_name, port = model_name.split(":")
        if args.infer_server == "LLM2-vllm":
            client = OpenAI(
                base_url="http://101.6.69.35:{}/v1".format(port),
                api_key="token-fiblab-20240425"
            )
        elif args.infer_server == "DL4-vllm":
            client = OpenAI(
                base_url="http://101.6.69.111:{}/v1".format(port),
                api_key="token-fiblab-20240425"
            )

    MAX_RETRIES = 1
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
            
def get_prompt_area(area_data, llm_prompt_weight, gnn_pred, index, city, year, city_dict):
    #feature_per[index], llm_prompt_weight, gnn_pred_per[index]
    community_name = pd.read_csv("/data/duyuwei/data/Mobile_data_analysis/inter_crime_pred/data/CHI/index2name.csv")
    community_dict = {}
    for index, row in community_name.iterrows():
        key = int(row['index'])
        name = row['name']
        if key not in community_dict:
            community_dict[key] = name
    crime_data = area_data[0].T  #(cat,his_len)
    crime_data_list = list(crime_data)
    crime_data_proc = [process_ser(crime_item) for crime_item in crime_data_list]
    weight_dict = {
        0: "not at all",
        0.1: "hardly",
        0.2: "rarely",
        0.3: "seldom",
        0.4: "slightly",
        0.5: "moderately",
        0.6: "fairly",
        0.7: "quite",
        0.8: "very",
        0.9: "extremely",
        1: "completely"
    }
    degree = weight_dict[llm_prompt_weight]
    if city == "CHI":
        crime_list = ['THEFT', 'DECEPTIVE PRACTICE', 'CRIMINAL DAMAGE', 'CRIMINAL SEXUAL ASSAULT', 'BURGLARY', 'HOMICIDE',
                    'ASSAULT', 'ROBBERY', 'OFFENSE INVOLVING CHILDREN', 'BATTERY', 'CRIM SEXUAL ASSAULT', 
                    'SEX OFFENSE', 'MOTOR VEHICLE THEFT', 'CRIMINAL TRESPASS', 'NARCOTICS', 'PROSTITUTION', 'WEAPONS VIOLATION', 
                    'ARSON', 'KIDNAPPING', 'PUBLIC PEACE VIOLATION', 'GAMBLING', 'LIQUOR LAW VIOLATION', 'RITUALISM', 'INTIMIDATION', 
                    'INTERFERENCE WITH PUBLIC OFFICER', 'OTHER NARCOTIC VIOLATION', 'STALKING', 'PUBLIC INDECENCY', 'OBSCENITY', 'DOMESTIC VIOLENCE']
    else:
        crime_list = []
    community = community_dict[index]
    list_length = len(crime_list)
    gnn_pred_type_his = str({key: value for key, value in zip(crime_list, crime_data_proc)})
    gnn_pred_type = str({key: value for key, value in zip(crime_list, gnn_pred)})
    prompt_head = "The crime counts of each kind of crime in past 11 months of {community} are as follows:\n{gnn_pred_type_his}"
    prompt_tail = """I will give you the TGNN prediction of crime count of each kind of crime in next month of {community}.
    Please consider the TGNN prediction {degree}, and give the prediction of crime count of each kind of crime
    in next month of {community} without producing any additional text. Do not say anything like 'The predicted crime count for {community} for the next month is:', 
    just return a list of length {list_length},each value represents for a prediction of count for a certain crime type.The order of the crime types is the same as the TGNN prediction output:[the prediction count for {crime_list[0]},the prediction count for {crime_list[1]},...,the prediction count for {crime_list[-1]}].
    \n The prediction of TGNN:{gnn_pred_type}\nYour prediction:\n
    """

    total_population = city_dict[city][year][community]['total_population']
    Average_family_income = city_dict[city][year][community]['income']
    female_ratio = city_dict[city][year][community]['female_ratio']
    white_ratio = city_dict[city][year][community]['white_ratio']
    black_ratio = city_dict[city][year][community]['Black or African American ratio']
    asian_ratio = city_dict[city][year][community]['Asian ratio']
    Latino_ratio = city_dict[city][year][community]['Hispanic or Latino Origin ratio']
    poverty_ratio = city_dict[city][year][community]['poverty ratio']
    young_ratio = city_dict[city][year][community]['population below 35 years old ratio']
    unemployment_ratio = city_dict[city][year][community]['unemploymen ratio']
    enrollment_ratio = city_dict[city][year][community]['school enrollment ratio']

    prompt = """
    Now you arrive at {community}.Features of this community are as follows:
    Total population:{total_population}\nAverage family income last year:{Average_family_income}\nFemale ratio:{female_ratio}
    White ratio:{white_ratio}\nBlack or African American ratio:{black_ratio}\nAsian ratio:{asian_ratio}\nHispanic or Latino origin ratio:{Latino_ratio}\n
    Poverty ratio:{poverty_ratio}\nPopulation below 35 years old ratio:{young_ratio}\nUnemployment ratio:{unemployment_ratio}\nSchool enrollment ratio:{enrollment_ratio}\n
    """
    return prompt_head + prompt + prompt_tail

def extract_output(output):
    ###TODO:依据测试输出填写处理输出的函数
    return output
##feature_per（地区数，batch_size,历史11个月，犯罪类别）
        #gnn_pred(地区，batch_size,GNN预测未来的1个月，犯罪类别)
        
def llm_area_pred_func(area_data: np.array, llm_prompt_weight, gnn_pred, index, city, year, city_dict):  #model为gpt-3.5-turbo, temperature=0, max_tokens=200
    prompt = get_prompt_area(area_data, llm_prompt_weight, gnn_pred, index, city, year, city_dict)  
    #(area_data, llm_prompt_weight, gnn_pred, index, city, year, city_dict)
    output = get_agent_action_simple(
    session=  [{
                        "role": "system",
                        "content": "you are a helpful assistant that performs crime prediction of each community you arrive.The user will provide the description of the community, crime counts of each kind of crime in past 11 months,\
                            TGNN prediction of crime counts for each kind of crime in next month.You will predict the crime counts for each kind of crime in next month.You should consider the "
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
    model_name="gpt-3.5", temperature=0, max_tokens=200
    )
    result = extract_output(output)
    return result
if __name__ == "__main__":
    print("开始进行模型评估")
    area_data = np.random.randint(0, 41, (1, 30, 11))
    llm_prompt_weight = 0.5
    gnn_pred = np.random.randint(0, 41, (1, 30))
    index = 5
    city = "CHI"
    year = "2016" 
    with open("/data/duyuwei/data/Mobile_data_analysis/inter_crime_pred/data/community_feature_dict.json", "r") as f:
        city_dict = json.load(f)
    llm_area_pred_func(area_data, llm_prompt_weight, gnn_pred, index, city, year, city_dict)
