from llm_part import llm_area_pred_func
import json
import numpy as np
if __name__ == "__main__":
    print("开始进行模型评估")
    area_data = np.random.randint(0, 41, (1, 30, 11))
    llm_prompt_weight = 0.5
    gnn_pred = np.random.randint(0, 41, (1, 30))
    index = 5
    city = "CHI"
    year = "2020"
    with open("./data/community_feature_dict.json", "r") as f:
        city_dict = json.load(f)
    result = llm_area_pred_func(area_data, llm_prompt_weight, gnn_pred, index, city, year, city_dict)
    print(result)
