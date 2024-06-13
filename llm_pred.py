from data_loader import *
import numpy as np
from llm_part import llm_area_pred_func

##feature_per（地区数，batch_size,历史11个月，犯罪类别）
        #gnn_pred(地区，batch_size,GNN预测未来的1个月，犯罪类别)

def llm_pred_func(dataset, llm_prompt_weight, adj, city, year, city_dict, openai_key):
    dataloader = DataLoader(dataset, batch_size=1)
    llm_pred = []
    dataset_data = []
    for features, labels, gnn_pred in dataloader:
        batch_preds = []
        feature_per = features.permute(
            2, 0, 1, 3
        ).numpy()  # (num_region,batch_size,his_len,cat)
        gnn_pred_per = gnn_pred.permute(
            2, 0, 1, 3
        ).numpy()
        for index in range(feature_per.shape[0]):
            llm_area_pred = llm_area_pred_func(
                feature_per[index], llm_prompt_weight, gnn_pred_per[index], index, city, year, city_dict, openai_key
            )
            #llm_area_pred_func(area_data: np.array, llm_prompt_weight, gnn_pred, index, city, year, city_dict)
            batch_preds.append(llm_area_pred)
        llm_pred.append(batch_preds)
        dataset_data.append(
            np.concatenate((features.squeeze(0).numpy(), labels.numpy()), axis=0)
        )
    llm_pred = np.array(llm_pred)
    llm_pred = np.expand_dims(llm_pred, axis=1)
    llm_pred = np.expand_dims(llm_pred, axis=3)
    dataset = GNNDataset(data=dataset_data, llm_pred=llm_pred)
    return dataset
