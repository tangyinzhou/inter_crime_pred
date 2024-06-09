from data_loader import *
import numpy as np


def llm_area_pred_func(area_data: np.array, llm_prompt_weight):
    ###TODO:完善对于某个地区的LLM犯罪预测
    return 0


def llm_pred_func(dataset, llm_prompt_weight, G):
    dataloader = DataLoader(dataset, batch_size=1)
    llm_pred = []
    dataset_data = []
    for features, labels, _ in dataloader:
        batch_preds = []
        feature_per = features.permute(2, 0, 1, 3).numpy()
        for area_data in feature_per:
            llm_area_pred = llm_area_pred_func(area_data, llm_prompt_weight)
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
