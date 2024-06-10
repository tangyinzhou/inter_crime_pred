from data_loader import *
import numpy as np


def llm_area_pred_func(area_data: np.array, llm_prompt_weight, gnn_pred):
    ###TODO:完善对于某个地区的LLM犯罪预测
    return 0


def llm_pred_func(dataset, llm_prompt_weight, adj):
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
                feature_per[index], llm_prompt_weight, gnn_pred_per[index]
            )
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
