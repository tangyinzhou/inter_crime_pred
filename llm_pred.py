from data_loader import *
import numpy as np
from llm_part import llm_area_pred_func
from tqdm import *

##feature_per（地区数，batch_size,历史11个月，犯罪类别）
# gnn_pred(地区，batch_size,GNN预测未来的1个月，犯罪类别)


def llm_pred_func(
    use_dataset: str,
    dataset,
    llm_prompt_weight: float,
    adj,
    city: str,
    year,
    city_dict,
    openai_key,
):
    dataloader = DataLoader(dataset, batch_size=1)
    llm_pred = {}
    dataset_data = {}
    with open(
        "./data/{0}/normalize_param.json".format(use_dataset),
        "r",
    ) as f:
        normalizer = json.load(f)
    for features, labels, gnn_pred, idx in tqdm(dataloader):
        batch_preds = []
        feature_per = features.permute(
            2, 0, 1, 3
        ).numpy()  # (num_region,batch_size,his_len,cat)
        feature_per = feature_per * normalizer["std"] + normalizer["mean"]
        gnn_pred_per = gnn_pred.permute(2, 0, 1, 3).numpy()
        gnn_pred_per = gnn_pred_per * normalizer["std"] + normalizer["mean"]
        for index in trange(feature_per.shape[0]):
            llm_area_pred = llm_area_pred_func(
                use_dataset,
                feature_per[index],
                llm_prompt_weight,
                gnn_pred_per[index],
                index,
                city,
                year,
                city_dict,
                openai_key,
            )
            batch_preds.append(
                (np.array(llm_area_pred) - normalizer["mean"]) / normalizer["std"]
            )
        batch_preds = np.array(batch_preds)
        dataset_data[int(idx)] = np.concatenate(
            (features.squeeze(0).numpy(), batch_preds[np.newaxis, :]), axis=0
        )

        llm_pred[int(idx)] = batch_preds[np.newaxis, :]
    llm_pred = np.array(llm_pred)
    dataset = GNNDataset(data=np.array(dataset_data), llm_pred=llm_pred)
    return dataset
