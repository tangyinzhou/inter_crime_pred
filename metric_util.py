import numpy as np

def cal_rmse(pred:np.ndarray,label:np.ndarray):
    assert pred.shape == label.shape, "Predictions and targets must have the same shape"
    
    # 计算误差的平方
    squared_errors = (pred - label) ** 2
    
    # 计算平均误差的平方（MSE）
    mean_squared_error = squared_errors.mean()
    
    # 计算RMSE
    rmse = mean_squared_error.sqrt()
    return rmse