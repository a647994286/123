import numpy as np

def RSE(pred, true):
    mask = true > 0
    return np.sqrt(np.sum((pred[mask] - true[mask]) ** 2)) / np.sqrt(np.sum((true[mask] - true[mask].mean()) ** 2))

def CORR(pred, true):
    mask = true > 0
    pred_masked = pred[mask]
    true_masked = true[mask]
    u = ((true_masked - true_masked.mean(0)) * (pred_masked - pred_masked.mean(0))).sum(0)
    d = np.sqrt(((true_masked - true_masked.mean(0)) ** 2 * (pred_masked - pred_masked.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    mask = true > 0
    return np.mean(np.abs(pred[mask] - true[mask]))

def MSE(pred, true):
    mask = true > 0
    return np.mean((pred[mask] - true[mask]) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    mask = true > 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))

def MSPE(pred, true):
    mask = true > 0
    return np.mean(np.square((pred[mask] - true[mask]) / true[mask]))