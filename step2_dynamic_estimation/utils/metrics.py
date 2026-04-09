import numpy as np
import warnings
from sklearn.metrics import root_mean_squared_error

def relative_root_mean_square_error(v_true, v_pred):
    if np.mean(v_true) == 0:
        warnings.warn('mean(true) is 0. Setting rRMSE to np.inf.')
        return np.inf
    elif np.any(np.isnan(v_pred)) or np.any(np.isnan(v_true)):
        return np.nan
    else:
        return root_mean_squared_error(v_true, v_pred) / np.mean(v_true)