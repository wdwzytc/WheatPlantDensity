import numpy as np
import warnings
from ..core.preprocess import adjust_t_s_and_n_leaves_s

def lut_invert(
        lut: np.ndarray,
        t_s: list,
        n_leaves_s: list,
        t_s_used_for_lut_invert=None,
        n_best_from_lut=20,
        t_interval=10,
        t_s_full=None,
) -> dict:
    """
    LUT-invert.
    It is based on the observed of leaf growth dynamic,
    and a simulated LUT.



    :param lut:
    LUT.
    It can be generated with code:
        lut2compact(generate_lut())

    :param n_leaves_s:
    A list of number of leaves of observations.
    Each 'n_leaves' needs to correspond to each thermal time in t_s

    :param t_s_used_for_lut_invert:
    If this parameter is not None, LUT-invert will only consider the
    thermal time values inside this list 't_s_used_for_lut_invert',
    and ignore those out of this list in the inversion.

    :param n_best_from_lut:
    The number of best data chosen from the LUT.
    The mean or median of these chosen data are the results of LUT-invert.

    :param t_interval:
    Describe the t_interval in lut_compact.
    Currently, the t_interval should be 10℃·d,

    :param t_s:
    A list of thermal-times of observations.
    t_s is a subset of t_s_full.
    At most, t_s = np.arange(0, 780, 10).

    :param: t_s_used_for_lut_invert
    specify the range of t_s used in LUT-invert.
    the dynamic data out of this t_s range
    will not be considered in LUT-invert.
    """

    len_t_s_full = lut.shape[1] - 5
    if t_s_full is None:
        raise ValueError('set t_s_full')

    # adjust t_s and check:
    t_s, n_leaves_s = adjust_t_s_and_n_leaves_s(t_s, n_leaves_s, t_s_full)

    # check len
    if not len(t_s) == len(n_leaves_s):
        raise ValueError("length of t_s and n_leaves_s should be the same")

    if t_s_used_for_lut_invert is None:
        # Default: use the full range for LUT-invert, skip check
        t_s_used_for_lut_invert = t_s_full
    else:
        # Check if all t_s_used_for_lut_invert are in t_s_full
        if not np.all(np.isin(t_s_used_for_lut_invert, t_s_full)):
            raise ValueError(
                "t_s_range_for_lut_invert should be within t_s_full")

    # project values from t_s to t_s_full.
    #   fill np.nan in places that are empty
    try:
        n_leaves_s_full = np.full((len_t_s_full), fill_value=np.nan)
        n_leaves_s_full[np.in1d(t_s_full, t_s)] = n_leaves_s
    except IndexError:
        dbg = True

    # set those values out of t_s_range_for_lut_invert as np.nan
    n_leaves_s_full[~np.in1d(t_s_full, t_s_used_for_lut_invert)] = np.nan
    if np.all(np.isnan(n_leaves_s_full)):
        warnings.warn("No value available for estimation."
                      "Maybe t_s_used_for_lut_invert and t_s do not overlap.")
        return {
            'best_n_params': np.full((n_best_from_lut, 5), fill_value=np.nan),
            'best_n_params_mean': np.full(5, fill_value=np.nan),
            'best_n_params_median': np.full(5, fill_value=np.nan),
            'best_n_dynamics': np.full((n_best_from_lut, len_t_s_full), fill_value=np.nan),
            't_s_full': t_s_full,
            'best_n_mse_of_dynamic': np.full(n_best_from_lut, fill_value=np.nan),
        }

    # tile the dynamic to the size of LUT
    n_leaves_s_tile = np.tile(n_leaves_s_full, (lut.shape[0], 1))

    # calculate MSE and ignore nan
    mse = np.nanmean((lut[:, 5:] - n_leaves_s_tile) ** 2, axis=1)

    # find the best n results
    best_n_ind = np.argsort(mse)[0:n_best_from_lut]
    best_n_mse_of_dynamic = mse[best_n_ind]

    # calculate the mean and median, for a stable result
    best_n_params = lut[best_n_ind, 0:5]
    best_n_params_mean = np.mean(best_n_params, axis=0)
    best_n_params_median = np.median(best_n_params, axis=0)
    best_n_dynamics = lut[best_n_ind, 5:]
    return {
        'best_n_params': best_n_params,
        'best_n_params_mean': best_n_params_mean,
        'best_n_params_median': best_n_params_median,
        'best_n_dynamics': best_n_dynamics,
        't_s_full': t_s_full,
        'best_n_mse_of_dynamic': best_n_mse_of_dynamic,
    }
