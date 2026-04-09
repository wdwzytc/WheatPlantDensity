import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings

from scipy.stats import qmc
import matplotlib.colors as mcolors
import datetime
import numbers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr

if "initialize":
    df_dataset_index_xlsx = pd.read_excel(r"./dataset_index.xlsx", sheet_name='Data')


def relative_root_mean_square_error(v_true, v_pred):
    "rrmse = rmse / mean(true_value)"

    if str(v_true.dtype) != 'float64':
        dbg = True

    if np.mean(v_true) == 0:
        warnings.warn('mean(true) is 0. Setting rRMSE to np.inf.')
        r_rmse = np.inf
    elif np.any(np.isnan(v_pred)) | np.any(np.isnan(v_true)):
        r_rmse = np.nan
    else:
        r_rmse = root_mean_squared_error(v_true, v_pred) / np.mean(v_true)
    return r_rmse


def scatter_and_show_accuracy(
        ax, v_true, v_pred,
        r2_round_digits=2,
        rmse_round_digits=4,
        rrmse_round_digits=0,
        mae_round_digits=4,
        axis_range=None,
        show_r_sqr=True,
        show_rmse=True,
        show_rrmse=True,
        show_rrmse_in_percentile=True,
        show_mae=False,
        show_pearson_corrcoef=False,
        show_pearson_corrcoef_sqr=False,
        show_spearman_corrcoef=False,
        scatter_facecolors=None,
        scatter_edgecolors=None,
        scatter_marker=None,
        scatter_size=10,
        scatter_alpha=None,
        scatter_linewidths=None,
        text_x_relative_anchor=0.05,
        text_y_relative_anchor=0.8,
        grid=False,
):
    v_true = np.array(v_true)
    v_pred = np.array(v_pred)
    scatter_object = ax.scatter(
        v_true, v_pred, s=scatter_size,
        marker=scatter_marker,
        facecolors=scatter_facecolors,
        edgecolors=scatter_edgecolors,
        linewidths=scatter_linewidths,
        alpha=scatter_alpha, )

    if axis_range is None:
        axis_min = min(list(ax.get_xlim()) + list(ax.get_ylim()) + [0])
        axis_max = max(list(ax.get_xlim()) + list(ax.get_ylim()))
    else:
        axis_min, axis_max = axis_range
    # ax.set(xlim=(axis_min, axis_max), ylim=(axis_min, axis_max))
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal')

    text_x = (1 - text_x_relative_anchor) * axis_min + text_x_relative_anchor * axis_max
    text_y = (1 - text_y_relative_anchor) * axis_min + text_y_relative_anchor * axis_max
    text_content = ''
    if show_r_sqr:
        text_content += '\nR$^2$={n1:.{c1}f}'.format(
            n1=r2_score(v_true, v_pred),
            c1=r2_round_digits,
        )
    if show_pearson_corrcoef:
        text_content += '\nr={n5:.2f}'.format(
            n5=np.corrcoef(v_true.ravel(), v_pred.ravel())[1, 0]
        )
    if show_pearson_corrcoef_sqr:
        text_content += '\nr$^2$={n5:.2f}'.format(
            n5=np.corrcoef(v_true.ravel(), v_pred.ravel())[1, 0] ** 2
        )
    if show_spearman_corrcoef:
        text_content += '\nr$_s$={n6:.2f}'.format(
            n6=spearmanr(v_true, v_pred).statistic
        )
    if show_rmse:
        text_content += '\nRMSE={n2:.{c2}f}'.format(
            n2=root_mean_squared_error(v_true, v_pred),
            c2=rmse_round_digits,
        )
    if show_rrmse and (not show_rrmse_in_percentile):
        text_content += '\nrRMSE={n3:.{c3}f}'.format(
            n3=relative_root_mean_square_error(v_true, v_pred),
            c3=rrmse_round_digits
        )
    if show_rrmse and show_rrmse_in_percentile:
        text_content += '\nrRMSE={n3:.{c3}%}'.format(
            n3=relative_root_mean_square_error(v_true, v_pred),
            c3=rrmse_round_digits
        )
    if show_mae:
        text_content += '\nMAE={n4:.{c4}f}'.format(
            n4=mean_absolute_error(v_true, v_pred),
            c4=mae_round_digits,
        )

    text_content = text_content.strip()  # remove first blank line
    text_object = ax.text(
        text_x, text_y,
        text_content,
        fontsize=12)

    ax.plot([axis_min, axis_max], [axis_min, axis_max], color='black', linestyle='--', linewidth=1)
    ax.plot([0, 0], [axis_min, axis_max], color='black', linestyle='-', linewidth=1)
    ax.plot([axis_min, axis_max], [0, 0], color='black', linestyle='-', linewidth=1)

    if grid:
        ax.grid(True)

    return {
        'scatter_object': scatter_object,
        'text_object': text_object,
    }


def str_is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_dataset_for_dynamic(df):
    """
    get target data from dataset_index.xlsx
    Data of Avignon 2023-2024 with full dynamics
        (remove some scattered images for image annotations)
    Data of Fourques 2022-2023 with full dynamics
    Data of Salin-de-Giraud 2022-2023 with full dynamics

    """

    # rows to keep
    flag1 = [site in ['Avignon', 'Fourques', 'Salin-de-Giraud'] for site in df['Site']]
    date_lst = [datetime.datetime(2023, 11, 29, 0, 0),
                datetime.datetime(2022, 10, 28, 0, 0),
                datetime.datetime(2022, 11, 18, 0, 0)]
    flag2 = [date in date_lst for date in df['Date of sowing']]
    flag3 = (df['CVAT task id'] != 60)  # small avignon dataset, cropped
    flag4 = df['view zenith angle (°)'] == 45
    flag5 = [str_is_float(i) for i in df['note14 Air temperature_Thermal time #4']]
    df_c = df[pd.Series(flag1)
              & pd.Series(flag2)
              & pd.Series(flag3)
              & pd.Series(flag4)
              & pd.Series(flag5)]

    pass  # keep all columns

    return df_c


def adjust_t_s_and_n_leaves_s(t_s, n_leaves_s, t_s_full, min_max_tolerance=5):
    """
    Problem to solve:
        t_s may contain thermal time values with decimal point,
        this is not in the leaf-tip-dynamic-LUT
        and cannot apply inversion-estimation
    Solution (this function):
        take values of n_leaves_s, move them from "t_s" to "t_s_full"
    Note:
        when more than 1 n_leaves data come want to fill in the same place,
            calculate their average and fill.
        when t_s is out of t_s_full, put "np.nan".
        when min(t_s) < min(t_s_full), check min_max_tolerance
        when max(t_s) > max(t_s_full), check min_max_tolerance

    try doctest module
    >>> t_s = [225.91, 246.17000000000002, 273.84000000000003, 287.65999999999997, 299.25, 309.07, 312.06, 330.85, 346.61]
    >>> n_leaves_s = [234.2857142857143, 228.57142857142858, 422.8571428571429, 474.28571428571433, 485.7142857142857, 508.5714285714286, 508.5714285714286, 680.0, 708.5714285714287]
    >>> t_s_full = np.arange(0, 780, 10)
    >>> t_s, n_leaves_s = adjust_t_s_and_n_leaves_s(t_s,n_leaves_s,t_s_full)
    >>> print(t_s)
    [230, 250, 270, 290, 300, 310, 330, 350]
    >>> print(np.array(n_leaves_s).astype(int).tolist())
    [234, 228, 422, 474, 485, 508, 680, 708]
    """

    # remove invalid items in t_s and n_leaves_s,
    #   not numbers:
    flags = [isinstance(i, numbers.Number) and isinstance(j, numbers.Number) for i, j in zip(t_s, n_leaves_s)]
    t_s = np.array(t_s)[flags].astype(float).tolist()
    n_leaves_s = np.array(n_leaves_s)[flags].astype(float).tolist()
    #   nans:
    flags = ~np.isnan(t_s) & ~np.isnan(n_leaves_s)
    t_s = np.array(t_s)[flags].astype(float).tolist()
    n_leaves_s = np.array(n_leaves_s)[flags].astype(float).tolist()

    # extract the closest pairs of t_s vs t_s_full, and save in t_s_round
    t_s_ = np.tile(np.array(t_s), (len(t_s_full), 1))
    t_s_full_ = np.tile(np.array(t_s_full)[:, np.newaxis], len(t_s))
    delta = (t_s_ - t_s_full_) ** 2
    ind_min = np.argmin(delta, axis=0)  # if two values are the same smallest, np.argmin() choses the first.
    t_s_round = np.array(t_s_full)[ind_min].astype(int).tolist()

    # if some t_s is out of the range of t_s_full, raise ValueError
    dbg = True
    out_flag = ((np.array(t_s) < (np.min(t_s_full) - min_max_tolerance))
                | (np.array(t_s) > (np.max(t_s_full) + min_max_tolerance)))
    if (out_flag).sum() > 0:
        raise ValueError('t_s range out of t_s_full range')

    d = {}
    t_s_new = []
    n_leaves_s_new = []
    for t, n in zip(t_s_round, n_leaves_s):
        if t in d:
            d[t].append(n)
        else:
            d[t] = [n]
    for t in d:
        t_s_new.append(t)
        n_leaves_s_new.append(np.mean(d[t]))
    assert (set(t_s_new).issubset(set(t_s_full)))  # double check, may remove.
    return t_s_new, n_leaves_s_new


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


def get_n_leaves_dynamic_of_plot(
        d=300,
        mean_t_e=160,
        std_t_e=10,
        mean_t_p=100,
        std_t_p=None,
        plot_size=1.0,
        t_interval=10,
        t_s_full=None,
        random_seed=42,
        leaf_tip_observation_error_rate=None,
        repeat_observation=None,
        return_dynamics_of_each_plant_without_error=False,
        p_no_tiller=0.12,
        p_start_tiller_0=0.01,
        p_start_tiller_1=0.71,
        p_start_tiller_2=0.16,
):
    r"""
    ------------update on 2023-12-8 -----------------
    Added T0 in consideration.
    Todo: reorganize the parameters:
        p_no_tiller,
        p_t0_in_tiller,
        p_t1_in_tiller,
        p_t2_in_tiller.
        The sum of them are 1.0.

    ------------update on 2023-9-17 -----------------
    Add a parameter to enable different tiller-dynamics for the plot.
    Instead of a fixed probability to start tillering from
        [T1: 72%; T2: 16%; No tiller: 12%],
        it is allowed to have earlier tillers,
        later tillers or no tiller at all.
    Later tillering (Start from T2 / have no tiller) were observed in
        Tiancheng's flowerpot experiment.
    This is parameterized by two probabilities:
        1. proportion of no-tiller (default value is 12%)
            p_no_tiller
        2. Among all the plants with tiller, the proportion to start
            with T1 (default value is 72%/(72%+16%)=81.8%)
            p_t1
        and after that, the other plants start with T2.
    To simplify, no plants start from T0, T3, T4, ...

    ------------bug on 2023-9-8 -----------------
    stem_i_exist was not used.

    ------------update on 2023-9-8 -----------------
    The tillers emerge when the main-stem has n-leaves:
        T1: 2.46
        T2: 3.24
        T3: 4.25
        T4: 5.50
        T5: 6.91
    See in Table-6, https://doi.org/10.1016/j.fcr.2018.01.010

    To simplify, the main-stem and the tillers grow in the same speed.

    To simplify, the probability to start tillering with
        T1: 72%,
        T2: 16%.
        No tiller: 12%

    The density and <Leaf stage to stop tiller generation> have a relationship:
        LS(t_stop) = 17.17 * (density ** -0.204)
        - when 50 < density < 500 plants/(m*m)
        - see cultivar Soissons, Fig. 9. a), https://doi.org/10.1016/j.fcr.2018.01.010

    Deleted the plotting function and parameters for plotting:
        (show_in_figure_n,plot_color,scatter_size,scatter_label,xlim,ylim)
    """

    np.random.seed(random_seed)
    n_plants = int(d * plot_size)
    if (std_t_p is not None) and (std_t_p != 0):
        warnings.warn('Setting std_t_p is useless. It is not used.')

    # # # # #
    # 0. How does this function calculate the dynamics of plants?
    # These data will be generated and saved in an array:
    #   - The number of leaves for each plant (n_plants)
    #   - The number of leaves for each stem (MS, T0, T1, T2,..., T5)
    #   - The number of leaves for each thermal_time
    #   So the array will have 3 dimensions (n_plants, n_stems(=7), n_thermal_times)
    # Judge if a stem exists based on random value. (variable: "stem_i_exists")
    # Calculate the number of leaves on a plant (variable: "ls")
    # Apply "stem_i_exists" to "ls", and calculate the sum.

    # # # # #
    # 1. Apply the cessation of tillering
    # [l]eaf [s]tage of main stem at [t]illering [c]essation
    # 产生分蘖的截止时间
    assert (50 <= d <= 500), '50 <= density <= 500'
    ls_tc = 17.17 * (d ** -0.204)
    # The tillers emerge when the main-stem has n-leaves:
    #     MS: 0
    #     T0: 1.9 (≈)
    #     T1: 2.46
    #     T2: 3.24
    #     T3: 4.25
    #     T4: 5.50
    #     T5: 6.91
    # shape = (n_plants, (MS, T1, T2, T3, T4, T5))
    stem_i_ls = np.repeat(
        np.array((0, 1.9, 2.46, 3.24, 4.25, 5.50, 6.91))[np.newaxis, :],
        n_plants,
        axis=0
    )
    stem_i_exists = stem_i_ls < ls_tc
    # Note: Since ls_tc > 0, that MS always exists, which corresponds with the real world.

    # # # # #
    # 2. Apply the start of tillering
    # tiller_start_seed: determines which category this plant belongs to.
    #   2.1. 12% no tiller (0 <= seed < 12)
    #   2.3. 1% start from tiller T0 (12 <= seed < 13)
    #   2.3. 71% start from tiller T1 (13 <= seed < 88)
    #   2.4. 16% start from tiller T2 (88 <= seed < 100)
    # Check these four parameters:
    #   p_no_tiller,
    #   p_t0_in_tiller,
    #   p_t1_in_tiller,
    #   p_t2_in_tiller.
    if (p_no_tiller + p_start_tiller_0 + p_start_tiller_1 + p_start_tiller_2) == 1.0:
        pass
    else:
        warnings.warn("\n"
                      "\tSum of them is not 1.0, so normalizing: \n"
                      "\tp_no_tiller, p_start_tiller_0, p_start_tiller_1, p_start_tiller_2.\n")
        assert (0 <= p_no_tiller <= 1)
        assert (0 <= p_start_tiller_0 <= 1)
        assert (0 <= p_start_tiller_1 <= 1)
        assert (0 <= p_start_tiller_2 <= 1)
        arr = np.array((p_no_tiller, p_start_tiller_0, p_start_tiller_1, p_start_tiller_2))
        arr = arr / arr.sum()
        (p_no_tiller, p_start_tiller_0, p_start_tiller_1, p_start_tiller_2) = arr.tolist()

    # tiller_start_seed shape: (n_plants,)
    tiller_start_seed = np.random.rand(n_plants)
    # no-tiller plants
    stem_i_exists[(0 <= tiller_start_seed)
                  & (tiller_start_seed < p_no_tiller), 1:] = False
    # plants start tillering from t0 -> all exist
    # stem_i_exists[(p_no_tiller <= tiller_start_seed)
    #               & (tiller_start_seed < (p_no_tiller + p_start_tiller_0)), :]  # do nothing
    # plants start tillering from t1 -> t0 does not exist
    stem_i_exists[((p_no_tiller + p_start_tiller_0) <= tiller_start_seed)
                  & (tiller_start_seed < (p_no_tiller + p_start_tiller_0 + p_start_tiller_1)), 1] = False
    # plants start tillering from t2 -> t0 and t1 do not exist
    stem_i_exists[((p_no_tiller + p_start_tiller_0 + p_start_tiller_1) <= tiller_start_seed) & (
            tiller_start_seed < (p_no_tiller + p_start_tiller_0 + p_start_tiller_1 + p_start_tiller_2)), 1:3] = False

    # # # # #
    # 3. thermal_time_after_emergence.
    # It depends on each plant,
    #   since each plant has a t_e
    #   (thermal_time_of_emergence).
    #
    # t_e_plants, shape: (n_plants,)
    t_e_plants = mean_t_e + std_t_e * np.random.randn(n_plants)

    # thermal_time values for one dynamic
    if t_s_full is None:
        t_s = np.arange(0, 780, t_interval)
    else:
        t_s = np.array(t_s_full)

    # t_as: thermal_time after sowing
    # t_as shape:
    #   (n_plants,
    #   7 (for MS, T0, T1, T2, T3, T4, T5),
    #   thermal_time_s)
    t_as = np.broadcast_to(
        t_s[np.newaxis, np.newaxis, :],
        (n_plants, 7, len(t_s))
    )

    # t_ae: thermal_time after emergence. For each Plant/Tiller/Thermal_time.
    t_ae = t_as - np.broadcast_to(
        t_e_plants[:, np.newaxis, np.newaxis],
        (n_plants, 7, len(t_s))
    )

    # # # # #
    # 4. Apply the leaf stage,
    # shape = (n_plants, (MS, T0, T1, T2, T3, T4, T5), thermal_time_s)
    ls_base = np.broadcast_to(
        np.array((0, 1.9, 2.46, 3.24, 4.25, 5.50, 6.91)
                 )[np.newaxis, :, np.newaxis],
        (n_plants, 7, len(t_s))
    )
    phyllochorn = mean_t_p
    ls = (t_ae / phyllochorn - ls_base + 1).astype(int)
    ls[ls < 0] = 0
    ls = ls * np.broadcast_to(stem_i_exists[:, :, np.newaxis], ls.shape)
    n_leaves_s = ls.sum(axis=(0, 1)).tolist()  # list for return
    t_s = t_s.tolist()  # list for return

    # # # # #
    # 4. Add noise of observation,
    #   and apply repeat_observation
    #   to simulate reducing the noise.
    #
    #   Using parameters:
    #   - leaf_tip_observation_error_rate
    #   - repeat_observation
    if leaf_tip_observation_error_rate:
        if repeat_observation is None:
            repeat_observation = 1

        error_factor = 1 \
                       + np.random.randn(repeat_observation,
                                         len(n_leaves_s)).mean(axis=0) \
                       * leaf_tip_observation_error_rate

        n_leaves_s = (np.array(n_leaves_s) * error_factor
                      ).astype(int).tolist()

    if return_dynamics_of_each_plant_without_error:
        return (t_s, n_leaves_s, ls)

    return (t_s, n_leaves_s)


def generate_tiller_probabilities(
        p_no_tiller_min_max,
        p_t0_min_max,
        p_t1_min_max,
        p_t2_min_max,
):
    """
    code example:
        ps = generate_tiller_probabilities(
            (0.10, 0.14),  # pn_mm,
            (0.00, 0.02),  # p0_mm,
            (0.70, 0.74),  # p1_mm,
            (0.14, 0.18),  # p2_mm,
        )

    There are four parameters and one restriction:
        pn+p0+p1+p2=1.

    Generate the three parameters with smaller ranges with uniform distribution,
        so the value of the fourth parameter (who has the largest range) is 1-p_-p_-p_
        and then check if the fourth parameter is within its range.
    """

    pn_mm = p_no_tiller_min_max
    p0_mm = p_t0_min_max
    p1_mm = p_t1_min_max
    p2_mm = p_t2_min_max

    assert (1 >= pn_mm[1] >= pn_mm[0] >= 0)
    assert (1 >= p0_mm[1] >= p0_mm[0] >= 0)
    assert (1 >= p1_mm[1] >= p1_mm[0] >= 0)
    assert (1 >= p2_mm[1] >= p2_mm[0] >= 0)
    assert (np.sum([pn_mm[0], p0_mm[0], p1_mm[0], p2_mm[0]]) <= 1)
    assert (np.sum([pn_mm[1], p0_mm[1], p1_mm[1], p2_mm[1]]) >= 1)

    n_tries = 0
    while True:
        pn = (pn_mm[1] - pn_mm[0]) * np.random.rand() + pn_mm[0]
        p0 = (p0_mm[1] - p0_mm[0]) * np.random.rand() + p0_mm[0]
        p1 = (p1_mm[1] - p1_mm[0]) * np.random.rand() + p1_mm[0]
        p2 = (p2_mm[1] - p2_mm[0]) * np.random.rand() + p2_mm[0]

        # choose the one with the largest range.
        # make it 1-p_-p_-p_.
        # check if it is within the range.
        i_dependent = np.argmax([
            pn_mm[1] - pn_mm[0],
            p0_mm[1] - p0_mm[0],
            p1_mm[1] - p1_mm[0],
            p2_mm[1] - p2_mm[0],
        ])

        if i_dependent == 0:
            pn = 1 - p0 - p1 - p2
            fourth_check = ((-1e-5 <= (pn - pn_mm[0]))
                            & (-1e-5 <= (pn_mm[1] - pn)))  # float comparison
        elif i_dependent == 1:
            p0 = 1 - pn - p1 - p2
            fourth_check = ((-1e-5 <= (p0 - p0_mm[0]))
                            & (-1e-5 <= (p0_mm[1] - p0)))  # float comparison
        elif i_dependent == 2:
            p1 = 1 - pn - p0 - p2
            fourth_check = ((-1e-5 <= (p1 - p1_mm[0]))
                            & (-1e-5 <= (p1_mm[1] - p1)))  # float comparison
        elif i_dependent == 3:
            p2 = 1 - pn - p0 - p1
            fourth_check = ((-1e-5 <= (p2 - p2_mm[0]))
                            & (-1e-5 <= (p2_mm[1] - p2)))  # float comparison
        else:
            raise ValueError('i_dependent')

        # print(fourth_check)
        # pprint([
        #     [pn_mm[0], pn, pn_mm[1]],
        #     [p0_mm[0], p0, p0_mm[1]],
        #     [p1_mm[0], p1, p1_mm[1]],
        #     [p2_mm[0], p2, p2_mm[1]],
        # ])

        if fourth_check:
            return ([pn, p0, p1, p2])
        else:
            n_tries += 1

        if n_tries > 1000:
            raise ValueError('more than 1000 failed tries. check the four p ranges.')


def qmc_scale_mod(sample, l_bounds, u_bounds, reverse=False):
    """
    complement to scipy.stats.qmc.scale.

    scipy.stats.qmc.scale does not accept "u_bounds = l_bounds",
    qmc_scale_mod() accepts that.

    when qmc_scale_mod() meets "None" values in bounds,
    it will return np.nan in that column.

    """

    assert len(l_bounds) == len(u_bounds)

    # check each pair.
    # check passed if:
    # - one of them is None
    # - u = l
    # - u > l
    # flag_1: normal_latin_hypercube. when u > l.
    flag_1 = np.full(len(l_bounds), False)
    # flag_2: None
    flag_2 = np.full(len(l_bounds), False)
    # flag_3: equal. when u == l
    flag_3 = np.full(len(l_bounds), False)

    for lb, ub, i in zip(l_bounds, u_bounds, range(len(l_bounds))):
        if (lb is None) or (ub is None):
            flag_2[i] = True
        else:
            if ub == lb:
                flag_3[i] = True
            elif ub > lb:
                flag_1[i] = True
            elif ub < lb:
                raise ValueError(
                    f'upper_bound {ub} < lower_bound {lb}')

    sample_scaled = np.full(sample.shape, np.nan)

    # flag_1: normal_latin_hypercube
    l_bounds_1 = np.array(l_bounds)[flag_1]
    u_bounds_1 = np.array(u_bounds)[flag_1]
    sample_normal_latin_hypercube = sample[:, flag_1]
    sample_normal_latin_hypercube_scaled = qmc.scale(
        sample_normal_latin_hypercube,
        l_bounds_1,
        u_bounds_1,
    ).astype(float)
    sample_scaled[:, flag_1] = sample_normal_latin_hypercube_scaled

    # flag_2: None
    sample_scaled[:, flag_2] == np.nan

    # flag_3: equal
    u_bounds_3 = np.array(u_bounds)[flag_3].astype(float)
    l_bounds_3 = np.array(l_bounds)[flag_3].astype(float)
    assert np.all(u_bounds_3 == l_bounds_3)
    sample_scaled[:, flag_3] = l_bounds_3

    return sample_scaled


def generate_dataset_of_leaf_dynamics(
        method=('grid',
                'uniform',
                'latin_hypercube',
                'mymethod1',
                'mymethod2',
                )[3],
        t_interval=10,
        t_s_full=None,
        verbose=False,
        random_seed=42,
        d_range=None,
        mean_t_e_range=None,
        std_t_e_range=None,
        mean_t_p_range=None,
        std_t_p_range=None,
        leaf_tip_observation_error_rate=None,
        lower_bounds=None,
        upper_bounds=None,
        size=None,
        repeat_observation=None,
        p_no_tiller_min_max=(0.12, 0.12),
        p_t0_min_max=(0.00, 0.00),
        p_t1_min_max=(0.72, 0.72),
        p_t2_min_max=(0.16, 0.16),
):
    r"""
    :param method: 'grid' or 'uniform' or 'latin_hypercube'
    :param t_interval:
    :param verbose:
    :param random_seed:
    :param d_range:
    :param mean_t_e_range:
    :param std_t_e_range:
    :param mean_t_p_range:
    :param std_t_p_range:
    :param leaf_tip_observation_error_rate:
    :param lower_bounds: (d, mean_t_e, std_t_e, mean_t_p, std_t_p)
    :param upper_bounds: (d, mean_t_e, std_t_e, mean_t_p, std_t_p)
    :param size:
    :param repeat_observation:
        when there is an error in leaf_tip_observation,
        averaging over repeated observations to reduce error rate.
        This parameter only works when
        leaf_tip_observation_error_rate is not None
    :param p_no_tiller_min_max:
        The min and max value of the probability that,
        a plant in a plot has no tillers.
        Each plot has a probability value.
    :param p_t1_in_tillers_min_max:
        The min and max value of the probability that,
        when a plant in a plot has tiller,
        the tillering starts from T1.
        Each plot has a probability value.
    :return:


    -----------update 2023-9-17--------------------
    Allow different tiller situations.

    example_often use:
        ds = generate_dataset_of_leaf_dynamics(
            method='mymethod1',
            size=45441,
            lower_bounds=(150, 140, 0, None, 0),
            upper_bounds=(450, 180, 50, None, 0),
            leaf_tip_observation_error_rate=0.1,
            verbose=True,
        )

    example1:
        ds = generate_dataset_of_leaf_dynamics(
            method='grid',
            d_range=range(150, 601, 200),
            mean_t_e_range=range(140, 181, 20),
            std_t_e_range=range(0, 51, 25),
            mean_t_p_range=range(80, 121, 20),
            std_t_p_range=range(0, 21, 10),
            leaf_tip_observation_error_rate=0.1,
        )

    example2:
        ds = generate_dataset_of_leaf_dynamics(
            method='uniform',
            size=100,
            lower_bounds=(150, 140, 0, 80, 0),
            upper_bounds=(600, 180, 50, 120, 0),
            leaf_tip_observation_error_rate=0.1,
        )

    example3:
        ds = generate_dataset_of_leaf_dynamics(
            method='latin_hypercube',
            size=100,
            lower_bounds=(150, 140, 0, 80, 0),
            upper_bounds=(600, 180, 50, 120, 0),
            leaf_tip_observation_error_rate=0.1
        )

    """
    grow_param_s = []
    np.random.seed(random_seed)

    if t_s_full is None:
        raise ValueError('set t_s_full')  # default

    if method == 'grid':
        # check if these parameters exist:
        #   d_range,
        #   mean_t_e_range,
        #   std_t_e_range,
        #   mean_t_p_range,
        #   std_t_p_range,
        if d_range is None:
            raise ValueError(f'd_range cannot be None when method = {method}')
        if mean_t_e_range is None:
            raise ValueError(f'mean_t_e_range cannot be None when method = {method}')
        if std_t_e_range is None:
            raise ValueError(f'std_t_e_range cannot be None when method = {method}')
        if mean_t_p_range is None:
            raise ValueError(f'mean_t_p_range cannot be None when method = {method}')
        if std_t_p_range is None:
            raise ValueError(f'std_t_p_range cannot be None when method = {method}')
        # collect values to grow_param_s
        for d in d_range:
            for mean_t_e in mean_t_e_range:
                for std_t_e in std_t_e_range:
                    for mean_t_p in mean_t_p_range:
                        for std_t_p in std_t_p_range:
                            grow_param_s.append([d, mean_t_e, std_t_e, mean_t_p, std_t_p])

    elif method == 'uniform':
        # uniform distribution between lower_bound and upper_bound
        # check if these parameters exist:
        # lower_bounds
        # upper_bounds
        # size ...
        if lower_bounds is None:
            raise ValueError(f'lower_bounds cannot be None when method = {method}')
        if upper_bounds is None:
            raise ValueError(f'upper_bounds cannot be None when method = {method}')
        if size is None:
            raise ValueError(f'size cannot be None when method = {method}')

        density_s = np.random.uniform(low=lower_bounds[0], high=upper_bounds[0], size=size).astype(int).tolist()
        mean_t_e_s = np.random.uniform(low=lower_bounds[1], high=upper_bounds[1], size=size).astype(int).tolist()
        std_t_e_s = np.random.uniform(low=lower_bounds[2], high=upper_bounds[2], size=size).astype(int).tolist()
        mean_t_p_s = np.random.uniform(low=lower_bounds[3], high=upper_bounds[3], size=size).astype(int).tolist()
        std_t_p_s = np.random.uniform(low=lower_bounds[4], high=upper_bounds[4], size=size).astype(int).tolist()

        for i in range(size):
            grow_param_s.append(
                [density_s[i],
                 mean_t_e_s[i],
                 std_t_e_s[i],
                 mean_t_p_s[i],
                 std_t_p_s[i]]
            )

    elif method == 'latin_hypercube':
        # latin_hypercube sampling between lower_bound and upper_bound
        # check if these parameters exist:
        # lower_bounds
        # upper_bounds
        # size ...
        if lower_bounds is None:
            raise ValueError(f'lower_bounds cannot be None when method = {method}')
        if upper_bounds is None:
            raise ValueError(f'upper_bounds cannot be None when method = {method}')
        if size is None:
            raise ValueError(f'size cannot be None when method = {method}')

        sampler = qmc.LatinHypercube(d=5, seed=random_seed)
        sample = sampler.random(n=size)
        sample_scaled = qmc_scale_mod(sample, lower_bounds, upper_bounds).astype(int)
        grow_param_s = sample_scaled.tolist()

    elif method == 'mymethod1':
        # In mymethod1, most parameters were set as latin-hypercube.
        # Only that the phyllochron was fixed to a logistic sampling.
        # Reference: Data of 106 observations show that
        #   phyllochron could be simulated with logistic(110.938, 7.296).
        #   See in Ghita (2023) (under Samuel Buis' supervision)

        assert not (lower_bounds is None)
        assert not (upper_bounds is None)
        assert not (size is None)

        sampler = qmc.LatinHypercube(d=5, seed=random_seed)
        sample = sampler.random(n=size)
        sample_scaled = qmc_scale_mod(sample, lower_bounds, upper_bounds)

        # Generate phyllochron and only keep values within [80, 150].
        while True:
            phyllochron = np.random.logistic(
                110.938, 7.296, size=int(size * 1.5)).astype(int)
            phyllochron = phyllochron[(phyllochron >= 80) & (phyllochron <= 150)]
            if phyllochron.size > size:
                phyllochron = phyllochron[:size]
                break
        sample_scaled[:, 3] = phyllochron
        grow_param_s = sample_scaled.astype(int).tolist()

    elif method == 'mymethod2':
        # In mymethod2, most parameters were set as latin-hypercube.
        # Only that the phyllochron was fixed to a normal function
        #
        # Reference: From the observations in the experiment of
        #   Avignon 2023-2024 (by Tiancheng and Sylvain),
        #   The early phyllochrons calculated using Haun-stage
        #   may be described using normal distribution
        #   (p-value=0.7559, mean=85.365, std=11.280).
        #
        # Remove phyllochron values less than observed_min
        # or more than observed_max, as an imitation of "mymethod1".

        assert not (lower_bounds is None)
        assert not (upper_bounds is None)
        assert not (size is None)

        sampler = qmc.LatinHypercube(d=5, seed=random_seed)
        sample = sampler.random(n=size)
        sample_scaled = qmc_scale_mod(sample, lower_bounds, upper_bounds)

        # generate phyllochron values in LUT
        while True:
            phyllochron = np.random.normal(
                85.365, 11.280, size=int(size * 1.5)).astype(int)
            phyllochron = phyllochron[(phyllochron >= 56.52) & (phyllochron <= 116.68)]
            if phyllochron.size > size:
                phyllochron = phyllochron[:size]
                break
        sample_scaled[:, 3] = phyllochron
        grow_param_s = sample_scaled.astype(int).tolist()
    else:
        raise ValueError('check parameter: method')

    # add random tiller settings
    grow_param_s_arr = np.array(grow_param_s).astype('object')

    p_s_s = []
    for i in range(grow_param_s_arr.shape[0]):
        p_s = generate_tiller_probabilities(
            p_no_tiller_min_max,
            p_t0_min_max,
            p_t1_min_max,
            p_t2_min_max,
        )
        p_s_s.append(p_s)
    grow_param_s_arr = np.concatenate((grow_param_s_arr, np.array(p_s_s)), axis=1)
    grow_param_s = grow_param_s_arr.tolist()

    # fill "observation" with "grow_param"
    i = 0
    leaf_dynamic_dataset = {}
    if verbose:
        print('generating leaf_dynamic_dataset:')
    for grow_param in tqdm(grow_param_s, disable=(not verbose)):
        t_s, n_leaves_s = get_n_leaves_dynamic_of_plot(
            d=grow_param[0],
            mean_t_e=grow_param[1],
            std_t_e=grow_param[2],
            mean_t_p=grow_param[3],
            std_t_p=grow_param[4],
            p_no_tiller=grow_param[5],
            p_start_tiller_0=grow_param[6],
            p_start_tiller_1=grow_param[7],
            p_start_tiller_2=grow_param[8],
            plot_size=1,
            t_interval=t_interval,
            leaf_tip_observation_error_rate=leaf_tip_observation_error_rate,
            repeat_observation=repeat_observation,
            random_seed=np.random.randint(0, 42000),
            t_s_full=t_s_full,
        )

        leaf_dynamic_dataset[str(i)] = {
            'grow_param': {
                'density': grow_param[0],
                'mean_t_e': grow_param[1],
                'std_t_e': grow_param[2],
                'mean_t_p': grow_param[3],
                'std_t_p': grow_param[4],
                'plot_size': 1,
                'p_no_tiller': grow_param[5],
                'p_start_tiller_0': grow_param[6],
                'p_start_tiller_1': grow_param[7],
                'p_start_tiller_2': grow_param[8],
            },
            'observation': {
                't_interval': t_interval,
                't_s': t_s,
                'n_leaves_s': n_leaves_s
            }
        }
        i += 1

    return leaf_dynamic_dataset


def dataset_to_lut(leaf_dynamic_dataset):
    """
    make lut for invert-estimation.

    n_rows: len(lut)
    n_cols: 5 + ...
            [density, mean_t_e, std_t_e, mean_t_p, std_t_p, ...(n columns of n_leaves_dynamic)]
    """

    # initialize t_s
    k0 = list(leaf_dynamic_dataset.keys())[0]
    ts0 = leaf_dynamic_dataset[k0]['observation']['t_s']
    t_s_full = set(ts0)
    for k, v in leaf_dynamic_dataset.items():
        if len(v['observation']['t_s']) != len(ts0):
            warnings.warn()
        t_s_full = t_s_full | set(v['observation']['t_s'])
    t_s_full = list(t_s_full)
    t_s_full.sort()

    lut = np.full((len(leaf_dynamic_dataset), 5 + len(t_s_full)), np.nan)

    i = 0
    for key in leaf_dynamic_dataset:
        # when leaf dynamic have different lengths
        # (same head but different tails),
        # fill with nan
        this_row = (
                [
                    leaf_dynamic_dataset[key]['grow_param']['density'],
                    leaf_dynamic_dataset[key]['grow_param']['mean_t_e'],
                    leaf_dynamic_dataset[key]['grow_param']['std_t_e'],
                    leaf_dynamic_dataset[key]['grow_param']['mean_t_p'],
                    leaf_dynamic_dataset[key]['grow_param']['std_t_p']
                ]
                + leaf_dynamic_dataset[key]['observation']['n_leaves_s']
        )
        this_row += [np.nan] * (5 + len(t_s_full) - len(this_row))
        this_row = np.array(this_row)
        lut[i, :] = this_row
        i += 1
    return lut


def plant_density_estimation_from_leaf_tip_dynamics__per_site(
        use_what_dynamic_as_input='corrected_p2pnet_detection',
        flag_show_image=False,
):
    # ---setting variables---
    df = df_dataset_index_xlsx

    fig_out_dir = r'C:\Users\Yang\PycharmProjects\pythonProject\leaf_tip_dynamic_figures'
    Path(fig_out_dir).mkdir(parents=True, exist_ok=True)
    print(f'fig_out_dir: {fig_out_dir}')
    xlim_dyn = (0, 500)
    ylim_dyn = (0, 2000)
    xlim_den = ylim_den = (0, 500)
    text_y_relative_anchor = 0.6
    tt_name_dict = {
        1: 'note11 Air temperature_Thermal time #1',
        2: 'note12 Air temperature_Thermal time #2',
        3: 'note13 Air temperature_Thermal time #3',
        4: 'note14 Air temperature_Thermal time #4',
    }
    tt_name = tt_name_dict[4]
    scatter_point_size = 54
    plt.rcParams.update({
        "font.size": 14,  # 全局字体大小
        "axes.titlesize": 14,  # 坐标轴标题字体
        "axes.labelsize": 14,  # 坐标轴标签字体
        "xtick.labelsize": 12,  # x轴刻度字体
        "ytick.labelsize": 12,  # y轴刻度字体
        "legend.fontsize": 12  # 图例字体
    })
    dynamic_scatter_alpha = 0.8
    flag_show_ground_truth_plant_density = False
    flag_show_chosen_tt_obs = True
    flag_show_p2pnet_detection_density = False
    tt_interval = 1
    min_phyl, max_phyl = (75.66866666666668, 95.46066666666665)  # data from experiment
    n_best_from_lut = 5
    lut_size = 1000
    pass
    ## tt_obs chosen based on corrected-P2PNET-detected-leaf-tip-dynamics
    t_s_full = np.arange(0, 780, tt_interval)
    tt_chosen_avignon = (238, 246)
    tt_chosen_fourques = (260, 348)
    tt_chosen_salin = (168, 283)
    ## try full tt_obs
    # tt_chosen_avignon = t_s_full
    # tt_chosen_fourques = t_s_full
    # tt_chosen_salin = t_s_full
    std_t_e = 30
    random_seed_train = 748
    random_seed_val = 648
    # --- end of setting variables ---
    pass
    print('building LUTs...', end='')
    dataset_avignon2324 = generate_dataset_of_leaf_dynamics(
        method='latin_hypercube',
        size=lut_size,
        lower_bounds=(90, 115, std_t_e, min_phyl - 2, 0),
        upper_bounds=(450, 125, std_t_e, max_phyl + 2, 0),
        p_no_tiller_min_max=(0.12, 0.12),
        p_t0_min_max=(0.01, 0.01),
        p_t1_min_max=(0.71, 0.71),
        p_t2_min_max=(0.16, 0.16),
        t_interval=tt_interval,
        t_s_full=t_s_full,
        random_seed=random_seed_train,
    )
    lut_avignon2324 = dataset_to_lut(dataset_avignon2324)
    dataset_fourques2223 = generate_dataset_of_leaf_dynamics(
        method='latin_hypercube',
        size=lut_size,
        lower_bounds=(90, 160, std_t_e, min_phyl - 2, 0),
        upper_bounds=(450, 200, std_t_e, max_phyl + 2, 0),
        p_no_tiller_min_max=(0.12, 0.12),
        p_t0_min_max=(0.01, 0.01),
        p_t1_min_max=(0.71, 0.71),
        p_t2_min_max=(0.16, 0.16),
        t_interval=tt_interval,
        t_s_full=t_s_full,
        random_seed=random_seed_train,
    )
    lut_fourques2223 = dataset_to_lut(dataset_fourques2223)
    dataset_salin2223 = generate_dataset_of_leaf_dynamics(
        method='latin_hypercube',
        size=lut_size,
        lower_bounds=(90, 106, std_t_e, min_phyl - 2, 0),
        upper_bounds=(450, 130, std_t_e, max_phyl + 2, 0),
        # To get "the result of 2024-7-1",
        # use mean_Te_range = 101~129
        p_no_tiller_min_max=(0.12, 0.12),
        p_t0_min_max=(0.01, 0.01),
        p_t1_min_max=(0.71, 0.71),
        p_t2_min_max=(0.16, 0.16),
        t_interval=tt_interval,
        t_s_full=t_s_full,
        random_seed=random_seed_train,
    )
    lut_salin2223 = dataset_to_lut(dataset_salin2223)
    print('finish building LUT')
    # ---end of setting variables---

    if not flag_show_image:
        matplotlib.use('agg')
    df_c = get_dataset_for_dynamic(df)

    colors = list(mcolors.TABLEAU_COLORS.values())  # matplotlib default colors

    # extract the dynamic for one plot:
    #   One plot has the same ['Site', 'Date of sowing', 'note2 plot_num']
    #   This is a key.
    plot_key_lst = []
    for i in df_c.index:
        plot_key_lst.append((df_c.at[i, 'Site'],
                             df_c.at[i, 'Date of sowing'],
                             df_c.at[i, 'note2 plot_num']))
    plot_key_set = set(plot_key_lst)

    est_den_lst = []
    manual_den_lst = []
    site_lst = []

    for plot_key in plot_key_set:
        site = plot_key[0]
        year = plot_key[1].year
        plot_num = plot_key[2]

        dfi = df_c[
            (df_c['Site'] == plot_key[0])
            & (df_c['Date of sowing'] == plot_key[1])
            & (df_c['note2 plot_num'] == plot_key[2])
            ]
        dfi = dfi.sort_values('Date of measurement')

        # check: one plot should have the same plant density value
        assert len(pd.unique(dfi['Manual density value (plants/m2)'])) == 1
        plant_density = pd.unique(dfi['Manual density value (plants/m2)']).item()

        fig, ax = plt.subplots()

        tt = dfi[tt_name].tolist()
        s_area = dfi['Soil surface area of target (m2)'].tolist()

        lt_den2 = dfi['note15 P2PNET_N016_detected_masked leaf_tip density (#/m2)'].tolist()
        if flag_show_p2pnet_detection_density:
            ax.scatter(tt, lt_den2, color=colors[3], s=scatter_point_size,
                       alpha=dynamic_scatter_alpha,
                       label='P2PNet detection density',
                       edgecolor='none',
                       )

        lt_den3 = dfi['note17 Annotated leaf_tip density (#/m2)'].tolist()
        if sum(~pd.isnull(lt_den3)) > 0:
            ax.scatter(tt, lt_den3, color=colors[2], s=scatter_point_size,
                       alpha=dynamic_scatter_alpha,
                       label='On-image manual density',
                       edgecolor='none', )

        lt_den4 = dfi[
            'note32 calibrated P2P detected leaf tip density (Sylvain_variety_wize_CV)'].tolist()
        ax.scatter(tt, lt_den4, color=colors[1], s=scatter_point_size,
                   alpha=dynamic_scatter_alpha,
                   label='Corrected P2PNet detection density',
                   edgecolor='none', )

        # draw the "manual" lastly, because it has fewer points
        lt_den1 = dfi['Manual leaf tip density value (interpolated)  (#leaf tip/m2)'].tolist()
        ax.scatter(tt, lt_den1, color=colors[0], s=scatter_point_size,
                   alpha=dynamic_scatter_alpha,
                   label='In-field manual density',
                   edgecolor='none', )

        dbg = True

        # LUT-invert, and plot on figures:
        if use_what_dynamic_as_input == 'corrected_p2pnet_detection':
            tta, lta = adjust_t_s_and_n_leaves_s(tt, lt_den4, t_s_full=t_s_full)
        elif use_what_dynamic_as_input == 'p2pnet_detection':
            tta, lta = adjust_t_s_and_n_leaves_s(tt, lt_den2, t_s_full=t_s_full)
        else:
            raise ValueError('Check parameter: use_what_dynamic_as_input')

        if site == 'Avignon':
            lut = lut_avignon2324
            tt_chosen = tt_chosen_avignon
            dbg = True  # debugging scratch_1.py
        elif site == 'Fourques':
            lut = lut_fourques2223
            tt_chosen = tt_chosen_fourques
        elif site == 'Salin-de-Giraud':
            lut = lut_salin2223
            tt_chosen = tt_chosen_salin
        else:
            raise ValueError

        if flag_show_chosen_tt_obs:
            for x in tt_chosen:
                ax.vlines(x, ylim_dyn[0], ylim_dyn[1], linestyles='dotted',
                          colors='black')
            ax.vlines([], ylim_dyn[0], ylim_dyn[1],
                      linestyles='dotted',
                      colors='black',
                      label=f'chosen tt_obs')
            # ax.vlines([], ylim_dyn[0], ylim_dyn[1],
            #           linestyles='dotted',
            #           colors='black',
            #           label=f'chosen tt_obs: {tt_chosen}')

        if flag_show_ground_truth_plant_density:
            ax.plot([], [], alpha=0, label=f'Ground truth:{plant_density:.1f} (plants/m²)')

        try:
            r = lut_invert(lut, tta, lta, tt_chosen,
                           n_best_from_lut=n_best_from_lut,
                           t_interval=tt_interval,
                           t_s_full=t_s_full,
                           )
            est_den = r['best_n_params_mean'][0]
            ax.plot(t_s_full[:, np.newaxis],
                    r['best_n_dynamics'].T,
                    color='gray',
                    alpha=1.5 / r['best_n_dynamics'].shape[0], )
            # ax.plot([], [], color='gray', alpha=0.5,
            #         label=f'LUT estimations:{est_den:.1f}(plants/m²)')
            ax.plot([], [], color='gray', alpha=0.5,
                    label=f'LUT estimations')
        except ValueError as e:
            if str(e) != 'No value available for estimation.Maybe t_s_used_for_lut_invert and t_s do not overlap.':
                raise e
            ax.plot([], [], color='gray', alpha=0.5, label=f'LUT estimations: None')
            est_den = None

        ax.grid(True)
        ax.legend(loc='upper left', facecolor='white', framealpha=1)

        fig_name = f"{site}_plot{plot_num}"
        density_text = f'\nplant density:{plant_density:.1f}(#/m²)'
        ax.set_title(fig_name + density_text)
        ax.set_xlabel("Thermal time after sowing (℃·d)")
        ax.set_ylabel("Leaf tip density (leaf tips/m²)")
        ax.set_xlim(xlim_dyn)
        ax.set_ylim(ylim_dyn)

        fig.savefig(Path(fig_out_dir) / (fig_name + '.png'), dpi=300)

        est_den_lst.append(est_den)
        manual_den_lst.append(plant_density)
        site_lst.append(site)

    # plot overall-scatter
    for site in set(site_lst):
        chosen_ind = np.argwhere(np.array(site_lst) == site)
        chosen_est_den = np.array(est_den_lst)[chosen_ind]
        chosen_manual_den = np.array(manual_den_lst)[chosen_ind]
        fig, ax = plt.subplots()
        scatter_and_show_accuracy(ax, chosen_manual_den, chosen_est_den,
                                  rmse_round_digits=0, axis_range=xlim_den,
                                  show_pearson_corrcoef=True,
                                  text_y_relative_anchor=text_y_relative_anchor,
                                  scatter_size=scatter_point_size,
                                  )
        ax.set_xlabel('Manual plant density (plants/m²)')
        ax.set_ylabel('Estimated plant density (plants/m²)')
        ax.set_title(site)
        fig.savefig((Path(fig_out_dir) / f'00_est_scatter_{site}').__str__(), dpi=300)

    if not flag_show_image:
        plt.close('all')
        matplotlib.use('tkagg')  # go back to interactive backend

    dbg = True

    return {
        'est_den_lst': est_den_lst,
        'manual_den_lst': manual_den_lst,
        'site_lst': site_lst,
    }


def results_3_4_3__visualise_lut_est():
    plant_density_estimation_from_leaf_tip_dynamics__per_site(
        use_what_dynamic_as_input='corrected_p2pnet_detection')
    pass


if __name__ == "__main__":
    pass
    results_3_4_3__visualise_lut_est()
    pass
