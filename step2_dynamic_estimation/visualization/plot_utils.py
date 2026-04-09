import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr

from ..utils.metrics import relative_root_mean_square_error

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
