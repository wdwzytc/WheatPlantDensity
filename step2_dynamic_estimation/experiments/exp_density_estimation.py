import numpy as np
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from ..data_loader.loader import load_dataset, get_dataset_for_dynamic
from ..simulation.generator import generate_dataset_of_leaf_dynamics
from ..simulation.lut_builder import dataset_to_lut
from ..core.lut import lut_invert
from ..core.preprocess import adjust_t_s_and_n_leaves_s
from ..visualization.plot_utils import scatter_and_show_accuracy
import matplotlib.colors as mcolors
import os


def run_density_estimation(
        fig_out_dir=r'./outputs',
        input_source='corrected_p2pnet_detection',
        flag_show_image=False,
):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = load_dataset(os.path.join(BASE_DIR, "../../data/leaftip_dynamic_data/dataset_index.xlsx"))

    # ---setting variables---

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

        # LUT-invert, and plot on figures:
        if input_source == 'corrected_p2pnet_detection':
            tta, lta = adjust_t_s_and_n_leaves_s(tt, lt_den4, t_s_full=t_s_full)
        else:
            raise ValueError('Check parameter: input_source(use_what_dynamic_as_input)')

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
