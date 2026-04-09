import numpy as np
import warnings

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
