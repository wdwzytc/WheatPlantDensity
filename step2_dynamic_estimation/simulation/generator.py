import numpy as np
from tqdm import tqdm
from ..simulation.plant_model import get_n_leaves_dynamic_of_plot
from ..simulation.tiller import generate_tiller_probabilities
from scipy.stats import qmc
from ..utils.sampling import qmc_scale_mod


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
