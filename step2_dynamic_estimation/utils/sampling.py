import numpy as np
from scipy.stats import qmc

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
