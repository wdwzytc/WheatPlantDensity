import numpy as np
import warnings

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
