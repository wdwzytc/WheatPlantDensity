import numpy as np
import numbers

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
