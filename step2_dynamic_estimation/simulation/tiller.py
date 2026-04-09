import numpy as np

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
