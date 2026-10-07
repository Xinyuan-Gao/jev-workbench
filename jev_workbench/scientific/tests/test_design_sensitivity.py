import pytest
from scipy.stats import binomtest

from jev_workbench.scientific.design_sensitivity import exact_paired_power, worst_case_wilson_half_width


def test_exact_power_matches_enumeration_of_small_paired_trial():
    # Four independent tasks: A-only probability .35, B-only .15, ties .5.
    import itertools
    probability = 0.0
    for outcomes in itertools.product((0, 1, 2), repeat=4):
        a = outcomes.count(1)
        b = outcomes.count(2)
        p = binomtest(a, a+b, .5).pvalue if a+b else 1.0
        if p <= .2:
            weight = 1.0
            for item in outcomes:
                weight *= (.5, .35, .15)[item]
            probability += weight
    assert exact_paired_power(4, .2, .5, alpha=.2) == pytest.approx(probability)


def test_null_rejection_is_no_larger_than_alpha_and_more_samples_help():
    assert exact_paired_power(80, 0, .4, alpha=.05) <= .05
    assert exact_paired_power(500, .05, .2) > exact_paired_power(200, .05, .2)


@pytest.mark.parametrize('n,d,q,alpha', [(0,.05,.2,.05),(10,.3,.2,.05),(10,.05,0,.05),(10,.05,.2,0)])
def test_invalid_assumptions_are_rejected(n,d,q,alpha):
    with pytest.raises(ValueError):
        exact_paired_power(n,d,q,alpha=alpha)


def test_worst_case_width_decreases_with_n():
    assert worst_case_wilson_half_width(200) == pytest.approx(.068639, abs=.00001)
    assert worst_case_wilson_half_width(558) < worst_case_wilson_half_width(200)
