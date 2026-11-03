"""Option matching for the statistics store: signs, zero, scales, ties, and invariance to the options' order, for the
descriptive owid_mem.choose and the runtime's xa_num.pick (which must agree)."""
import itertools
import random

from experiments.statistics import owid_mem
from experiments.statistics.runtime import xa_num


def both(values, options):
    a = owid_mem.choose([dict(value=v, weight=w) for v, w in values], options)
    b = xa_num.pick(values, options)
    assert a == b
    return a


def test_exact_and_near():
    assert both([(56.5, 0.9)], ["22.1", "44.1", "79.8", "56.5"]) == 3
    assert both([(56.5, 0.9)], ["22.1", "44.1", "79.8", "57"]) == 3


def test_sign_is_kept():
    assert both([(-2.0, 0.9)], ["2", "-2", "5", "-9"]) == 1          # not the +2 that comes first
    assert both([(-2.0, 0.9)], ["−2.1", "2", "3", "4"]) == 0
    assert both([(2.0, 0.9)], ["-2", "2", "-5", "-9"]) == 1


def test_mixed_signs_never_match():
    assert both([(-2.0, 0.9)], ["2", "3", "4", "5"]) is None
    assert both([(3.0, 0.9)], ["-3", "-1", "-0.5", "-7"]) is None


def test_zero():
    assert both([(0.0, 0.9)], ["1", "0", "2", "-1"]) == 1
    assert both([(0.0, 0.9)], ["1", "2", "3", "4"]) is None
    assert both([(3.0, 0.9)], ["0", "3.1", "9", "20"]) == 1


def test_scales_and_percent():
    assert both([(1.23e6, 0.9)], ["12,300", "1.23 million", "3 billion", "500"]) == 1
    assert both([(45.0, 0.9)], ["45%", "60%", "20%", "80%"]) == 0
    assert both([(2.5e9, 0.9)], ["$2.5 billion", "$250 million", "$25 billion", "$2.5 trillion"]) == 0


def test_ties_abstain():
    assert both([(10.0, 0.9)], ["5", "20", "100", "1"]) is None       # 5 and 20 are equally far (ratio 2)
    assert both([(4.0, 0.5), (16.0, 0.5)], ["8", "x", "y", "z"]) == 0


def test_bracketed_numbers_do_not_parse():
    assert both([(5.0, 0.9)], ["(5)", "(A)", "(-5)", "( 5 )"]) is None
    assert both([(5.0, 0.9)], ["(5)", "5.1", "9", "20"]) == 1


def test_whole_string_and_fractions():
    assert owid_mem.option_value("1/2") == 0.5 == xa_num.val("1/2")
    assert owid_mem.option_value("-3/4") == -0.75 == xa_num.val("-3/4")
    for bad in ("1 1/2", "5 apples", "5x", "~5", "1,23", "1.5/2", "3/0", "1/2/3", "5 %%", "about 5", "2.5e3", ".5"):
        assert owid_mem.option_value(bad) != owid_mem.option_value(bad), bad          # NaN: unsupported
        assert xa_num.val(bad) != xa_num.val(bad), bad
    assert owid_mem.option_value("1,234") == 1234 and owid_mem.option_value("1,234.5") == 1234.5
    assert both([(1.0, 1.0)], ["1/2", "2"]) is None                 # 1/2 and 2 are equally far from 1: a tie
    assert both([(0.5, 0.9)], ["1/2", "0.6"]) == 0                  # the exact fraction wins
    assert both([(0.5, 0.9)], ["1/2", "0.5"]) is None               # equal values: a tie
    assert both([(5.0, 0.9)], ["5 apples", "5x", "~5", "9"]) == 3   # only "9" parses


def test_nothing_parses_or_nothing_retrieved():
    assert both([(3.0, 0.9)], ["a", "b", "c", "d"]) is None
    assert both([], ["1", "2", "3", "4"]) is None


def test_invariant_to_option_order():
    rng = random.Random(0)
    for _ in range(200):
        values = [(rng.choice([-1, 1]) * rng.choice([0, 10 ** rng.uniform(-3, 9)]), rng.random()) for _ in range(3)]
        options = [f"{rng.choice([-1, 1]) * 10 ** rng.uniform(-3, 9):.3g}" for _ in range(4)]
        picked = {None if c is None else p[c] for p in itertools.permutations(options) for c in [both(values, list(p))]}
        assert len(picked) == 1, (values, options, picked)
