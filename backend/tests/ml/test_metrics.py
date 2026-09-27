"""Hand-computed cases for every metric (arithmetic shown in comments)."""
import numpy as np
import pytest

from disstruments.ml.eval import metrics as M
from disstruments.ml.taxonomy import Taxonomy


def test_ap_basic():
    # ranked: 1(.9) 0(.8) 1(.7) 0(.1) -> P@1=1 (R=.5), P@3=2/3 (R=1)
    # AP = .5*1 + .5*(2/3) = 0.8333...
    assert M.average_precision([1, 0, 1, 0], [.9, .8, .7, .1]) == pytest.approx(5 / 6)
    assert M.average_precision([1, 1, 0], [.9, .8, .1]) == pytest.approx(1.0)


def test_ap_ties_grouped():
    # tie group {.5: one pos, one neg} -> P=.5, R=.5 ; then .2 (pos) -> P=2/3, R=1
    # AP = .5*.5 + .5*(2/3) = 0.58333 (sklearn's definition; tie order irrelevant)
    assert M.average_precision([1, 0, 1], [.5, .5, .2]) == pytest.approx(0.25 + 1 / 3)
    assert M.average_precision([0, 1, 1], [.5, .5, .2]) == pytest.approx(0.25 + 1 / 3)


def test_ap_requires_positive():
    with pytest.raises(ValueError):
        M.average_precision([0, 0], [.1, .2])


def test_masked_ap_and_skip_reasons():
    y = np.array([[1, 0, 1, 1],
                  [0, 0, 1, 0],
                  [1, 0, 1, 0],
                  [0, 0, 1, 1]], bool)
    s = np.array([[.9, .1, .5, .2],
                  [.8, .2, .5, .9],
                  [.7, .3, .5, .1],
                  [.1, .4, .5, .8]])
    m = np.ones_like(y)
    m[1, 0] = False          # col0 without row1: ranked 1(.9) 1(.7) 0(.1) -> AP 1.0
    m[:, 3] = False          # col3 unobserved
    ap, reasons = M.per_class_ap(y, s, m)
    assert ap[0] == pytest.approx(1.0)
    assert reasons == [None, M.SKIP_NO_POSITIVES, M.SKIP_NO_NEGATIVES, M.SKIP_UNOBSERVED]
    assert np.isnan(ap[1:]).all()
    macro = M.mean_average_precision(y, s, m)
    assert macro.value == pytest.approx(1.0) and macro.n_classes == 1 and macro.n_skipped == 3
    assert macro.skipped == {"no_positives": 1, "no_negatives": 1, "unobserved": 1}
    # unmasked col0: 1(.9) 0(.8) 1(.7) 0(.1) -> 5/6 ; mask changes the answer
    assert M.per_class_ap(y, s)[0][0] == pytest.approx(5 / 6)


def test_all_positive_column_would_inflate_map():
    # a column only ever observed when positive: AP would be 1.0 for ANY scores -> skipped
    y = np.array([[1], [1]], bool)
    s = np.array([[.01], [.02]])
    assert M.per_class_ap(y, s)[1] == [M.SKIP_NO_NEGATIVES]
    assert M.mean_average_precision(y, s).value is None


def test_prf_at_threshold_masked():
    y = np.array([[1, 0], [1, 0], [0, 0], [1, 1]], bool)
    s = np.array([[.9, .7], [.4, .6], [.6, .1], [.8, .9]])
    m = np.array([[1, 1], [1, 1], [1, 1], [1, 0]], bool)
    prf = M.prf_at_threshold(y, s, m, 0.5)
    # col0: pred [1,0,1,1] vs y [1,1,0,1] -> tp2 fp1 fn1 -> P=R=F1=2/3
    assert (prf.tp[0], prf.fp[0], prf.fn[0]) == (2, 1, 1)
    assert prf.f1[0] == pytest.approx(2 / 3)
    # col1: observed rows 0-2 all negative -> skipped; but its 2 false alarms still count in micro
    assert prf.reasons[1] == M.SKIP_NO_POSITIVES and np.isnan(prf.f1[1])
    assert (prf.fp[1], prf.tp[1]) == (2, 0)
    micro = M.micro_prf(prf)
    # tp2 fp3 fn1 -> P=.4 R=2/3 F1=.5
    assert micro["precision"] == pytest.approx(0.4) and micro["recall"] == pytest.approx(2 / 3)
    assert micro["f1"] == pytest.approx(0.5)
    macro = M.macro_f1(prf)
    assert macro.value == pytest.approx(2 / 3) and macro.n_skipped == 1


def test_prf_no_predictions_is_zero_not_nan():
    prf = M.prf_at_threshold(np.array([[1], [0]], bool), np.array([[.1], [.2]]), None, 0.5)
    assert prf.precision[0] == 0.0 and prf.recall[0] == 0.0 and prf.f1[0] == 0.0


@pytest.fixture(scope="module")
def tiny():
    return Taxonomy.from_dict({"version": "0.1.0", "nodes": [
        {"id": "a"}, {"id": "a.x", "data": ["slakh"]}, {"id": "a.y", "data": ["slakh"]},
        {"id": "b", "data": ["slakh"]}]})


def test_hierarchical_partial_credit(tiny):
    anc = tiny.ancestor_matrix()                 # columns: a, a.x, a.y, b
    y = np.array([[0, 1, 0, 0]], bool)           # truth: a.x
    pred = np.array([[0, 0, 1, 0]], bool)        # predicted sibling a.y
    # P^ = {a, a.y}, T^ = {a, a.x}: |P^&T^| = 1 -> hP = hR = hF = 0.5 (flat F1 would be 0)
    h = M.hierarchical_prf(y, pred, np.ones_like(y), anc)
    assert (h["precision"], h["recall"], h["f1"]) == (0.5, 0.5, 0.5)
    # masking a.y (unknown) removes the wrong leaf: P^ = {a} -> hP = 1, hR = 1/2, hF = 2/3
    m = np.array([[1, 1, 0, 1]], bool)
    h = M.hierarchical_prf(y, pred, m, anc)
    assert h["precision"] == 1.0 and h["recall"] == 0.5 and h["f1"] == pytest.approx(2 / 3)


def test_hierarchical_micro_over_samples(tiny):
    anc = tiny.ancestor_matrix()
    y = np.array([[0, 1, 0, 0], [0, 0, 0, 1]], bool)       # a.x ; b
    pred = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)    # a.x (exact) ; a (wrong family)
    # s1: P^=T^={a,a.x} -> inter 2 ; s2: P^={a}, T^={b} -> inter 0
    # hP = 2/3, hR = 2/3 (micro: sums, not a mean of per-sample F1s)
    h = M.hierarchical_prf(y, pred, np.ones_like(y), anc)
    assert h["precision"] == pytest.approx(2 / 3) and h["recall"] == pytest.approx(2 / 3)
    assert (h["intersection"], h["n_pred"], h["n_true"]) == (2, 3, 3)


def test_hierarchical_empty_prediction_undefined_precision(tiny):
    h = M.hierarchical_prf(np.array([[0, 1, 0, 0]], bool), np.zeros((1, 4), bool),
                           np.ones((1, 4), bool), tiny.ancestor_matrix())
    # predicting nothing while positives exist: precision undefined, but F1 = 2TP/(2TP+FP+FN)
    # = 0.0 (arbiter: tester bug #2 — None here blocked promotion forever)
    assert h["precision"] is None and h["recall"] == 0.0 and h["f1"] == 0.0


def test_ece_hand_example():
    # 2 bins. bin0: p=.1,.2 (labels 0,1): conf .15 acc .5 -> .5*.35 = .175
    #         bin1: p=.8,.9 (labels 1,1): conf .85 acc 1  -> .5*.15 = .075   => ECE .25
    e = M.expected_calibration_error([.1, .2, .8, .9], [0, 1, 1, 1], n_bins=2)
    assert e["ece"] == pytest.approx(0.25) and e["n"] == 4 and len(e["bins"]) == 2
    # perfectly calibrated within bins -> 0 ; p = 1.0 lands in the last bin
    assert M.expected_calibration_error([0.0, 1.0], [0, 1], n_bins=10)["ece"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        M.expected_calibration_error([1.5], [1])


def test_multilabel_ece_pools_only_observed():
    y = np.array([[1, 0], [0, 0]], bool)
    s = np.array([[.9, .9], [.1, .9]])
    m = np.array([[1, 0], [1, 1]], bool)
    # pooled observed pairs: (.9,1) (.1,0) (.9,0), 1 bin: conf 1.9/3, acc 1/3 -> |1/3-1.9/3| = .3
    e = M.multilabel_ece(y, s, m, n_bins=1)
    assert e["ece"] == pytest.approx(0.3) and e["n"] == 3
    # restricting to column 0: (.9,1) (.1,0), 1 bin: conf .5 acc .5 -> 0
    assert M.multilabel_ece(y, s, m, cols=[0], n_bins=1)["ece"] == pytest.approx(0.0)


def test_shape_and_nan_checks():
    with pytest.raises(ValueError):
        M.per_class_ap(np.zeros((2, 2), bool), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        M.per_class_ap(np.array([[1], [0]], bool), np.array([[np.nan], [0.1]]))
