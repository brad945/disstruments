from disstruments.ml.eval.triangle import arm_of, node_list
from disstruments.ml.taxonomy import get_taxonomy


def test_node_lists_partition_the_openmic_classes():
    tax = get_taxonomy()
    t9, u11 = node_list("trained9", tax), node_list("untrained11", tax)
    assert len(t9) == 9 and len(u11) == 11 and not set(t9) & set(u11)
    assert {"guitar", "keys.piano", "drums", "bass"} <= set(t9)   # the M3 trained-9


def test_arm_names_split_seed():
    assert arm_of("c3_cnn_n254_s2") == ("c3_cnn_n254", 2)
    assert arm_of("c1_mert_s0") == ("c1_mert", 0)
    assert arm_of("reference") == ("reference", None)
