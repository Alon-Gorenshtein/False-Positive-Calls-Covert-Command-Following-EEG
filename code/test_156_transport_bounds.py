import importlib

tb = importlib.import_module("156_transport_bounds")






def test_flags():
    assert "spatial_filter_outside_folds" in tb.flags("mbmi", "or17_120_cspglobal_peak")
    assert "random_folds_on_block_design" in tb.flags("dubois", "c3_b_assessment_random_phase1")
    assert "random_folds_on_block_design" in tb.flags("dubois", "assessment_random_phase1")
    assert "random_folds_on_block_design" in tb.flags("edlow17", "permsecond_cvsecond_alt")
    assert tb.flags("dubois", "c3_b_assessment_blocked_phase1") == []
    assert tb.flags("dubois", "training_random_phase1") == []
    assert "selection_on_test_data" in tb.flags("mbmi", "or17_120_cspfold_peak")
    assert "selection_on_test_data" in tb.flags("mbvt", "vt3_mur20_seg8_max_gt23")
    assert tb.flags("mbvt", "vt3_mur20_seg8_median_gt23") == [] and tb.flags("mbmi", "or17_120_cspfold_mean57") == []
    assert tb.flags("dubois", "c3_b_assessment_blocked_phase1") == []








def test_false_share():
    s = tb.false_share(rate=0.05, prevalence=0.25)
    assert abs(s - 0.05 * 0.75 / (0.05 * 0.75 + 0.25)) < 1e-12


