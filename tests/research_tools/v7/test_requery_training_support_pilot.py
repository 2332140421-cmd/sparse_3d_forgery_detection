from research_tools.v7.periodic_requery_probe.requery_training_support_pilot import independent_label


def test_independent_real_label_does_not_depend_on_support():
    assert independent_label("real", 10.0, 1.0, []) == ("REAL_NEGATIVE", True, 0, "real role")


def test_fake_union_label_uses_window_boundaries():
    segments = [{"start_s": 10.0, "end_s": 11.0}, {"start_s": 11.0, "end_s": 12.0}]
    assert independent_label("fake", 10.0, 1.0, segments)[0:3] == ("FAKE_MANIPULATION", True, 1)
    assert independent_label("fake", 9.5, 1.0, segments)[0:3] == ("BOUNDARY_MIXED", False, None)
    assert independent_label("fake", 12.0, 1.0, segments)[0:3] == ("OUTSIDE_ANNOTATED_MANIPULATION", False, None)
