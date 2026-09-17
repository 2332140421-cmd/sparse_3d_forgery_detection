"""Small contracts for the V7 data-flow/cache identity guards."""

from types import SimpleNamespace

from research_tools.v7.attention_pooling_pilot import runner as attention
from research_tools.v7.multi_order_sequence_probe import runner as multi
from research_tools.v7.pair_trajectory_probe import runner as pair
from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.source128_extension import runner as source128


def _row(offset: float = 0.5) -> dict[str, object]:
    return {
        "source_id": "S",
        "role": "fake",
        "pair_id": "P",
        "parent_id": "S::fake::MANIP50",
        "window_id": "S::fake::MANIP50::b05",
        "offset_s": offset,
    }


def test_r_sequence_identity_rejects_wrong_video_or_cohort() -> None:
    sequence = SimpleNamespace(
        source_video_id="OTHER::fake",
        num_tracks=289,
        lineage={"source_id": "S", "role": "fake", "pair_id": "P", "parent_id": "S::fake::MANIP50", "window_id": "S::fake::MANIP50::b05"},
        provenance={"query_cohort": "R"},
    )
    assert not pair._sequence_identity_matches_row(sequence, _row())
    sequence.source_video_id = "S::fake"
    sequence.provenance = {"query_cohort": "O"}
    assert not pair._sequence_identity_matches_row(sequence, _row())


def test_model_identity_guards_reject_same_key_different_inputs() -> None:
    expected = {"condition": "SET_A", "held_out_source": "S", "training_window_ids_sha256": "a"}
    record = {"input_identity": dict(expected)}
    assert multi._model_record_identity_matches(record, expected)
    changed = dict(expected, training_window_ids_sha256="b")
    assert not multi._model_record_identity_matches(record, changed)
    assert not attention._model_record_identity_matches(record, dict(expected, condition="MEAN_BASELINE"))
    assert not source128._model_record_identity_matches(record, dict(expected, source_count=128))


def test_periodic_model_identity_requires_input_fingerprint() -> None:
    expected = {"condition": "R_SET", "held_out_source": "S", "support_sha256": "support"}
    assert periodic._model_record_identity_matches({"input_identity": expected}, expected)
    assert not periodic._model_record_identity_matches({"condition": "R_SET"}, expected)


def test_pair_identity_helper_does_not_use_pair_name_as_video_identity() -> None:
    sequence = SimpleNamespace(
        source_video_id="S::real",
        num_tracks=289,
        lineage={"source_id": "S", "role": "fake", "pair_id": "P", "parent_id": "S::fake::MANIP50", "window_id": "S::fake::MANIP50::b05"},
        provenance={"query_cohort": "R"},
    )
    assert not pair._sequence_identity_matches_row(sequence, _row())
