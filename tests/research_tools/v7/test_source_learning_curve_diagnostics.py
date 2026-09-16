"""Focused checks for source-learning-curve diagnostic accounting."""

from research_tools.v7.source_learning_curve.diagnose import _source_bootstrap


def test_source_bootstrap_uses_source_values_and_fixed_protocol() -> None:
    result = _source_bootstrap({"b": 0.75, "a": 0.25})
    assert result["source_count"] == 2
    assert result["mean"] == 0.5
    assert result["sources"] == ["a", "b"]
    assert result["seed"] == 20260909
    assert result["replicates"] == 10_000
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]
