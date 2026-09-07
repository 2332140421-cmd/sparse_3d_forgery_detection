from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from research_tools.v7.datasets.genvidbench.audit_core_pilot import REVIEW_FIELDS, _review_rows
from research_tools.v7.datasets.genvidbench.finalize_core_pilot import run as finalize
from research_tools.v7.datasets.genvidbench.metadata import (
    LabeledVideo, SemanticRecord, build_pilot_selection, parse_label_file,
    parse_semantic_file, select_real_candidates,
)

def _record(ordinal: str, identity: str, obj: str = "Vehicles", action: str = "Active Engagement") -> SemanticRecord:
    return SemanticRecord("HD-VG-130M", ordinal, identity, obj, action, "Others", f"caption {ordinal}")

def test_semantic_parser_uses_official_taxonomy(tmp_path: Path) -> None:
    classes = tmp_path / "classes_list.txt"
    classes.write_text('Object_dict = {"Vehicles": 7}\nAction_dict = {"Active Engagement": 3}\nLocation_dict = {"Others": 0}\n')
    semantic = tmp_path / "semantic.txt"
    semantic.write_text("00001|||abc|||7_3_0|||a car\n")
    assert parse_semantic_file(semantic, "HD-VG-130M", classes)[0].object_label == "Vehicles"

def test_label_parser_preserves_space_containing_paths(tmp_path: Path) -> None:
    labels = tmp_path / "labels.txt"
    labels.write_text("Pair2/cogvideo/00001___a car on a road.mp4 1\nPair2/hd_vg_130m/00001___abc.1_0.mp4 0\n")
    rows = parse_label_file(labels)
    assert rows[0].ordinal == "00001" and rows[0].source_identity is None
    assert rows[1].source_identity == "abc"

def test_selection_is_deterministic_and_sorted() -> None:
    records = (_record("00002", "b"), _record("00001", "a"))
    videos = (LabeledVideo("Pair2/hd_vg_130m/00002___b.0_0.mp4", "hd_vg_130m", 0, "00002", "b"),
              LabeledVideo("Pair2/hd_vg_130m/00001___a.0_0.mp4", "hd_vg_130m", 0, "00001", "a"))
    selected = select_real_candidates(records, videos, role="test_real_source", real_source="HD-VG-130M")
    assert [row.ordinal for row in selected] == ["00001", "00002"]

def test_static_and_nonvehicle_are_not_auto_core_candidates() -> None:
    records = (_record("1", "a", action="Static Postures"), _record("2", "b", obj="People"))
    videos = tuple(LabeledVideo(f"x/{i}.mp4", "x", 0, str(i), ident) for i, ident in [(1, "a"), (2, "b")])
    assert select_real_candidates(records, videos, role="train_real", real_source="Vript") == ()

def test_pilot_selection_keeps_train_real_and_test_fake_separate() -> None:
    vript = (_record("00001", "train"),)
    hdvg = (_record("00001", "test"),)
    pair1 = (LabeledVideo("Pair1/vript/train.mp4", "vript", 0, "00001", "train"),)
    pair2 = (LabeledVideo("Pair2/hd_vg_130m/00001___test.0_0.mp4", "hd_vg_130m", 0, "00001", "test"),
             LabeledVideo("Pair2/musev/00001___caption.mp4", "musev", 1, "00001", None))
    selected = build_pilot_selection(pair1_labels=pair1, pair2_labels=pair2, vript_records=vript, hdvg_records=hdvg)
    assert len(selected["train_real"]) == 1 and len(selected["test_fake_paired"]) == 1
    assert selected["test_fake_paired"][0]["source_id"] == selected["test_real_source"][0]["source_id"]
    assert selected["test_fake_paired"][0]["pair_lineage"]

def test_review_schema_has_no_model_or_fake_decision_fields() -> None:
    row = {"source_id": "x", "role": "train_real", "real_source": "Vript", "relative_path": "x.mp4", "semantic_object": "Vehicles", "semantic_action": "Active Engagement", "semantic_location": "Others"}
    rendered = _review_rows({"train_real": [row], "test_real_source": [], "test_fake_paired": []})[0]
    assert tuple(rendered) == REVIEW_FIELDS
    assert not {"label", "generator", "score", "model_output"} & set(rendered)

def _write_manifest_root(root: Path, decisions: list[str]) -> None:
    (root / "real_review").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "paired_test").mkdir()
    fields = list(REVIEW_FIELDS)
    with (root / "real_review" / "review_core.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, decision in enumerate(decisions):
            writer.writerow({field: (f"id-{i}" if field == "source_id" else decision if field == "video_decision" else "") for field in fields})
    train = [{"source_id": "id-0", "role": "train_real"}]
    test = [{"source_id": "id-1", "role": "test_real_source"}]
    fake = [{"source_id": "id-1", "generator": "musev", "fake_path": "fake.mp4"}]
    for name, value in [("train_real_pilot.json", train), ("test_real_source_pilot.json", test)]:
        (root / "manifests" / name).write_text(json.dumps(value))
    (root / "paired_test" / "paired_test_manifest.json").write_text(json.dumps(fake))

def test_finalizer_excludes_uncertain_rows(tmp_path: Path) -> None:
    _write_manifest_root(tmp_path, ["IN_DOMAIN", "UNCERTAIN"])
    result = finalize(tmp_path)
    assert [row["source_id"] for row in result["train_real"]] == ["id-0"]
    assert result["test_real_source"] == [] and result["test_fake_paired"] == []

def test_finalizer_rejects_incomplete_review(tmp_path: Path) -> None:
    _write_manifest_root(tmp_path, ["", "OUT_OF_DOMAIN"])
    with pytest.raises(ValueError, match="incomplete"):
        finalize(tmp_path)


def test_pilot_manifest_has_unique_source_ids_and_real_train_only() -> None:
    vript = (_record("00001", "train"), _record("00002", "train2"))
    hdvg = (_record("00001", "test"),)
    pair1 = (LabeledVideo("Pair1/vript/train.mp4", "vript", 0, "00001", "train"),
             LabeledVideo("Pair1/vript/train2.mp4", "vript", 0, "00002", "train2"),
             LabeledVideo("Pair1/ms/fake.mp4", "ms", 1, None, None))
    pair2 = (LabeledVideo("Pair2/hd_vg_130m/00001___test.0_0.mp4", "hd_vg_130m", 0, "00001", "test"),)
    selected = build_pilot_selection(pair1_labels=pair1, pair2_labels=pair2, vript_records=vript, hdvg_records=hdvg)
    ids = [row["source_id"] for row in selected["train_real"]]
    assert len(ids) == len(set(ids))
    assert all(row["role"] == "train_real" and row["real_source"] == "Vript" for row in selected["train_real"])

def test_fake_pair_manifest_contains_required_lineage_fields() -> None:
    vript = (_record("00001", "train"),)
    hdvg = (_record("00001", "test"),)
    pair1 = (LabeledVideo("Pair1/vript/train.mp4", "vript", 0, "00001", "train"),)
    pair2 = (LabeledVideo("Pair2/hd_vg_130m/00001___test.0_0.mp4", "hd_vg_130m", 0, "00001", "test"),
             LabeledVideo("Pair2/svd/00001___caption.mp4", "svd", 1, "00001", None))
    selected = build_pilot_selection(pair1_labels=pair1, pair2_labels=pair2, vript_records=vript, hdvg_records=hdvg)
    fake = selected["test_fake_paired"][0]
    assert {"source_id", "real_video_path", "source_identity", "generator", "fake_path", "pair_lineage"} <= set(fake)
    assert fake["generator"] == "svd"
