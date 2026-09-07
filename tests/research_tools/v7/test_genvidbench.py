from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from research_tools.v7.datasets.genvidbench.audit_core_pilot import REVIEW_FIELDS, _review_rows
from research_tools.v7.datasets.genvidbench.finalize_core_pilot import run as finalize
from research_tools.v7.datasets.genvidbench.materialize_core_pilot import run as materialize, uniform_frame_indices
from research_tools.v7.datasets.genvidbench.resolve_real_pilot_archives import (
    build_plan, extract_requested_members, resolve_targets,
)
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


def _archive_fixture(tmp_path: Path, *, duplicate: bool = False) -> tuple[Path, Path, dict[str, object]]:
    metadata = tmp_path / "metadata"; metadata.mkdir()
    output = tmp_path / "output"; (output / "manifests").mkdir(parents=True)
    (metadata / "Pair1_verify.txt").write_text("GenVidBench/vript/a/a.mp4 0\n" + ("GenVidBench/vript/b/a.mp4 0\n" if duplicate else ""))
    (metadata / "HD_VG_130M_verify.txt").write_text("GenVidBench/hd_vg_130m/00001___hd 0\n")
    (output / "manifests" / "train_real_pilot.json").write_text(json.dumps([{"source_id": "Vript:1:a", "role": "train_real", "real_source": "Vript", "relative_path": "Pair1/vript/a.mp4", "source_identity": "a"}]))
    (output / "manifests" / "test_real_source_pilot.json").write_text(json.dumps([{"source_id": "HD:1:hd", "role": "test_real_source", "real_source": "HD-VG-130M", "relative_path": "Pair2/hd_vg_130m/00001___hd.1_0.mp4", "source_identity": "hd"}]))
    inventory = {"files": [
        {"path": "GenVidBench/Pair1/vript.rar", "size": 49_190_751_379, "oid": "vript"},
        *[{"path": f"GenVidBench/Pair2/hd_vg_130m.7z.00{i}", "size": 21_474_836_480, "oid": str(i)} for i in range(1, 4)],
        {"path": "GenVidBench/Pair2/hd_vg_130m.7z.004", "size": 15_594_648_596, "oid": "4"},
    ]}
    return metadata, output, inventory

def test_resolver_preserves_frozen_target_identity_and_is_deterministic(tmp_path: Path) -> None:
    metadata, output, inventory = _archive_fixture(tmp_path)
    first = resolve_targets(metadata, output, inventory)
    second = resolve_targets(metadata, output, inventory)
    assert first == second
    assert [row["source_id"] for row in first] == ["Vript:1:a", "HD:1:hd"]
    assert first[0]["archive_member"] == "vript/a/a.mp4"

def test_ambiguous_official_member_mapping_is_rejected(tmp_path: Path) -> None:
    metadata, output, inventory = _archive_fixture(tmp_path, duplicate=True)
    rows = resolve_targets(metadata, output, inventory)
    assert rows[0]["status"] == "AMBIGUOUS"
    assert rows[0]["archive_member"] is None

def test_unresolved_7z_member_is_not_fuzzy_selected(tmp_path: Path) -> None:
    metadata, output, inventory = _archive_fixture(tmp_path)
    rows = resolve_targets(metadata, output, inventory)
    assert rows[1]["status"] == "UNRESOLVED"
    assert rows[1]["archive_member"] is None

def test_size_gate_blocks_minimum_tranche(tmp_path: Path) -> None:
    metadata, output, inventory = _archive_fixture(tmp_path)
    rows = resolve_targets(metadata, output, inventory)
    plan = build_plan(inventory, rows, metadata_root=metadata, output_root=output)
    assert plan["minimum_tranche"]["train_real"] == 1
    assert plan["decision"] == "SELECTIVE_MATERIALIZATION_TOO_EXPENSIVE"
    assert plan["actual_archive_download_bytes"] == 0

def test_resolver_targets_contain_no_fake_rows(tmp_path: Path) -> None:
    metadata, output, inventory = _archive_fixture(tmp_path)
    rows = resolve_targets(metadata, output, inventory)
    assert all(row["role"] in {"train_real", "test_real_source"} for row in rows)
    assert all(row["real_source"] in {"Vript", "HD-VG-130M"} for row in rows)

def test_resolution_sha_lineage_starts_unmaterialized(tmp_path: Path) -> None:
    metadata, output, inventory = _archive_fixture(tmp_path)
    rows = resolve_targets(metadata, output, inventory)
    assert all(row["video_sha256"] is None for row in rows)


def test_archive_extraction_writes_only_requested_members(tmp_path: Path) -> None:
    import tarfile
    archive = tmp_path / "sample.tar"
    with tarfile.open(archive, "w") as handle:
        for name, content in (("keep.mp4", b"keep"), ("do_not_extract.mp4", b"secret")):
            source = tmp_path / name; source.write_bytes(content); handle.add(source, arcname=name)
    out = tmp_path / "out"
    extracted = extract_requested_members(archive, ["keep.mp4"], out)
    assert [path.name for path in extracted] == ["keep.mp4"]
    assert (out / "keep.mp4").read_bytes() == b"keep"
    assert not (out / "do_not_extract.mp4").exists()

def test_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        extract_requested_members(tmp_path / "missing.tar", ["../escape.mp4"], tmp_path / "out")


def test_contact_sheet_sampling_spans_full_timeline() -> None:
    positions = uniform_frame_indices(101)
    assert positions[0] == 0 and positions[-1] == 100
    assert positions == sorted(set(positions))

def test_materializer_distinguishes_access_failure_and_keeps_review_blank(tmp_path: Path) -> None:
    review = tmp_path / "real_review"; review.mkdir()
    (tmp_path / "manifests").mkdir()
    (tmp_path / "paired_test").mkdir()
    with (review / "review_core.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS); writer.writeheader()
        writer.writerow({field: "id" if field == "source_id" else "train_real" if field == "role" else "" for field in REVIEW_FIELDS})
    original = (review / "review_core.csv").read_text()
    result = materialize(tmp_path / "missing-media", tmp_path)
    assert result["decode_success"] == 0 and result["decode_failure"] == 1
    output = list(csv.DictReader((review / "review_core_materialized.csv").open()))
    assert output[0]["video_decision"] == output[0]["video_reason_code"] == output[0]["notes"] == ""
    assert (review / "review_core.csv").read_text() == original
