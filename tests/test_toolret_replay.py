import json

import pytest

from experiments.retrieval.toolret_replay import select_training_rows, write_outputs


def test_replay_excludes_reserved_and_explicit_evaluation_ids():
    rows = [dict(qid=f"{source}{i}", source=source, pool="train") for source in ("a", "b") for i in range(5)]
    rows += [dict(qid="a0", source="a", pool="eval"), dict(qid="b0", source="b", pool="gatefit")]
    selected, report = select_training_rows(rows, 4, exclusions={"external_eval": {"a1", "b1"}})
    assert len(selected) == 4
    assert all(row["qid"] not in {"a0", "b0", "a1", "b1"} for row in selected)
    assert report["per_source"] == {"a": 2, "b": 2}
    assert report["excluded_among_eligible_train_rows"]["union"] == 4
    assert selected == select_training_rows(rows, 4, exclusions={"external_eval": {"a1", "b1"}})[0]


def test_replay_accepts_an_explicit_training_filter():
    rows = [dict(qid=str(i), source="a", pool="train") for i in range(4)]
    selected, _ = select_training_rows(rows, 4, include={"1", "3"})
    assert {row["qid"] for row in selected} == {"1", "3"}
    with pytest.raises(ValueError, match="excluded"):
        select_training_rows(rows, 4, exclusions={"all": {"0", "1", "2", "3"}})


def test_extra_filter_cannot_replace_reserved_pool_exclusions():
    rows = [dict(qid=str(i), source="a", pool="train") for i in range(3)]
    rows.append(dict(qid="0", source="a", pool="eval"))
    selected, _ = select_training_rows(rows, 3, exclusions={"reserved_pools": set()})
    assert {row["qid"] for row in selected} == {"1", "2"}


def test_source_with_no_remaining_queries_does_not_take_sample_quota():
    rows = [dict(qid="a0", source="a", pool="train"), dict(qid="b0", source="b", pool="train")]
    selected, _ = select_training_rows(rows, 1, exclusions={"extra": {"b0"}})
    assert [row["qid"] for row in selected] == ["a0"]


def test_report_does_not_overwrite_extensionless_output(tmp_path):
    output = tmp_path / "nested" / "replay"
    report = write_outputs(output, [{"qid": "sample"}], {"n": 1})
    assert report != output
    assert [json.loads(line) for line in output.read_text().splitlines()] == [{"qid": "sample"}]
    assert json.loads(report.read_text()) == {"n": 1}
