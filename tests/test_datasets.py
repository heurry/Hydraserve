from __future__ import annotations

import gzip
import json
from pathlib import Path
import zipfile

import pytest

from hydraserve.benchmark import DatasetCatalog, DatasetFormatError, iter_dataset


def _jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_catalog_and_jsonl_adapters(tmp_path: Path) -> None:
    _jsonl(tmp_path / "gsm8k.jsonl", [{"question": "1+1?", "answer": "2"}])
    _jsonl(
        tmp_path / "wikitext-103-test.jsonl",
        [{"text": ""}, {"text": " = Title =\n"}],
    )
    assert set(DatasetCatalog(tmp_path).available()) == {"gsm8k", "wikitext"}
    gsm = list(iter_dataset(tmp_path, "gsm8k"))
    wiki = list(iter_dataset(tmp_path, "wikitext"))
    assert (gsm[0].prompt, gsm[0].reference) == ("1+1?", "2")
    assert [sample.prompt for sample in wiki] == [" = Title =\n"]


def test_humaneval_detects_gzip_magic_with_jsonl_suffix(tmp_path: Path) -> None:
    record = {
        "task_id": "HumanEval/0",
        "prompt": "def f():\n",
        "canonical_solution": "    return 1\n",
        "entry_point": "f",
        "test": "assert f() == 1",
    }
    with gzip.open(tmp_path / "humaneval.jsonl", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    sample = next(iter_dataset(tmp_path, "humaneval"))
    assert sample.sample_id == "HumanEval/0"
    assert sample.metadata["entry_point"] == "f"


def test_sharegpt_streams_top_level_array_across_tiny_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        {
            "id": "a",
            "conversations": [
                {"from": "human", "value": "hello"},
                {"from": "gpt", "value": "hi"},
            ],
        },
        {"id": "skip", "conversations": [{"from": "gpt", "value": "no user"}]},
        {
            "id": "b",
            "conversations": [
                {"from": "user", "value": "x" * 200},
                {"from": "assistant", "value": "done"},
            ],
        },
    ]
    (tmp_path / "sharegpt.json").write_text(json.dumps(records), encoding="utf-8")
    import hydraserve.benchmark.datasets as datasets

    original = datasets._iter_json_array
    monkeypatch.setattr(
        datasets,
        "_iter_json_array",
        lambda path: original(path, chunk_size=17),
    )
    samples = list(iter_dataset(tmp_path, "sharegpt"))
    assert [sample.sample_id for sample in samples] == ["a", "b"]
    assert samples[0].reference == "hi"
    assert len(samples[1].prompt) == 200


def test_longbench_reads_named_member_without_extracting(tmp_path: Path) -> None:
    archive_path = tmp_path / "longbench.zip"
    record = {
        "input": "question",
        "context": "long context",
        "answers": ["answer"],
        "length": 12,
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/toy.jsonl", json.dumps(record) + "\n")
    catalog = DatasetCatalog(tmp_path)
    assert catalog.longbench_subsets() == ("toy",)
    sample = next(iter_dataset(tmp_path, "longbench", subset="toy"))
    assert sample.dataset == "longbench/toy"
    assert sample.prompt == "long context\n\nquestion"
    assert sample.reference == ["answer"]
    assert sample.metadata["length"] == 12


def test_limits_and_format_errors(tmp_path: Path) -> None:
    _jsonl(
        tmp_path / "gsm8k.jsonl",
        [{"question": str(index), "answer": str(index)} for index in range(3)],
    )
    assert len(list(iter_dataset(tmp_path, "gsm8k", limit=2))) == 2
    assert list(iter_dataset(tmp_path, "gsm8k", limit=0)) == []
    with pytest.raises(ValueError, match="non-negative"):
        list(iter_dataset(tmp_path, "gsm8k", limit=-1))
    (tmp_path / "sharegpt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="top-level JSON array"):
        list(iter_dataset(tmp_path, "sharegpt"))
