"""The central dataset is data, so it gets tested like code."""
import gzip
import json
from pathlib import Path

import pytest

from freqai.record_contract import validate_information_record

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset/manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(),
                                reason="central dataset not built yet; run scripts/dataset_build.py")


def read_ids(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line)["id"] for line in stream if line.strip()]


def steps():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["steps"]


def test_manifest_describes_pinned_sources_and_its_own_files():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["record_contract"] == "declarative information text plus source, no question or answer field"
    assert manifest["sources"]["dataset"] == "wikimedia/wikipedia"
    assert len(manifest["sources"]["shards"]) == 4
    assert manifest["stats"]["source_articles_scanned"] == 569062
    for step in steps():
        path = ROOT / step["file"]
        assert path.exists(), step["file"]
        assert step["sha256"]
        assert step["records"] > 0


def test_expansion_steps_are_nested_supersets_in_a_single_location(tmp_path):
    previous = None
    for step in steps():
        ids = set(read_ids(ROOT / step["file"]))
        assert len(ids) == step["records"], "record ids must be unique inside one step"
        if previous is not None:
            assert previous < ids, "a later step must contain everything of the earlier one"
        previous = ids


def test_no_record_anywhere_carries_a_question_or_answer_field():
    checked = 0
    for step in steps():
        path = ROOT / step["file"]
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for position, line in enumerate(stream):
                if position % 97:
                    continue
                record = json.loads(line)
                validate_information_record({"text": record["text"],
                                             "prompt": record.get("prompt", "")})
                assert not str(record.get("prompt", "")).strip()
                assert record["text"].strip()
                checked += 1
    assert checked > 500


def test_evaluation_splits_never_entered_the_corpus():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    guard = manifest["leakage_guard"]
    assert guard["forbidden_texts"] > 1000
    assert guard["corpus_collisions"] == []
    for entry in guard["evaluation_files"]:
        assert (ROOT / "memory/evaluation" / entry["file"]).exists()


def test_first_step_is_the_corpus_that_is_actually_running():
    first = steps()[0]
    assert first.get("reproduces_productive_corpus") is True


def test_addressability_probes_only_use_declared_subject_and_copula_sentences():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from understanding_eval import addressability_probes
    records = [{"text": "Die Brillinge sind eine Gruppe kleiner Seevögel mit schwarzem Gefieder."},
               {"text": "Der Velaskörper rotiert langsam um seine Achse."},
               {"text": "Sie waren früher häufig, heute fehlen sie ganz."}]
    probes = addressability_probes(records)
    assert len(probes) == 1
    assert probes[0]["prompt"].startswith("Was ist ")
    assert probes[0]["reference"].startswith("Die Brillinge")
