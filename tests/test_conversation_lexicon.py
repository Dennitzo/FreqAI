"""Version/provenance contracts for declarative lexical information."""
import hashlib
import json
from pathlib import Path
import re


FOLDER = Path(__file__).resolve().parents[1] / "memory/information"
PATH = FOLDER / "conversation_lexicon_v1.jsonl"
ROWS = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines()]
MANIFEST = json.loads(PATH.with_suffix(".provenance.json").read_text(encoding="utf-8"))


def test_lexicon_is_bound_to_its_versioned_manifest():
    assert MANIFEST["corpus_version"] == 1
    assert MANIFEST["sha256"] == hashlib.sha256(PATH.read_bytes()).hexdigest()
    assert MANIFEST["records"] == len(ROWS)
    assert MANIFEST["bytes"] == PATH.stat().st_size
    assert all(row["provenance"]["corpus_version"] == 1 for row in ROWS)


def test_existing_facts_remain_identical_to_the_prelexicon_version():
    assert MANIFEST["original_facts"]["unchanged"] is True
    assert hashlib.sha256((FOLDER / "conversation_facts.jsonl").read_bytes()).hexdigest() == MANIFEST["original_facts"]["sha256"]


def test_lexicon_has_information_records_without_qa_or_turn_fields():
    forbidden = {"prompt", "question", "answer", "messages", "role", "input", "output", "cue"}
    for row in ROWS:
        assert set(row) == {"id", "text", "source", "provenance"}
        assert not forbidden.intersection(row)
        assert not forbidden.intersection(row["provenance"])
        assert all(isinstance(row[key], str) and row[key].strip() for key in ("id", "text", "source"))


def test_fixed_inventory_covers_all_six_general_lexical_classes():
    expected = {"greeting", "farewell", "gratitude", "courtesy", "agreement", "negation"}
    assert 30 <= len(ROWS) <= 60
    assert {row["provenance"]["lexical_class"] for row in ROWS} == expected
    for category, expressions in MANIFEST["inventory_fixed_before_evaluation"].items():
        assert expressions == [row["provenance"]["expression"] for row in ROWS
                               if row["provenance"]["lexical_class"] == category]
    assert len({row["id"] for row in ROWS}) == len({row["text"] for row in ROWS}) == len(ROWS)


def test_word_and_phrase_metadata_match_the_declared_compiler_contract():
    for row in ROWS:
        provenance = row["provenance"]
        title, separator, body = row["text"].partition("\n\n")
        assert separator and title == provenance["expression"] == provenance["article_title"]
        assert 1 <= len(title.split()) <= 6
        assert provenance["form_kind"] == ("phrase" if " " in title else "word")
        assert body.endswith(".") and "?" not in body and "\n" not in body
        assert re.fullmatch(r"(?:Das Wort .+ ist ein .+wort|Die Wendung .+ ist eine .+formel)\.", body)


def test_authorship_is_transparent_and_all_text_hashes_match():
    for row in ROWS:
        provenance = row["provenance"]
        assert provenance["kind"] == "authored_lexical_information"
        assert provenance["author"] and provenance["created_at"] and provenance["scope"]
        assert provenance["source_qa_pairs_used"] is False
        assert provenance["evaluation_questions_used"] is False
        assert provenance["external_dataset"] is False
        assert provenance["text_sha256"] == hashlib.sha256(row["text"].encode("utf-8")).hexdigest()


def test_frozen_legacy_reuse_audit_is_read_only_and_found_no_complete_answer_reuse():
    audit = MANIFEST["legacy_text_overlap_audit"]
    assert audit["read_only"] is True and MANIFEST["database_changed"] is False
    assert audit["legacy_rows_compared"] >= 12000
    assert audit["matches"] == []
    assert MANIFEST["evaluation_files_read"] == []
    assert MANIFEST["editorial_review"]["contains_prompt_conditions_or_response_instructions"] is False
