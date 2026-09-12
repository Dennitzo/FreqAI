"""Validate unpaired source structure and authorship, not desired chat replies.

These checks establish structural absence of QA fields/turns. The semantic
absence of disguised response rules is a separate editorial review recorded in
the corpus manifest; no regular expression can prove it for arbitrary prose.
"""
import hashlib
import json
from pathlib import Path
import re


CORPUS = Path(__file__).resolve().parents[1] / "memory/information/conversation_facts.jsonl"
ROWS = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]


def dictionary_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key.casefold()
            yield from dictionary_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionary_keys(child)


def test_unpaired_document_schema_contains_only_information_and_provenance():
    assert 20 <= len(ROWS) <= 50
    for row in ROWS:
        assert set(row) == {"id", "text", "source", "provenance"}
        assert all(isinstance(row[key], str) and row[key].strip() for key in ("id", "text", "source"))
        assert isinstance(row["provenance"], dict)


def test_no_question_answer_turn_or_prompt_fields_even_inside_metadata():
    forbidden = {"prompt", "question", "answer", "input", "output", "cue", "messages", "role", "turns"}
    assert not forbidden.intersection(dictionary_keys(ROWS))


def test_records_have_unique_addresses_and_distinct_information_texts():
    assert len({row["id"] for row in ROWS}) == len(ROWS)
    assert len({row["text"] for row in ROWS}) == len(ROWS)


def test_complete_titled_prose_has_no_embedded_chat_transcript():
    for row in ROWS:
        title, separator, body = row["text"].partition("\n\n")
        assert separator and title and body.endswith(".")
        assert row["provenance"]["article_title"] == title
        assert "?" not in body
        assert not re.search(r"(?im)^(?:Du|FreqAI|User|Assistant|Benutzer|Assistent)\s*:", body)
        assert not re.search(r"<\|(?:assistant|user|im_start|im_end)\|>", body)


def test_authorship_and_independent_source_scope_are_explicit():
    for row in ROWS:
        provenance = row["provenance"]
        assert provenance["kind"] == "authored_information"
        assert provenance["creation_method"] == "original_assistant_authored_declarative_prose"
        assert provenance["author"] and provenance["created_at"] and provenance["scope"]
        assert provenance["source_qa_pairs_used"] is False
        assert provenance["evaluation_questions_used"] is False
        assert provenance["external_dataset"] is False


def test_each_text_has_a_verifiable_content_hash():
    for row in ROWS:
        assert row["provenance"]["text_sha256"] == hashlib.sha256(row["text"].encode("utf-8")).hexdigest()


def test_no_explicit_conditional_reply_instruction_is_stored_as_information():
    # A limited lint for accidental rule serialization, not a semantic proof.
    for row in ROWS:
        assert not re.search(r"\b(?:wenn|falls)\b[^.]{0,250}\b(?:antworte|sage|erwidere|gib aus)\b",
                             row["text"], re.IGNORECASE)
