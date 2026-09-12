"""Validate source records before dropping metadata into the Document schema."""
from __future__ import annotations


QA_RECORD_FIELDS = frozenset({
    "question", "answer", "response", "instruction", "messages", "conversations",
    "turns", "user", "assistant", "pairs", "pair", "dialogue", "dialog", "cue",
    "input", "output",
})


def validate_information_record(record, *, allow_legacy_pairs=False):
    """Reject unsupported QA schemas without silently discarding their fields.

    The explicit archive path can preserve the historical prompt/text/source
    schema. Other dialogue structures are rejected there too, since the archive
    cannot retain them losslessly. Nested provenance remains source metadata.
    """
    if not isinstance(record, dict):
        raise ValueError("Jeder Informationseintrag muss ein JSON-Objekt sein.")
    forbidden = sorted(str(key) for key in record if str(key).casefold() in QA_RECORD_FIELDS)
    if forbidden:
        raise ValueError("Frage-Antwort- oder Dialogfelder sind nicht zulässig: "+", ".join(forbidden))
    prompt = record.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("Ein historisches prompt-Feld muss Text sein.")
    if prompt.strip() and not allow_legacy_pairs:
        raise ValueError("Nur Informationstexte ohne Frage-Antwort-Paare sind zulässig; Altpaare explizit archivieren.")
