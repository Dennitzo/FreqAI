"""Source filters protect role pairing, held-out split boundaries and answer spans."""
import gzip
import json

from experiments.import_public_corpus import (
    DOWNLOADS, answer_sentences, build_oasst, download_verified,
    normalize, oasst_message_rejection, parent_chain, is_plain_prose, has_foreign_persona,
)


def message(id_, parent=None, role="prompter", text="Wie kann ich meinen Tag entspannt beginnen?"):
    return {
        "message_id": id_, "parent_id": parent, "message_tree_id": "root",
        "role": role, "text": text, "lang": "de", "review_result": True,
        "deleted": False, "synthetic": False, "tree_state": "ready_for_export",
        "labels": {"quality": {"value": 0.8}},
    }


def test_gold_fragment_expands_to_its_complete_original_sentence():
    context = "Katzen\n\nDie Katze schläft auf dem warmen Sofa. Danach läuft sie hinaus."
    gold = "warmen Sofa"
    result = answer_sentences(context, context.index(gold), gold)
    assert result[0] == "Die Katze schläft auf dem warmen Sofa."
    assert normalize(context[result[1]:result[2]]) == result[0]


def test_backward_reference_preserves_previous_sentence():
    context = "Recht\nDas deutsche Recht hat viele Regeln. Diese Regeln werden sorgfältig geprüft."
    gold = "sorgfältig geprüft"
    assert answer_sentences(context, context.index(gold), gold)[0] == "Das deutsche Recht hat viele Regeln. Diese Regeln werden sorgfältig geprüft."


def test_wrong_answer_offset_cannot_silently_import_unrelated_text():
    assert answer_sentences("Berlin ist eine große Stadt.", 0, "große Stadt") is None


def test_abbreviation_and_ordinal_do_not_cut_sentence():
    context = "Geschichte\nDr. Müller lebte im 19. Jahrhundert in Berlin. Er schrieb Bücher."
    gold = "19. Jahrhundert"
    assert answer_sentences(context, context.index(gold), gold)[0] == "Dr. Müller lebte im 19. Jahrhundert in Berlin."


def test_span_crossing_two_sentences_preserves_both():
    context = "Wetter\nDer Himmel ist blau. Heute scheint die Sonne. Morgen regnet es."
    gold = "blau. Heute scheint"
    assert answer_sentences(context, context.index(gold), gold)[0] == "Der Himmel ist blau. Heute scheint die Sonne."


def test_unfinished_text_is_not_presented_as_complete_sentence():
    context = "Natur\nDer Baum hat viele grüne Blätter"
    assert answer_sentences(context, context.index("grüne"), "grüne Blätter") is None


def test_only_root_pairs_are_active_and_followups_retain_role_chain(tmp_path):
    rows = [message("root"), message("a1", "root", "assistant", "Du kannst den Morgen mit einer ruhigen Pause beginnen."),
            message("p2", "a1", text="Was kann ich dabei genau machen?"),
            message("a2", "p2", "assistant", "Du kannst dich gemütlich hinsetzen und ein wenig ausruhen.")]
    path = tmp_path / "source.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    active, inactive, _ = build_oasst(path)
    assert len(active) == len(inactive) == 1
    assert active[0]["prompt"] == rows[0]["text"]
    assert inactive[0]["prompt"] == rows[2]["text"]
    assert [row["role"] for row in inactive[0]["provenance"]["parent_chain"]] == ["prompter", "assistant", "prompter"]
    assert "inactive" in inactive[0]["import_status"]


def test_synthetic_and_deleted_messages_do_not_become_human_examples():
    synthetic = message("root")
    synthetic["synthetic"] = True
    assert oasst_message_rejection(synthetic) == "synthetic_or_unknown"
    synthetic["synthetic"] = False
    synthetic["deleted"] = True
    assert oasst_message_rejection(synthetic) == "not_accepted_or_deleted"


def test_parent_chain_rejects_wrong_roles():
    import pytest
    parent = message("root", role="assistant")
    child = message("child", "root", "assistant")
    with pytest.raises(ValueError, match="Role/tree mismatch"):
        parent_chain(child, {"root": parent})


def test_download_checksum_is_enforced_for_existing_cache(tmp_path):
    import pytest
    path = tmp_path / "cached"
    path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_verified({"path": path, "sha256": "0" * 64}, offline=True)


def test_only_official_train_split_is_declared_as_download_source():
    assert "/train/" in DOWNLOADS["germanquad"]["url"]
    assert "/test/" not in DOWNLOADS["germanquad"]["url"]


def test_normalization_does_not_paraphrase_or_invent_text():
    assert normalize("  Scho\u0308n!\n\tWie geht es dir? ") == "Schön! Wie geht es dir?"


def test_unfenced_pseudocode_and_unpunctuated_lists_are_not_dialogue_prose():
    code = message("root", text="Für die Ausgabe gilt n := 1 und dann x := n + 2.")
    assert oasst_message_rejection(code) == "code_or_external_reference"
    assert not is_plain_prose("1. Du kannst den ersten Knopf anklicken\nNun sollte das Fenster erscheinen.")
    assert not is_plain_prose("Ein schöner Song\nEin weiteres Lied\nViel Spaß damit!")
    assert is_plain_prose("Du kannst dich in Ruhe hinsetzen.\n\nDanach kannst du weiterarbeiten.")


def test_foreign_personas_and_roleplay_are_filtered_without_removing_ai_facts():
    assert has_foreign_persona("Du bist ein freundlicher Detektiv.", "Ich kann dir gerne helfen.")
    assert has_foreign_persona("Wie heißt du?", "Mein Name ist Hubert und ich bin dein Assistent.")
    assert has_foreign_persona("Wer bist du?", "Ich bin OpenAssistant und wurde trainiert.")
    assert has_foreign_persona("Wer bist du?", "Ich bin Sylvia, dein freundlicher Assistent.")
    assert not has_foreign_persona("Was ist ein Sprachmodell?", "Ein Sprachmodell verarbeitet und erzeugt Sprache.")
    assert not has_foreign_persona("Kannst du helfen?", "Ich bin gerne bereit, dir beim Planen zu helfen.")


def test_wiki_emphasis_removed_without_changing_gold_source_offsets():
    context = "Glas\nDie Maschine hat eine sogenannte ''Dannerpfeife'' zur Herstellung der Röhren."
    gold = "Dannerpfeife"
    text, start, end = answer_sentences(context, context.index(gold), gold)
    assert "''" not in text
    assert "''Dannerpfeife''" in context[start:end]
    assert gold in text
