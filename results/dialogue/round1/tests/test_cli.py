"""CLI commands use one append-only database and explicit NPZ exchange."""

import json
from pathlib import Path

import pytest

from freqai import cli
from freqai.memory import Document, WaveMemory
from freqai.store import MemoryStore


def jsonl(path, records):
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                    encoding="utf-8")
    return str(path)


def test_import_and_build_append_dedupe_and_do_not_replace(tmp_path, capsys):
    memory = tmp_path / "central.sqlite3"
    first = jsonl(tmp_path / "first.jsonl", [{"id": "one", "text": "Erste Information.", "source": "Buch"}])
    second = jsonl(tmp_path / "second.jsonl", [{"id": "two", "text": "Zweite Information.", "source": "Buch"}])
    cli.main(["import", first, "--memory", str(memory)])
    assert json.loads(capsys.readouterr().out)["added"] == 1
    cli.main(["build", second, "--memory", str(memory)])
    output = capsys.readouterr()
    assert "niemals ersetzt" in output.err
    assert json.loads(output.out)["added"] == 1
    cli.main(["import", first, "--memory", str(memory)])
    assert json.loads(capsys.readouterr().out)["existing"] == 1
    assert [document.id for document in MemoryStore(memory).snapshot()[1]] == ["one", "two"]


def test_cli_import_reports_conflict_without_partial_commit(tmp_path, capsys):
    path = tmp_path / "central.sqlite3"
    original = Document("stable", "Dieser Eintrag bleibt.")
    MemoryStore(path).append_documents([original])
    batch = jsonl(tmp_path / "conflict.jsonl", [
        {"id": "new", "text": "Nur Teil der fehlgeschlagenen Transaktion."},
        {"id": "stable", "text": "Würde überschreiben."},
    ])
    with pytest.raises(SystemExit) as error:
        cli.main(["import", batch, "--memory", str(path)])
    assert error.value.code == 2 and "ID conflict" in capsys.readouterr().err
    assert MemoryStore(path).snapshot() == (1, [original])


def test_npz_is_explicit_import_export_and_original_stays_unchanged(tmp_path, capsys):
    original = tmp_path / "original.npz"
    document = Document("npz-document", "Verlustfreie Rückrechnung: Grüß dich! 🌊", "Altbestand")
    WaveMemory([document], dimensions=64).save(original)
    original_bytes = original.read_bytes()
    database = tmp_path / "memory.sqlite3"
    cli.main(["import", str(original), "--memory", str(database)])
    assert json.loads(capsys.readouterr().out)["added"] == 1
    assert original.read_bytes() == original_bytes
    exported = tmp_path / "exported.npz"
    cli.main(["export", "--memory", str(database), "--output", str(exported)])
    capsys.readouterr()
    assert WaveMemory.load(exported).documents == [document]
    with pytest.raises(SystemExit) as error:
        cli.main(["ask", "Rückrechnung", "--memory", str(original)])
    assert error.value.code == 2
    assert "Austauschformat" in capsys.readouterr().err
    assert original.read_bytes() == original_bytes


def test_ask_infer_and_serve_use_same_default_database_across_working_directories(tmp_path, monkeypatch, capsys):
    from freqai import server
    database = tmp_path / "central" / "memory.sqlite3"
    monkeypatch.setattr(cli, "DEFAULT_MEMORY_PATH", database)
    source = jsonl(tmp_path / "corpus.jsonl", [{"id": "wartung", "source": "Handbuch",
                  "text": "Die Wartung der Testanlage Nova erfolgt jeden Montag um 10 Uhr."}])
    cli.main(["import", source])
    capsys.readouterr()
    different_cwd = tmp_path / "other"
    different_cwd.mkdir()
    monkeypatch.chdir(different_cwd)
    cli.main(["ask", "Wann erfolgt die Wartung der Testanlage Nova?", "--json"])
    answer = json.loads(capsys.readouterr().out)
    assert answer["matches"][0]["id"] == "wartung"
    relations = tmp_path / "relations.json"
    relations.write_text(json.dumps([
        {"subject": "Pudel", "predicate": "is_a", "object": "Hund"},
        {"subject": "Hund", "predicate": "is_a", "object": "Tier"},
    ]), encoding="utf-8")
    cli.main(["import-relations", str(relations)])
    capsys.readouterr()
    cli.main(["infer", "Pudel", "Tier"])
    assert json.loads(capsys.readouterr().out)["entailed"]
    captured = {}

    def capture_server(memory, **kwargs):
        captured["memory"] = memory
        captured.update(kwargs)

    monkeypatch.setattr(server, "serve", capture_server)
    cli.main(["serve", "--port", "8766"])
    capsys.readouterr()
    assert captured["central_store"].path == database
    assert captured["memory"].documents[0].id == "wartung"
    assert captured["port"] == 8766
    assert MemoryStore(database).stats()["history_count"] == 2
    assert not (different_cwd / "memory").exists()


def test_serve_explicit_data_only_adds_and_repeated_start_is_idempotent(tmp_path, monkeypatch, capsys):
    from freqai import server
    database = tmp_path / "memory.sqlite3"
    MemoryStore(database).append_documents([Document("keep", "Nutzerdaten bleiben erhalten.")])
    source = jsonl(tmp_path / "seed.jsonl", [{"id": "demo", "text": "Zusätzliches Beispiel."}])
    monkeypatch.setattr(server, "serve", lambda *args, **kwargs: None)
    for _ in range(2):
        cli.main(["serve", "--memory", str(database), "--data", source])
        capsys.readouterr()
    revision, documents = MemoryStore(database).snapshot()
    assert revision == 2
    assert [document.id for document in documents] == ["keep", "demo"]


def test_unnamed_json_records_have_content_ids_without_cross_file_collisions(tmp_path):
    first = jsonl(tmp_path / "one.jsonl", [{"text": "Erster Text", "source": "Quelle"}])
    second = jsonl(tmp_path / "two.jsonl", [{"text": "Zweiter Text", "source": "Quelle"}])
    docs = cli.read_documents(first) + cli.read_documents(second)
    assert docs[0].id != docs[1].id
    store = MemoryStore(tmp_path / "memory.sqlite3")
    assert len(store.append_documents(docs)["added"]) == 2
    assert len(store.append_documents(cli.read_documents(first))["existing"]) == 1


def test_export_requires_explicit_npz_destination(tmp_path, capsys):
    database = tmp_path / "memory.sqlite3"
    with pytest.raises(SystemExit) as error:
        cli.main(["export", "--memory", str(database)])
    assert error.value.code == 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        cli.main(["export", "--memory", str(database), "--output", str(tmp_path / "wrong.sqlite3")])
    assert error.value.code == 2
    assert not (tmp_path / "wrong.sqlite3").exists()


def test_dialogue_import_and_direct_add_use_prompt_as_key_and_text_as_response(tmp_path, capsys):
    database = tmp_path / "memory.sqlite3"
    corpus = jsonl(tmp_path / "dialogue.jsonl", [{
        "id": "sleep", "prompt": "Ich habe schlecht geschlafen.",
        "text": "Nimm dir heute etwas Ruhe, wenn du kannst.", "source": "Alltag",
    }])
    cli.main(["import", corpus, "--memory", str(database)])
    capsys.readouterr()
    cli.main(["ask", "Ich habe schlecht geschlafen.", "--json", "--memory", str(database)])
    answer = json.loads(capsys.readouterr().out)
    assert answer["answer"] == "Nimm dir heute etwas Ruhe, wenn du kannst."
    cli.main(["add", "--prompt", "Guten Morgen!", "--text", "Wie schön, dich zu sehen!",
              "--source", "Alltag", "--memory", str(database)])
    assert json.loads(capsys.readouterr().out)["added"] == 1
    cli.main(["add", "--prompt", "Guten Morgen!", "--text", "Wie schön, dich zu sehen!",
              "--source", "Alltag", "--memory", str(database)])
    assert json.loads(capsys.readouterr().out)["existing"] == 1
    exported = tmp_path / "dialogues.npz"
    cli.main(["export", "--output", str(exported), "--memory", str(database)])
    capsys.readouterr()
    assert WaveMemory.load(exported).documents == MemoryStore(database).snapshot()[1]


def test_archive_command_preserves_input_but_does_not_activate_legacy_answers(tmp_path, capsys):
    archive = tmp_path / "old.npz"
    documents = [Document("legacy", "Alter Faktentext.", "Archiv")]
    WaveMemory(documents, dimensions=64).save(archive)
    original_bytes = archive.read_bytes()
    database = tmp_path / "memory.sqlite3"
    cli.main(["archive", str(archive), "--memory", str(database)])
    assert json.loads(capsys.readouterr().out)["added"] == 1
    store = MemoryStore(database)
    assert store.snapshot() == (0, [])
    assert store.archived_documents() == documents
    assert archive.read_bytes() == original_bytes
