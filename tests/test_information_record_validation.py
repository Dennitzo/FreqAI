"""Real ingestion must reject QA metadata before projecting to Document."""
from http.client import HTTPConnection
import json
import threading

import pytest

from freqai.cli import main, read_documents
from freqai.memory import Document, WaveMemory
from freqai.server import make_server
from freqai.store import MemoryStore


def test_cli_rejects_raw_qa_fields_without_partial_batch(tmp_path):
    source = tmp_path/"input.jsonl"
    store_path = tmp_path/"memory.sqlite3"
    for field in ("question", "answer", "messages", "instruction"):
        rows = [{"id":"valid","text":"Eine Frequenz ist ein Messwert."},
                {"id":"paired","text":"Eine gespeicherte Antwort.",field:"Nicht verwerfen"}]
        source.write_text("\n".join(json.dumps(row) for row in rows),encoding="utf-8")
        with pytest.raises(SystemExit) as error:
            main(["import",str(source),"--memory",str(store_path)])
        assert error.value.code!=0
        assert MemoryStore(store_path).snapshot()[1]==[]


def test_http_rejects_qa_fields_in_rows_and_envelope_without_commit(tmp_path):
    store = MemoryStore(tmp_path/"memory.sqlite3")
    server = make_server(store.load_memory(),port=0,central_store=store)
    worker = threading.Thread(target=server.serve_forever,kwargs={"poll_interval":.01},daemon=True)
    worker.start()
    try:
        for body in (
            {"documents":[{"id":"valid","text":"Ein Messwert ist eine Zahl."},
                          {"id":"paired","text":"Die Antwort.","messages":[]}]},
            {"text":"Die Antwort.","question":"Eine Frage?"},
            {"documents":[{"text":"Die Antwort."}],"answer":"Nicht verwerfen"},
        ):
            connection = HTTPConnection("127.0.0.1",server.server_address[1],timeout=3)
            try:
                connection.request("POST","/api/documents",json.dumps(body),{"Content-Type":"application/json"})
                response = connection.getresponse()
                payload = json.loads(response.read())
                assert response.status==400, payload
                assert "Dialogfelder" in payload["error"]
            finally:
                connection.close()
            assert store.snapshot()[1]==[]
            assert server.engine.memory.documents==[]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive()


def test_explicit_archive_preserves_supported_legacy_pair_but_not_unknown_fields(tmp_path):
    source = tmp_path/"legacy.jsonl"
    row = {"id":"old","text":"Historische Antwort.","source":"Altdaten","prompt":"Historische Frage?"}
    source.write_text(json.dumps(row),encoding="utf-8")
    with pytest.raises(ValueError,match="Altpaare"):
        read_documents(source)
    main(["archive",str(source),"--memory",str(tmp_path/"memory.sqlite3")])
    store = MemoryStore(tmp_path/"memory.sqlite3")
    assert store.snapshot()[1]==[]
    assert store.archived_documents()==[Document("old","Historische Antwort.","Altdaten","Historische Frage?")]
    source.write_text(json.dumps({**row,"question":"Würde sonst verloren gehen"}),encoding="utf-8")
    with pytest.raises(ValueError,match="Dialogfelder"):
        read_documents(source,allow_legacy_pairs=True)


def test_npz_legacy_pairs_require_explicit_archive_read(tmp_path):
    document = Document("old","Historische Antwort.","Altdaten","Historische Frage?")
    source = tmp_path/"legacy.npz"
    WaveMemory([document]).save(source)
    with pytest.raises(ValueError,match="Altpaare"):
        read_documents(source)
    assert read_documents(source,allow_legacy_pairs=True)==[document]
