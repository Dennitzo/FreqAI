import json

import numpy as np
import pytest

from freqai.codec import WavePacket, decode_text, encode_text, modal_state, recover_coefficients
from freqai.memory import Document, WaveMemory


@pytest.mark.parametrize("text", ["", "Grüße 🌊\n\x00", "汉字 日本語 العربية", "A" * 10001])
@pytest.mark.parametrize("time_s", [0, 0.0083333333, 123.456, 1e8])
def test_wave_roundtrip_at_arbitrary_time(text, time_s):
    assert decode_text(encode_text(text), time_s) == text


def test_corrupted_payload_fails_closed():
    packet = encode_text("Validierter Text")
    altered = packet.coefficients.copy()
    altered[0] += 1.0
    with pytest.raises(ValueError, match="corruption"):
        decode_text(WavePacket(altered, packet.byte_length, packet.sha256))


def test_quadrature_recovers_displacement_zero_crossing():
    packet = encode_text("ab")
    time_s = 1 / (4 * 15)
    q, p = modal_state(packet, time_s)
    assert abs(q[1]) < 1e-14
    assert abs(p[1]) > 1e-5
    np.testing.assert_allclose(recover_coefficients(q, p, time_s), packet.coefficients, atol=1e-14)
    assert decode_text(packet, time_s) == "ab"


def test_archive_persistence_and_interference(tmp_path):
    documents = [Document("a", "Eine stehende Welle hat ortsfeste Knoten.", "α"),
                 Document("b", "Ein Apfel wächst an einem Baum.", "β")]
    memory = WaveMemory(documents)
    path = tmp_path / "memory.npz"
    memory.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"header", "coefficients"}
    restored = WaveMemory.load(path)
    assert restored.documents == documents
    assert json.loads(decode_text(restored.archive))[1]["source"] == "β"
    for time_s in [0, 1.234, 1e7]:
        answer = restored.ask("Was ist eine stehende Welle?", time_s=time_s)
        assert answer["matches"][0]["id"] == "a"
        assert answer["answer"] == documents[0].text
        np.testing.assert_allclose(restored.scores("stehende Welle", time_s),
                                   restored.direct_cosine_scores("stehende Welle"), atol=1e-14)


def test_unknown_empty_duplicate_and_add(tmp_path):
    memory = WaveMemory([])
    assert memory.ask("Mond")['abstained']
    with pytest.raises(ValueError):
        memory.ask(" ")
    memory.add_document("Ein Apfel ist eine Frucht.", document_id="fruit")
    assert memory.ask("Quantenchromodynamik")['abstained']
    with pytest.raises(ValueError, match="already exists"):
        memory.add_document("Nicht speichern", document_id="fruit")
    assert len(memory.documents) == 1
    with pytest.raises(ValueError, match="unique"):
        WaveMemory([Document("a", "1"), Document("a", "2")])


def test_energy_is_invariant_and_snapshot_covers_addressed_document_waves():
    memory = WaveMemory([Document("a", "Hallo Wellen"), Document("b", "Mehr Daten 🌊")])
    first = memory.snapshot(0)
    later = memory.snapshot(834234.12)
    assert later['modal_count'] == later['text_mode_count'] + later['metadata_mode_count']
    assert later['text_mode_count'] == sum(len(doc.text.encode('utf-8')) for doc in memory.documents)
    assert later['metadata_mode_count'] > 0
    assert later['key_mode_count'] == len(memory.documents) * memory.dimensions
    assert later['energy'] == pytest.approx(first['energy'], rel=1e-14)
    assert later['physical_energy'] == pytest.approx(first['physical_energy'], rel=1e-14)
    assert later['key_energy'] == pytest.approx(first['key_energy'], rel=1e-14)
    assert later['displacement'] != first['displacement']


def test_append_preserves_existing_live_wave_and_source_at_same_time():
    memory = WaveMemory([Document("original", "Schwingung mit Umlauten 🌊", "Quelle α")])
    old_packet = memory.payloads[0]
    old_metadata = memory.metadata_payloads[0]
    before = memory.snapshot(27.137, points=10000)
    expanded = memory.with_documents_added([Document("new", "Neuer dauerhaft gespeicherter Text.", "Quelle β")])
    after = expanded.snapshot(27.137, points=10000)
    assert memory.documents == [Document("original", "Schwingung mit Umlauten 🌊", "Quelle α")]
    assert expanded.payloads[0] is old_packet
    assert expanded.metadata_payloads[0] is old_metadata
    count = before['modal_count']
    np.testing.assert_array_equal(after['displacement'][:count], before['displacement'])
    np.testing.assert_array_equal(after['quadrature'][:count], before['quadrature'])
    assert after['energy'] > before['energy']
    assert decode_text(expanded.payloads[0], 27.137) == memory.documents[0].text
    assert json.loads(decode_text(expanded.metadata_payloads[0], 27.137))['source'] == 'Quelle α'
    assert expanded.ask("Neuer dauerhaft gespeicherter Text")["matches"][0]["id"] == "new"


def test_append_encodes_only_new_documents(monkeypatch):
    memory = WaveMemory([Document("old", "Alter Inhalt")])
    import freqai.memory as module
    original = module.encode_text
    encoded = []
    def record(text):
        encoded.append(text)
        return original(text)
    monkeypatch.setattr(module, "encode_text", record)
    expanded = memory.with_documents_added([Document("new", "Neuer Inhalt")])
    assert "Alter Inhalt" not in encoded
    assert "Neuer Inhalt" in encoded
    assert expanded._archive is None  # Export archive not rebuilt on ingestion.


def test_coverage_rejects_unseen_entity_and_handles_fixed_genitive_rules():
    memory = WaveMemory([Document("jp", "Die Hauptstadt Japans heißt Tokio."),
                         Document("it", "Die Hauptstadt Italiens ist Rom.")],
                        feature_mode="morphology", retrieval_policy="coverage")
    assert memory.ask("Welche Hauptstadt hat Japan?")["matches"][0]["id"] == "jp"
    assert memory.ask("Welche Hauptstadt hat Brasilien?")["abstained"]
    assert memory.ask("Welche Hauptstadt hat Italien?")["matches"][0]["id"] == "it"
    assert memory.with_documents_added([Document("br", "Die Hauptstadt Brasiliens ist Brasília.")]).ask(
        "Welche Hauptstadt hat Brasilien?")["matches"][0]["id"] == "br"


def test_conversation_keys_use_stored_utterance_and_decode_response(tmp_path):
    document = Document("greeting", "Hallo! Wie geht es dir?", "Alltag / Begrüßung", "Guten Morgen")
    memory = WaveMemory([document], feature_mode="morphology", retrieval_policy="coverage")
    response = memory.ask("Guten Morgen!", time_s=23.54)
    assert response['answer'] == document.text
    assert response['matches'][0]['matched_prompt'] == document.prompt
    assert json.loads(decode_text(memory.metadata_payloads[0], 23.54))['prompt'] == document.prompt
    np.testing.assert_allclose(memory.scores("Guten Morgen"), memory.direct_cosine_scores("Guten Morgen"), atol=1e-13)
    destination = tmp_path / 'conversation-export.npz'
    memory.save(destination)
    restored = WaveMemory.load(destination)
    assert restored.documents == [document]
    assert restored.ask("Guten Morgen!")['answer'] == document.text
    assert restored.ask("Reaktorsteuerung")['abstained']
