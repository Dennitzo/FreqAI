"""Independent data-integrity and incremental numerical invariants."""

from collections import Counter
import json

import numpy as np

from experiments.run_extended import validate_inputs
from freqai.codec import decode_coefficients, decode_text, modal_state, recover_coefficients
from freqai.memory import Document, WaveMemory


def extension_inputs():
    """Fresh numerical fixtures, not copies of the removed training pairs."""
    extension = [{"id": f"station-{index:03}",
                  "text": f"Die Station Standort{index:03} besitzt {index + 7} Messgeräte für Größe α.",
                  "source": f"Synthetischer Prüfbereich {index // 10}"} for index in range(120)]
    cases = []
    for split, indices, nulls in (("development", range(50), 10), ("holdout", range(50, 120), 14),
                                  ("stress", range(0), 20)):
        for index in indices:
            cases.append({"id": f"{split}-{index}", "split": split, "group": "known",
                          "prompt": f"Welche Messgeräte besitzt Standort{index:03}?",
                          "expected_id": f"station-{index:03}"})
        for index in range(nulls):
            cases.append({"id": f"{split}-null-{index}", "split": split, "group": "null",
                          "prompt": f"Unbekannter Prüfgegenstand Fremdort{index:03}?", "expected_id": None})
    return [], extension, cases


def test_extension_is_additional_diverse_and_has_separate_holdout():
    base, extension, cases = extension_inputs()
    validate_inputs(base, extension, cases)
    assert len(extension) == 120
    assert not ({item["id"] for item in base} & {item["id"] for item in extension})
    assert len({item["source"] for item in extension}) == 12
    counts = Counter(case["split"] for case in cases)
    assert counts == {"development": 60, "holdout": 84, "stress": 20}
    development_ids = {case["expected_id"] for case in cases if case["split"] == "development"}
    holdout_ids = {case["expected_id"] for case in cases if case["split"] == "holdout"}
    assert development_ids & holdout_ids == {None}


def test_extension_payloads_survive_full_quadrature_recovery():
    from freqai.codec import encode_text

    for document in extension_inputs()[1]:
        packet = encode_text(document["text"])
        for time_s in (.017, 1e12):
            q, p = modal_state(packet, time_s)
            restored = recover_coefficients(q, p, time_s)
            assert decode_coefficients(restored, packet.byte_length, packet.sha256) == document["text"]


def test_120_incremental_additions_preserve_previous_modes_and_return_new_payloads():
    extension = [
        {"id": f"station-{index:03}",
         "text": f"Die Station Standort{index:03} besitzt {index + 7} Messgeräte.",
         "source": "Deklarative Fakten für die inkrementelle Speicherprüfung"}
        for index in range(120)
    ]
    memory = WaveMemory([], dimensions=256)
    for data in extension:
        packets = memory.payloads[:]
        metadata_packets = memory.metadata_payloads[:]
        spectra = memory.spectra.copy()
        frequencies = memory._relative_frequencies.copy()
        document = Document(**data)
        replacement = memory.with_documents_added([document])
        assert memory.documents == replacement.documents[:-1]
        assert all(old is replacement.payloads[index] for index, old in enumerate(packets))
        assert all(old is replacement.metadata_payloads[index] for index, old in enumerate(metadata_packets))
        np.testing.assert_array_equal(replacement.spectra[:-1], spectra)
        np.testing.assert_array_equal(replacement._relative_frequencies[:len(frequencies)], frequencies)
        assert decode_text(replacement.payloads[-1], 1e12) == document.text
        assert json.loads(decode_text(replacement.metadata_payloads[-1], 1e12)) == {
            "id": document.id, "source": document.source}
        assert replacement.ask(document.text)["matches"][0]["id"] == document.id
        memory = replacement
    snapshot = memory.snapshot(.213, points=32)
    assert snapshot["document_count"] == 120
    assert snapshot["key_mode_count"] == 120 * 256
    assert snapshot["modal_count"] == sum(len(packet.coefficients)
                                          for packet in memory.payloads + memory.metadata_payloads)


def test_incremental_and_full_construction_have_same_numerical_lookup():
    extension = [Document(**item) for item in extension_inputs()[1]]
    initial = WaveMemory(extension[:40], dimensions=1024, feature_mode="morphology",
                         retrieval_policy="coverage", min_coverage=.6)
    appended = initial.with_documents_added(extension[40:])
    rebuilt = WaveMemory(extension, dimensions=1024, feature_mode="morphology",
                         retrieval_policy="coverage", min_coverage=.6)
    np.testing.assert_array_equal(appended.spectra, rebuilt.spectra)
    cases = extension_inputs()[2]
    # Only development prompts are used by regression tests before freeze.
    for case in cases:
        if case["split"] != "development":
            continue
        np.testing.assert_allclose(appended.scores(case["prompt"], 1e12),
                                   rebuilt.direct_cosine_scores(case["prompt"]), atol=1e-13, rtol=0)
        assert appended.ask(case["prompt"], time_s=1e12) == rebuilt.ask(case["prompt"], time_s=1e12)
