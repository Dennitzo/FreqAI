import numpy as np
import pytest

from freqai.reasoning import WaveReasoner


def triples(*edges):
    return [{"subject": a, "predicate": "is_a", "object": b, "source": f"edge:{i}"}
            for i, (a, b) in enumerate(edges)]


def test_new_answer_has_explicit_two_hop_proof():
    supplied = triples(("Pudel", "Hund"), ("Hund", "Tier"))
    reasoner = WaveReasoner(supplied)
    answer = reasoner.infer("Pudel", "Tier")
    assert answer["answer"] == "Pudel ist ein Tier."
    assert answer["entailed"]
    assert answer["path"] == ["Pudel", "Hund", "Tier"]
    assert answer["steps"] == 2
    assert answer["proof"] == supplied
    assert answer["sources"] == ["edge:0", "edge:1"]
    assert not any(x["subject"] == "Pudel" and x["object"] == "Tier" for x in supplied)


def test_operator_is_graph_in_fourier_basis():
    reasoner = WaveReasoner(triples(("A", "B"), ("B", "C"), ("B", "A")))
    f = np.fft.fft(np.eye(3), axis=0, norm="ortho")
    np.testing.assert_allclose(f.conj().T @ reasoner.operator @ f,
                               reasoner.adjacency, atol=1e-13)
    assert reasoner.infer("A", "C")["entailed"]


def test_cycle_unknown_identity_and_hop_bound():
    reasoner = WaveReasoner(triples(("A", "B"), ("B", "A"), ("B", "C"), ("D", "D")))
    limited = reasoner.infer("A", "C", max_hops=1)
    assert not limited["entailed"] and limited["reason"] == "hop_limit"
    assert reasoner.infer("A", "C", max_hops=2)["entailed"]
    unreachable = reasoner.infer("A", "D", max_hops=100)
    assert not unreachable["entailed"] and unreachable["reason"] == "no_path"
    assert unreachable["steps"] <= len(reasoner.nodes)
    assert "unbekannt" in unreachable["answer"]
    assert reasoner.infer("A", "missing")["reason"] == "unknown_node"
    assert reasoner.infer("A", "A", max_hops=0)["reason"] == "reflexivity"
    assert reasoner.infer("missing", "missing")["reason"] == "unknown_node"
    assert WaveReasoner([]).infer("A", "B")["reason"] == "unknown_node"


def test_numerical_failure_cannot_generate_claim():
    reasoner = WaveReasoner(triples(("A", "B"), ("C", "C")))
    reasoner.operator = reasoner.operator + (0.1 + 0.3j)
    answer = reasoner.infer("A", "C")
    assert not answer["entailed"] and answer["reason"] == "numerical_error"
    assert not answer["proof"]


def test_integer_sized_operator_corruption_still_requires_a_witness():
    reasoner = WaveReasoner(triples(("A", "B"), ("C", "C")))
    fake = reasoner.adjacency.copy()
    fake[reasoner.node_index["C"], reasoner.node_index["A"]] = 1
    f = np.fft.fft(np.eye(3), axis=0, norm="ortho")
    reasoner.operator = f @ fake @ f.conj().T
    result = reasoner.infer("A", "C")
    assert not result["entailed"] and result["reason"] == "numerical_error"


@pytest.mark.parametrize("record", [
    {"subject": "a", "predicate": "likes", "object": "b"},
    {"subject": "", "predicate": "is_a", "object": "b"},
    {"subject": "a", "predicate": "is_a", "object": "b", "source": ""},
])
def test_invalid_relation_rejected(record):
    with pytest.raises(ValueError):
        WaveReasoner([record])


@pytest.mark.parametrize("max_hops", [-1, 1.5, True])
def test_invalid_hop_bound_rejected(max_hops):
    with pytest.raises(ValueError):
        WaveReasoner(triples(("a", "b"))).infer("a", "b", max_hops=max_hops)


def test_duplicate_edges_use_deterministic_provenance():
    data = [dict(subject="A", predicate="is_a", object="B", source=source)
            for source in ["z", "a"]]
    assert WaveReasoner(data).infer("A", "B")["sources"] == ["a"]
