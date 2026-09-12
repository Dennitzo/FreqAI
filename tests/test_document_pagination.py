"""Real HTTP pagination stays bounded and detects changes between pages."""
import pytest

from test_server import TinyMemory, request, running_server


def many_documents(count=123):
    memory = TinyMemory()
    memory.documents = [{"id": f"row-{i:04d}",
                         "text": f"Ein vollständiger Text mit der Nummer {i}.", "source": "Seitenprüfung"}
                        for i in range(count)]
    return memory


def test_legacy_listing_stays_complete_and_pages_are_stable():
    with running_server(many_documents()) as server:
        status, legacy, _ = request(server, "GET", "/api/documents")
        assert status == 200 and len(legacy["documents"]) == 123
        assert "limit" not in legacy
        status, first, _ = request(server, "GET", "/api/documents?offset=0&limit=50")
        assert status == 200 and len(first["documents"]) == 50
        assert first["document_count"] == first["total"] == 123
        assert first["offset"] == 0 and first["limit"] == 50
        assert first["previous_offset"] is None and first["next_offset"] == 50
        revision = first["live_revision"]
        status, second, _ = request(server, "GET", f"/api/documents?offset=50&limit=50&revision={revision}")
        assert status == 200 and second["live_revision"] == revision
        assert second["previous_offset"] == 0 and second["next_offset"] == 100
        status, third, _ = request(server, "GET", f"/api/documents?offset=100&limit=50&revision={revision}")
        assert status == 200 and len(third["documents"]) == 23
        assert third["previous_offset"] == 50 and third["next_offset"] is None
        assert first["documents"] + second["documents"] + third["documents"] == legacy["documents"]


def test_page_revision_conflict_does_not_mix_old_and_new_results():
    with running_server(many_documents(55)) as server:
        _, first, _ = request(server, "GET", "/api/documents?limit=50")
        request(server, "POST", "/api/documents", {"text": "Ein neuer Eintrag ergänzt die laufende Schwingung."})
        status, stale, _ = request(server, "GET", f"/api/documents?offset=50&limit=50&revision={first['revision']}")
        assert status == 409 and "error" in stale and "documents" not in stale
        status, fresh, _ = request(server, "GET", "/api/documents?offset=0&limit=50")
        assert status == 200 and fresh["revision"] > first["revision"]
        assert fresh["total"] == 56 and len(fresh["documents"]) == 50


@pytest.mark.parametrize("query", [
    "limit=0", "limit=201", "limit=-1", "limit=1.5", "limit=true", "limit=", "offset=-1",
    "offset=1.0", "offset=", "revision=-1", "revision=abc", "limit=50&limit=10",
    "offset=0&limit=50&revision=0&extra=1", "unknown=50", "limit=%D9%A5%D9%A0",
])
def test_invalid_pagination_parameters_are_rejected(query):
    with running_server() as server:
        status, result, _ = request(server, "GET", "/api/documents?" + query)
        assert status == 400 and "error" in result


def test_paginated_requests_preserve_origin_guard():
    with running_server() as server:
        status, result, _ = request(server, "GET", "/api/documents?limit=50", headers={"Origin": "https://unrelated.invalid"})
        assert status == 403 and "error" in result


def test_empty_and_out_of_range_pages_are_bounded():
    with running_server(many_documents(0)) as server:
        status, result, _ = request(server, "GET", "/api/documents?limit=50")
        assert status == 200 and result["documents"] == [] and result["total"] == 0
        assert result["next_offset"] is None and result["previous_offset"] is None
    with running_server(many_documents(2)) as server:
        status, result, _ = request(server, "GET", "/api/documents?offset=999&limit=50")
        assert status == 200 and result["documents"] == [] and result["total"] == 2
        assert result["next_offset"] is None and result["previous_offset"] == 0


def test_only_selected_rows_are_converted_to_response_documents():
    class UnreadableRow:
        def keys(self):
            raise AssertionError("Rows outside the requested page must not be serialized")

    memory = many_documents(1)
    memory.documents.extend([UnreadableRow()] * 11999)
    with running_server(memory) as server:
        status, result, _ = request(server, "GET", "/api/documents?offset=0&limit=1")
        assert status == 200 and result["total"] == 12000 and len(result["documents"]) == 1
