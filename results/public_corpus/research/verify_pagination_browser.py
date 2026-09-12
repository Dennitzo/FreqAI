"""Verify the real paginated UI against 12k texts in an isolated temporary DB."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright
from freqai.cli import read_documents
from freqai.memory import Document
from freqai.server import make_server
from freqai.store import MemoryStore


def run():
    output = Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix="freqai-pagination-", ignore_cleanup_errors=True) as temporary:
        store = MemoryStore(Path(temporary) / "memory.sqlite3")
        documents = read_documents(ROOT / "memory/fixtures/extension_120.jsonl")
        documents += read_documents(ROOT / "memory/language/generative_corpus.jsonl")
        documents += read_documents(ROOT / "memory/imports/public_dialogue_expansion.jsonl")
        documents.append(Document("pagination-seed", "Diese zusätzliche Zeile vervollständigt den Testbestand.", "Browserprüfung"))
        assert len(documents) == 12000
        store.append_documents(documents)
        server = make_server(store.load_memory(), port=0, central_store=store)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        worker.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.set_default_timeout(120000)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                requested_pages = []
                page.on("request", lambda request: requested_pages.append(request.url) if "/api/documents" in request.url and request.method == "GET" else None)
                url = "http://127.0.0.1:" + str(server.server_address[1])
                page.goto(url)
                page.wait_for_function("() => document.querySelector('#documents').textContent === '12.000'")
                assert page.locator("#memory-rows tr").count() == 0
                page.locator("#memory-details summary").click()
                page.wait_for_function("() => document.querySelectorAll('#memory-rows tr').length === 50")
                first = page.locator("#memory-rows tr td:first-child").all_text_contents()
                first_summary = page.locator("#memory-page-summary").inner_text()
                assert "1–50 von 12.000" in first_summary
                assert page.locator("#memory-previous").is_disabled()
                page.locator("#memory-next").click()
                page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('51–100')")
                second = page.locator("#memory-rows tr td:first-child").all_text_contents()
                second_summary = page.locator("#memory-page-summary").inner_text()
                assert len(second) == 50 and not set(first).intersection(second)
                page.locator("#memory-previous").click()
                page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('1–50')")
                assert first == page.locator("#memory-rows tr td:first-child").all_text_contents()
                # Check a public source through the same UI loader without clicking 13 times.
                page.evaluate("() => refreshDocuments(650)")
                page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('651–700')")
                public_source = page.locator("#memory-rows tr td:first-child").first.inner_text()
                assert "OpenAssistant/oasst2" in public_source and "Apache-2.0" in public_source
                before = server.engine.state()
                store.append_documents([Document("pagination-live", "Ein weiterer Text kommt während der laufenden Prüfung dazu.", "Browser-Liveimport")])
                page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('1–50 von 12.001')")
                after = server.engine.state()
                assert after["live_revision"] > before["live_revision"] and after["time_s"] > before["time_s"]
                assert page.locator("#memory-rows tr").count() == 50
                page.locator("#memory-details").scroll_into_view_if_needed()
                page.screenshot(path=str(output / "pagination-desktop.png"))

                page.locator("#prompt").fill("Hallo")
                started = time.perf_counter()
                with page.expect_response(lambda response: response.url.endswith("/api/ask") and response.request.method == "POST", timeout=120000) as response:
                    page.locator("#ask-button").click()
                result = response.value.json()
                ask_seconds = time.perf_counter() - started
                assert response.value.status == 200, result
                assert result["wave_generation"] and result["matches"] == [] and result["decoder"]["steps"]
                page.wait_for_function("() => !document.querySelector('#ask-button').disabled")
                mobile = browser.new_page(viewport={"width": 390, "height": 844})
                mobile.goto(url)
                mobile.locator("#memory-details summary").click()
                mobile.wait_for_function("() => document.querySelectorAll('#memory-rows tr').length === 50")
                mobile.locator("#memory-details").scroll_into_view_if_needed()
                mobile.screenshot(path=str(output / "pagination-mobile.png"))
                assert mobile.locator("#memory-next").is_visible()
                assert not errors, errors
                assert requested_pages and all("limit=50" in address for address in requested_pages)
                report = {
                    "database": "temporary only; productive memory untouched", "initial_documents": 12000,
                    "final_documents": 12001, "maximum_rendered_document_rows": 50,
                    "first_page": first_summary, "second_page": second_summary,
                    "first_and_second_disjoint": True, "previous_page_identical": True,
                    "public_source": public_source, "live_revision_resets_page_zero": True,
                    "time_advances_during_live_import": True, "mobile_pagination_visible": True,
                    "document_requests": requested_pages, "javascript_errors": errors,
                    "wave_prompt": "Hallo", "wave_answer": result["answer"], "wave_ask_seconds": ask_seconds,
                    "wave_vocabulary": result["decoder"]["vocabulary_size"],
                    "server_sha256": hashlib.sha256((ROOT / "freqai/server.py").read_bytes()).hexdigest(),
                    "dashboard_sha256": hashlib.sha256((ROOT / "freqai/dashboard.html").read_bytes()).hexdigest(),
                }
                (output / "pagination-browser.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(report, ensure_ascii=True, indent=2))
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == "__main__":
    run()
