"""Reproducible browser acceptance with isolated memory; no productive writes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright, expect
from freqai.memory import Document
from freqai.server import make_server
from freqai.store import MemoryStore


def main():
    output = Path(__file__).resolve().parent
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "checks": {}}
    with TemporaryDirectory(prefix="freqai-browser-live-") as temporary:
        store = MemoryStore(Path(temporary) / "memory.sqlite3")
        store.append_documents([
            Document("day", "Erzähl mir, was du heute gemacht hast.", "Browser-Test", prompt="Mein Tag war entspannt"),
            Document("thanks", "Gern!", "Browser-Test", prompt="Danke für das Gespräch"),
            Document("bye", "Tschüss! Bis zum nächsten Mal.", "Browser-Test", prompt="Tschüss"),
        ])
        server = make_server(store.load_memory(), port=0, central_store=store)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1180})
                errors, navigations = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("framenavigated", lambda frame: navigations.append(frame.url))
                page.goto(url)
                expect(page.locator("#documents")).to_have_text("3")
                initial_state = page.evaluate("fetch('/api/state').then(r => r.json())")
                page.locator("#memory-details summary").click()
                expect(page.locator("#memory-rows tr")).to_have_count(3)
                text = "Hallo! Wie läuft dein Tag?"
                page.locator("#source").fill("Eigene Alltagsantwort")
                page.locator("#cue").fill("Hallo")
                page.locator("#document").fill(text)
                start = time.perf_counter()
                page.locator("#ingest-button").click()
                expect(page.locator("#ingest-feedback")).to_contain_text("sofort aktiv")
                report["browser_ingest_visible_ms"] = 1000 * (time.perf_counter() - start)
                expect(page.locator("#memory-rows tr")).to_have_count(4)
                for cue, answer in [("Hallo", text), ("Mein Tag war entspannt", "Erzähl mir, was du heute gemacht hast."),
                                    ("Danke für das Gespräch", "Gern!"), ("Tschüss", "Tschüss! Bis zum nächsten Mal.")]:
                    page.locator("#prompt").fill(cue)
                    page.locator("#ask-button").click()
                    expect(page.locator("#answer")).to_contain_text(answer)
                    expect(page.locator("#ask-button")).to_be_enabled()
                expect(page.locator("#conversation .chat-message")).to_have_count(8)
                report["checks"]["ui_add_immediate_answer"] = True
                report["checks"]["four_turn_conversation_visible"] = True
                page.screenshot(path=str(output / "browser_live_add.png"), full_page=True)

                # A separate store instance simulates a CLI import while the
                # browser remains open; import all 120 documents in one commit.
                external = MemoryStore(store.path)
                corpus = ROOT / "memory" / "fixtures" / "extension_120.jsonl"
                rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
                report["imported_corpus_sha256"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
                additions = [Document(**row) for row in rows]
                assert len(additions) >= 120
                assert all(document.prompt for document in additions)
                imported = external.append_documents(additions)
                expected_count = 4 + len(additions)
                expect(page.locator("#documents")).to_have_text(str(expected_count))
                expect(page.locator("#memory-rows tr")).to_have_count(expected_count)
                probe = additions[0]
                page.locator("#prompt").fill(probe.prompt)
                page.locator("#ask-button").click()
                expect(page.locator("#answer")).to_contain_text(probe.text)
                expect(page.locator("#memory-summary")).to_contain_text("5 Abfragen")
                final_state = page.evaluate("fetch('/api/state').then(r => r.json())")
                assert final_state["live_revision"] == imported["revision"] == 3
                assert final_state["time_s"] > initial_state["time_s"]
                assert len(navigations) == 1
                assert not errors
                report["checks"].update({"external_120_import_visible": True,
                                          "external_text_immediately_answered": True,
                                          "no_page_reload": True, "clock_continued": True,
                                          "history_separate_from_document_index": store.stats()["document_count"] == expected_count,
                                          "javascript_errors": errors})
                report["initial"] = {key: initial_state[key] for key in ("ticks", "time_s", "document_count", "live_revision")}
                report["final"] = {key: final_state[key] for key in ("ticks", "time_s", "document_count", "live_revision")}
                report["history_count"] = store.stats()["history_count"]
                report["temporary_server_port"] = server.server_address[1]
                report["navigation_count"] = len(navigations)
                page.screenshot(path=str(output / "browser_external_120.png"), full_page=True)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
    (output / "browser_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
