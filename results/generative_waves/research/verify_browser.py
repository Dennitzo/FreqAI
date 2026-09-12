"""Real browser integration check; all imports and history stay in a temp DB."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

from freqai.cli import read_documents
from freqai.server import make_server
from freqai.store import MemoryStore


def run():
    out = Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix="freqai-wave-browser-") as temporary:
        store = MemoryStore(Path(temporary) / "memory.sqlite3")
        documents = read_documents(ROOT / "memory/fixtures/extension_120.jsonl")
        documents += read_documents(ROOT / "memory/language/generative_corpus.jsonl")
        store.append_documents(documents)
        server = make_server(store.load_memory(), port=0, central_store=store)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        worker.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1100})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                url = "http://127.0.0.1:" + str(server.server_address[1]) + "/"
                page.goto(url)
                page.wait_for_function("() => !document.querySelector('#ask-button').disabled")
                assert page.locator("#generation-mode").input_value() == "wave"

                def ask(prompt):
                    page.locator("#prompt").fill(prompt)
                    started = time.perf_counter()
                    with page.expect_response(lambda r: r.url.endswith("/api/ask") and r.request.method == "POST",
                                              timeout=90000) as response:
                        page.locator("#ask-button").click()
                    result = response.value.json()
                    assert response.value.status == 200, result
                    page.wait_for_function("() => !document.querySelector('#ask-button').disabled")
                    return result, time.perf_counter() - started

                first, first_seconds = ask("Mir geht es gut, wie geht es dir denn?")
                assert first["wave_generation"] and first["matches"] == [] and first["decoder"]["steps"]
                session = page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')")
                page.locator("#context-details").evaluate("element => element.open = true")
                page.locator("#answer-method").evaluate("element => element.parentElement.open = true")
                page.locator("#token-details").evaluate("element => element.open = true")
                page.wait_for_function("() => currentState && currentState.context_wave && currentState.context_wave.energy > 0")
                frame_a = page.evaluate("() => currentState.context_wave")
                page.wait_for_function("oldTime => currentState.context_wave.time_s > oldTime + .15", arg=frame_a["time_s"])
                frame_b = page.evaluate("() => currentState.context_wave")
                assert frame_a["displacement"] != frame_b["displacement"]
                assert abs(frame_a["energy"] - frame_b["energy"]) < 1e-10
                page.screenshot(path=str(out / "generative-browser-desktop.png"), full_page=True)

                before_legacy = store.load_conversation(session)[1]["generation"]
                page.locator("#generation-mode").select_option("dialogue")
                legacy, _ = ask("Hallo")
                assert legacy["mode"] == "dialogue"
                assert store.load_conversation(session)[1]["generation"] == before_legacy
                page.locator("#generation-mode").select_option("wave")
                page.reload()
                page.wait_for_function("() => !document.querySelector('#ask-button').disabled")
                assert page.locator("#generation-mode").input_value() == "wave"
                assert page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')") == session
                assert page.locator(".chat-message.assistant").count() == 2

                page.locator("#source").fill("Browserprüfung")
                page.locator("#cue").fill("Beschreibe Leuchtkäfer")
                page.locator("#document").fill("Leuchtkäfer schimmern bernsteinfarben.")
                with page.expect_response(lambda r: r.url.endswith("/api/documents") and r.request.method == "POST",
                                          timeout=90000) as response:
                    page.locator("#ingest-button").click()
                imported = response.value.json()
                assert response.value.status == 201, imported
                page.wait_for_function("() => !document.querySelector('#ingest-button').disabled")
                second, second_seconds = ask("Beschreibe Leuchtkäfer")
                assert second["decoder"]["vocabulary_size"] > first["decoder"]["vocabulary_size"]
                assert second["conversation_revision"] == 3 and second["wave_generation"]
                assert imported["live_revision"] == second["live_revision"]

                other = browser.new_page(viewport={"width": 390, "height": 844})
                other.on("pageerror", lambda error: errors.append(str(error)))
                other.goto(url)
                other.wait_for_function("() => !document.querySelector('#ask-button').disabled")
                other_session = other.evaluate("() => sessionStorage.getItem('freqai.conversation.id')")
                assert other_session != session and other.locator(".chat-message").count() == 0
                assert store.load_conversation(other_session) == (0, {})
                other.screenshot(path=str(out / "generative-browser-mobile.png"), full_page=True)
                old_session = session
                page.locator("#new-conversation").click()
                session = page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')")
                assert session != old_session and page.locator(".chat-message").count() == 0
                assert store.load_conversation(session) == (0, {})
                assert store.load_conversation(old_session)[0] == 3
                assert not errors, errors
                report = {
                    "corpus_documents_before_live_append": len(documents),
                    "first_answer": first["answer"], "first_seconds": first_seconds,
                    "first_token_steps": len(first["decoder"]["steps"]), "first_decoder": first["decoder"],
                    "second_answer": second["answer"], "second_seconds": second_seconds,
                    "second_vocabulary_size": second["decoder"]["vocabulary_size"],
                    "first_vocabulary_size": first["decoder"]["vocabulary_size"],
                    "live_append_document_count": imported["document_count"],
                    "context_moves_with_constant_energy": True, "context_energy": frame_b["energy"],
                    "legacy_toggle_preserves_field": True, "reload_preserves_mode_session_history": True,
                    "separate_page_starts_empty_session": True, "new_conversation_retains_previous_history": True,
                    "javascript_errors": errors, "productive_database_touched": False,
                }
                (out / "generative-browser.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({k: v for k, v in report.items() if k != "first_decoder"}, ensure_ascii=True, indent=2))
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    run()
