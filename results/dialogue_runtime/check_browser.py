"""Exercise the real dashboard against an isolated temporary SQLite store."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading

from playwright.sync_api import expect, sync_playwright

from freqai.cli import read_documents
from freqai.server import make_server
from freqai.store import MemoryStore


OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="freqai-dialogue-browser-") as temporary:
        store = MemoryStore(Path(temporary) / "memory.sqlite3")
        store.append_documents(read_documents(PROJECT / "memory/fixtures/extension_120.jsonl"))
        server = make_server(store.load_memory(), port=0, central_store=store)
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        errors, replies = [], []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                context = browser.new_context(viewport={"width": 1440, "height": 1160})
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                expect(page.locator("#ask-button")).to_be_enabled()
                expect(page.locator("#connection")).to_contain_text("Berechnung läuft")
                initial = page.evaluate("fetch('/api/state').then(r => r.json())")
                original_session = page.evaluate("sessionStorage.getItem('freqai.conversation.id')")

                def ask(target, prompt):
                    target.locator("#prompt").fill(prompt)
                    with target.expect_response(lambda response: response.url.endswith("/api/ask")
                                                and response.request.method == "POST") as pending:
                        target.locator("#ask-button").click()
                    response = pending.value
                    assert response.status == 200, response.text()
                    result = response.json()
                    expect(target.locator("#ask-button")).to_be_enabled()
                    expect(target.locator("#answer")).to_contain_text(result["answer"])
                    replies.append({"prompt": prompt, "answer": result["answer"],
                                    "abstained": result["abstained"], "method": result.get("method"),
                                    "generated": result.get("generated"),
                                    "session_id": result.get("session_id"),
                                    "conversation_revision": result.get("conversation_revision")})
                    return result

                for prompt in ["Hallo", "Mir geht es gut, wie geht es dir denn?", "gut und dir?"]:
                    assert not ask(page, prompt)["abstained"]
                assert "bereit" in replies[1]["answer"].lower() and "bereit" in replies[2]["answer"].lower()
                page.screenshot(path=str(OUTPUT / "reported-dialogue-desktop.png"), full_page=True)
                assert not ask(page, "Ich heiße Elena.")["abstained"]
                expect(page.locator(".chat-message")).to_have_count(8)
                page.reload()
                expect(page.locator("#ask-button")).to_be_enabled()
                expect(page.locator(".chat-message")).to_have_count(8)
                assert page.evaluate("sessionStorage.getItem('freqai.conversation.id')") == original_session
                assert "Elena" in ask(page, "Wie heiße ich?")["answer"]

                other = context.new_page()
                other.on("pageerror", lambda error: errors.append(str(error)))
                other.goto(url)
                expect(other.locator("#ask-button")).to_be_enabled()
                expect(other.locator(".chat-message")).to_have_count(0)
                assert other.evaluate("sessionStorage.getItem('freqai.conversation.id')") != original_session
                assert "Elena" not in ask(other, "Wie heiße ich?")["answer"]
                expect(page.locator(".chat-message")).to_have_count(10)

                page.locator("#new-conversation").click()
                expect(page.locator(".chat-message")).to_have_count(0)
                assert page.evaluate("sessionStorage.getItem('freqai.conversation.id')") != original_session
                assert "Elena" not in ask(page, "Wie heiße ich?")["answer"]

                page.locator("#source").fill("Isolierter Browsertest")
                page.locator("#cue").fill("Wann trifft sich der Gartenclub Silberlinde?")
                page.locator("#document").fill("Der Gartenclub Silberlinde trifft sich jeden Donnerstag.")
                page.locator("#ingest-button").click()
                expect(page.locator("#ingest-feedback")).to_contain_text("Gespeichert und sofort aktiv")
                expect(page.locator("#documents")).to_have_text("121")
                live_reply = ask(page, "Wann trifft sich der Gartenclub Silberlinde?")
                assert live_reply["answer"] == "Der Gartenclub Silberlinde trifft sich jeden Donnerstag."
                final = page.evaluate("fetch('/api/state').then(r => r.json())")
                assert final["ticks"] > initial["ticks"] and final["time_s"] > initial["time_s"]
                assert final["displacement"] != initial["displacement"]
                assert final["running"] and final["error"] is None
                assert store.stats()["document_count"] == 121
                assert store.stats()["history_count"] == len(replies)
                assert not errors
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=str(OUTPUT / "conversation-mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                report = {"checked_at": datetime.now(timezone.utc).isoformat(), "url": url,
                          "isolated_temporary_database": True, "production_database_modified": False,
                          "reported_three_turn_dialogue_passed": True,
                          "reload_restored_history_and_context": True,
                          "tabs_have_independent_sessions": True, "new_conversation_resets_context": True,
                          "live_addition_answered_without_restart": True,
                          "ticks_advanced": final["ticks"] - initial["ticks"],
                          "document_count": store.stats()["document_count"],
                          "history_count": store.stats()["history_count"],
                          "javascript_errors": errors, "replies": replies}
                browser.close()
            (OUTPUT / "browser-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)


if __name__ == "__main__":
    main()
