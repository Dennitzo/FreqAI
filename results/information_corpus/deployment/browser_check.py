"""Audit the deployed information UI using an isolated browser conversation.

Run only after deployment approval, with its exact process ID. This harness
never submits the ingestion form or imports source data. Three audit prompts
create only the explicitly requested new audit conversation/history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
URL = "http://127.0.0.1:8765"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CORPUS = ROOT / "memory/imports/information_wikipedia_20000_v3.jsonl"
CORPUS_SHA = "2f5b876d904346ca4b9082e05ff5e4bc5e8a729f2a479c777f47e1332a0a5674"


def process_metadata():
    return json.loads((ROOT / "runtime/server-process.json").read_text(encoding="utf-8-sig"))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(expected_pid, expected_documents=32000, timeout_seconds=650):
    deployed = process_metadata()
    assert deployed["pid"] == expected_pid, "Deployment process differs from explicit audit target"
    assert hashlib.sha256(CORPUS.read_bytes()).hexdigest() == CORPUS_SHA
    report = {"url": URL, "server_process": deployed, "expected_document_count": expected_documents,
              "corpus_sha256": CORPUS_SHA, "source_data_mutations": 0,
              "answers": [], "javascript_errors": [], "console_errors": [],
              "document_requests": [], "posts": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=EDGE, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.set_default_timeout(timeout_seconds * 1000)
        page.on("pageerror", lambda error: report["javascript_errors"].append(str(error)))
        page.on("console", lambda message: report["console_errors"].append(message.text)
                if message.type == "error" else None)
        page.on("request", lambda request: report["document_requests"].append(request.url)
                if request.method == "GET" and "/api/documents" in request.url else None)
        page.on("request", lambda request: report["posts"].append(request.url)
                if request.method == "POST" else None)
        try:
            page.goto(URL)
            page.wait_for_function("count => currentState && currentState.document_count === count && knownRevision !== null",
                                   arg=expected_documents)
            page.wait_for_function("() => !document.querySelector('#ask-button').disabled")
            session = page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')")
            state = page.evaluate("() => currentState")
            assert state["conversation_revision"] == 0
            assert page.locator(".chat-message").count() == 0
            assert page.locator("#generation-mode").input_value() == "wave"
            assert page.locator("#memory-rows tr").count() == 0
            report.update({"session_id": session, "new_empty_audit_session": True,
                           "initial_live_revision": state["live_revision"],
                           "initial_state": state})
            page.locator("#memory-details summary").click()
            page.wait_for_function("() => document.querySelectorAll('#memory-rows tr').length === 50")
            first = page.locator("#memory-rows tr td:first-child").all_text_contents()
            report["first_page"] = page.locator("#memory-page-summary").inner_text()
            assert page.locator("#memory-previous").is_disabled()
            page.locator("#memory-next").click()
            page.wait_for_function("() => documentOffset === 50 && !document.querySelector('#memory-previous').disabled")
            second = page.locator("#memory-rows tr td:first-child").all_text_contents()
            assert len(second) == 50 and not set(first).intersection(second)
            report["second_page"] = page.locator("#memory-page-summary").inner_text()
            page.locator("#memory-previous").click()
            page.wait_for_function("() => documentOffset === 0 && document.querySelector('#memory-previous').disabled")
            assert page.locator("#memory-rows tr td:first-child").all_text_contents() == first
            report["previous_page_identical"] = True
            last_offset = ((expected_documents - 1) // 50) * 50
            page.evaluate("offset => refreshDocuments(offset)", last_offset)
            page.wait_for_function("offset => documentOffset === offset && document.querySelector('#memory-next').disabled",
                                   arg=last_offset)
            assert page.locator("#memory-rows tr").count() == expected_documents - last_offset
            report["last_page"] = page.locator("#memory-page-summary").inner_text()
            report["source_example"] = page.locator("#memory-rows tr td:first-child").first.inner_text()
            assert "Wikimedia Wikipedia 20231101.de" in report["source_example"]
            assert "https://de.wikipedia.org/wiki/" in report["source_example"]
            assert set(page.locator("#memory-rows tr td:nth-child(2)").all_text_contents()) == {"—"}
            report["maximum_rendered_document_rows"] = 50
            report["information_prompts_empty_in_last_page"] = True
            page.locator("#memory-details").scroll_into_view_if_needed()
            page.screenshot(path=str(OUT / "browser.last-page.png"))
            save("browser.pending.json", report)
            for prompt in ("Hallo", "Was ist eine Frequenz?", "Und welche Einheit hat sie?"):
                page.locator("#prompt").fill(prompt)
                started = time.perf_counter()
                with page.expect_response(lambda response: response.url.endswith("/api/ask") and
                                          response.request.method == "POST", timeout=timeout_seconds * 1000) as response:
                    page.locator("#ask-button").click()
                result = response.value.json()
                assert response.value.status == 200, result
                assert result["session_id"] == session
                assert result["wave_generation"] and result["matches"] == []
                page.wait_for_function("() => !document.querySelector('#ask-button').disabled")
                report["answers"].append({"prompt": prompt, "seconds": time.perf_counter() - started,
                                          "result": result})
                save("browser.pending.json", report)
                print(json.dumps({"prompt": prompt, "answer": result["answer"],
                                  "seconds": report["answers"][-1]["seconds"],
                                  "method": result["method"]}, ensure_ascii=True), flush=True)
            revision = report["answers"][-1]["result"]["conversation_revision"]
            page.locator("#context-details").evaluate("element => element.open = true")
            page.wait_for_function("revision => currentState.conversation_revision === revision && currentState.context_wave && currentState.context_wave.energy > 0",
                                   arg=revision)
            first_wave_state = page.evaluate("() => currentState")
            first_wave = first_wave_state["context_wave"]
            page.wait_for_function("before => currentState.conversation_revision === before.revision && currentState.context_wave.time_s > before.time + .3",
                                   arg={"revision": revision, "time": first_wave["time_s"]})
            second_wave_state = page.evaluate("() => currentState")
            second_wave = second_wave_state["context_wave"]
            assert first_wave_state["live_revision"] == second_wave_state["live_revision"]
            assert first_wave["displacement"] != second_wave["displacement"]
            assert abs(first_wave["energy"] - second_wave["energy"]) < 1e-8 * max(1, first_wave["energy"])
            report["context_wave"] = {"conversation_revision": revision, "motion_verified": True,
                                      "energy_stable_at_same_revision": True,
                                      "before": first_wave, "after": second_wave}
            page.locator("#context-details").scroll_into_view_if_needed()
            page.screenshot(path=str(OUT / "browser.context.png"))
            page.reload()
            page.wait_for_function("revision => currentState && currentState.conversation_revision === revision && document.querySelectorAll('.chat-message.assistant').length === 3",
                                   arg=revision)
            assert page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')") == session
            displayed = page.locator(".chat-message.assistant span:last-child").all_text_contents()
            assert displayed == [row["result"]["answer"] or "Keine Textausgabe berechnet."
                                 for row in report["answers"]]
            page.locator("#memory-details summary").click()
            page.wait_for_function("() => documentOffset === 0 && document.querySelectorAll('#memory-rows tr').length === 50")
            assert page.locator("#memory-rows tr td:first-child").all_text_contents() == first
            final_state = page.evaluate("() => currentState")
            assert final_state["document_count"] == expected_documents
            assert final_state["live_revision"] == state["live_revision"]
            assert final_state["time_s"] > state["time_s"]
            assert process_metadata()["pid"] == expected_pid
            assert report["posts"] == [URL + "/api/ask"] * 3
            assert not report["javascript_errors"], report["javascript_errors"]
            report.update({"document_count": final_state["document_count"],
                           "live_revision": final_state["live_revision"],
                           "clock_advances": True, "audit_conversation_survives_reload": True,
                           "reload_resets_page_zero": True, "result": "passed",
                           "answer_quality_scope": "Actual outputs recorded; browser functionality does not establish general answer quality."})
            page.locator("#ask-heading").scroll_into_view_if_needed()
            page.screenshot(path=str(OUT / "browser.png"))
            save("browser.json", report)
            print(json.dumps({"result": "passed", "session_id": session, "pid": expected_pid,
                              "document_count": expected_documents, "report": str(OUT / "browser.json")}), flush=True)
        except Exception as error:
            report["failure"] = f"{type(error).__name__}: {error}"
            save("browser.failure.json", report)
            page.screenshot(path=str(OUT / "browser.failure.png"))
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--expected-documents", type=int, default=32000)
    parser.add_argument("--timeout-seconds", type=int, default=650)
    args = parser.parse_args()
    run(args.expected_pid, args.expected_documents, args.timeout_seconds)
