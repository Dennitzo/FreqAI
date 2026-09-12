"""Browser audit of the deployed service; never imports or changes source data.

Pagination stage is read-only. Full stage uses a fresh browser/session for three
explicit audit prompts, never a pre-existing user conversation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
URL = "http://127.0.0.1:8765"


def run(stage):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(300000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        document_requests = []
        page.on("request", lambda request: document_requests.append(request.url) if "/api/documents" in request.url and request.method == "GET" else None)
        page.goto(URL)
        page.wait_for_function("() => currentState && currentState.document_count === 12000 && knownRevision !== null")
        session = page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')")
        assert page.locator(".chat-message").count() == 0
        state_before = page.evaluate("() => currentState")
        assert page.locator("#memory-rows tr").count() == 0
        page.locator("#memory-details summary").click()
        page.wait_for_function("() => document.querySelectorAll('#memory-rows tr').length === 50")
        first_rows = page.locator("#memory-rows tr td:first-child").all_text_contents()
        first_summary = page.locator("#memory-page-summary").inner_text()
        page.locator("#memory-next").click()
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('51–100')")
        second_rows = page.locator("#memory-rows tr td:first-child").all_text_contents()
        assert not set(first_rows).intersection(second_rows)
        second_summary = page.locator("#memory-page-summary").inner_text()
        page.locator("#memory-previous").click()
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('1–50')")
        assert first_rows == page.locator("#memory-rows tr td:first-child").all_text_contents()
        page.evaluate("() => refreshDocuments(650)")
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('651–700')")
        source = page.locator("#memory-rows tr td:first-child").first.inner_text()
        assert "OpenAssistant/oasst2" in source and "Apache-2.0" in source
        page.evaluate("() => refreshDocuments(11950)")
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('11.951–12.000')")
        assert page.locator("#memory-next").is_disabled()
        assert page.locator("#memory-rows tr").count() == 50
        last_summary = page.locator("#memory-page-summary").inner_text()
        page.reload()
        page.wait_for_function("() => currentState && currentState.document_count === 12000")
        assert page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')") == session
        if not page.locator("#memory-details").evaluate("element => element.open"):
            page.locator("#memory-details summary").click()
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('1–50')")
        page.wait_for_function("time => currentState.time_s > time + .2", arg=state_before["time_s"])
        state_after = page.evaluate("() => currentState")
        assert state_after["live_revision"] == state_before["live_revision"]
        page.locator("#memory-details").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / f"browser.{stage}.png"))
        report = {
            "stage": stage, "url": URL, "session_id": session, "new_empty_audit_session": True,
            "document_count": state_after["document_count"], "live_revision": state_after["live_revision"],
            "maximum_rendered_document_rows": 50, "first_page": first_summary,
            "second_page": second_summary, "last_page": last_summary,
            "first_and_second_disjoint": True, "previous_page_identical": True,
            "reload_resets_page_zero_preserves_session": True,
            "source": source, "time_before": state_before["time_s"], "time_after": state_after["time_s"],
            "clock_advances": True, "document_requests": document_requests, "source_data_mutations": 0,
            "server_process": json.loads((ROOT / "runtime/server-process.json").read_text(encoding="utf-8-sig")),
        }
        (OUT / f"browser.{stage}.pending.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if stage == "final":
            answers = []
            for prompt in ("Hallo", "Mir geht es gut, wie geht es dir denn?", "Danke."):
                page.locator("#prompt").fill(prompt)
                started = time.perf_counter()
                with page.expect_response(lambda response: response.url.endswith("/api/ask") and response.request.method == "POST", timeout=300000) as response:
                    page.locator("#ask-button").click()
                result = response.value.json()
                assert response.value.status == 200, result
                assert result["wave_generation"] and result["matches"] == []
                assert result["decoder"]["steps"] and result["session_id"] == session
                assert all(step.get("carrier_bins") == 44800 for step in result["decoder"]["steps"]), "Expected final padded FFT carrier"
                page.wait_for_function("() => !document.querySelector('#ask-button').disabled")
                answers.append({"prompt": prompt, "answer": result["answer"], "seconds": time.perf_counter()-started,
                                "session_id": result["session_id"], "live_revision": result["live_revision"],
                                "conversation_revision": result["conversation_revision"],
                                "decoder": result["decoder"], "abstained": result["abstained"]})
                (OUT / "browser.final.answers.pending.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report["answers"] = answers
            page.locator("#context-details").evaluate("element => element.open = true")
            page.wait_for_function("revision => currentState && currentState.conversation_revision === revision && currentState.context_wave && currentState.context_wave.energy > 0", arg=result["conversation_revision"])
            first_wave = page.evaluate("() => currentState.context_wave")
            page.wait_for_function("before => currentState.context_wave.time_s > before + .2", arg=first_wave["time_s"])
            assert page.evaluate("() => currentState.conversation_revision") == result["conversation_revision"]
            second_wave = page.evaluate("() => currentState.context_wave")
            assert first_wave["displacement"] != second_wave["displacement"]
            assert abs(first_wave["energy"] - second_wave["energy"]) < 1e-8 * max(1, first_wave["energy"])
            report["context_wave"] = {"time_before": first_wave["time_s"], "time_after": second_wave["time_s"],
                                      "energy_before": first_wave["energy"], "energy_after": second_wave["energy"],
                                      "mode_count": second_wave["mode_count"], "carrier_size": second_wave.get("carrier_size"),
                                      "motion_verified": True,
                                      "energy_stable": True}
            page.reload()
            page.wait_for_function("() => document.querySelectorAll('.chat-message.assistant').length === 3")
            assert page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')") == session
            report["audit_conversation_survives_reload"] = True
            report["server_process"] = json.loads((ROOT / "runtime/server-process.json").read_text(encoding="utf-8-sig"))
            assert report["server_process"]["pid"] == 29008, "Unexpected deployed process during final audit"
            page.locator("#ask-heading").scroll_into_view_if_needed()
            page.screenshot(path=str(OUT / "browser.final.chat.png"))
            page.screenshot(path=str(OUT / "browser.png"))
        assert not errors, errors
        report["javascript_errors"] = errors
        (OUT / f"browser.{stage}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if stage == "final":
            (OUT / "browser.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "answers"}, ensure_ascii=True, indent=2))
        if "answers" in report:
            print(json.dumps([{key: value for key, value in answer.items() if key != "decoder"} for answer in report["answers"]], ensure_ascii=True, indent=2))
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pagination", "final"))
    run(parser.parse_args().stage)
