"""Finish read-only checks for the dedicated session of the first browser run.

No chat POST is repeated. Energy is compared at a fixed conversation revision;
the first harness attempt compared unversioned UI frames after a new answer.
"""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
SESSION = "a2106535-2fbb-48e6-9cc2-9a6025d31822"
URL = "http://127.0.0.1:8765"


def run():
    answers = json.loads((OUT / "browser.final.answers.pending.json").read_text(encoding="utf-8"))
    assert len(answers) == 3 and all(row["session_id"] == SESSION for row in answers)
    assert all(step.get("carrier_bins") == 44800 for answer in answers for step in answer["decoder"]["steps"])
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.set_default_timeout(300000)
        page.add_init_script("sessionStorage.setItem('freqai.conversation.id', " + json.dumps(SESSION) + ");")
        errors, document_requests, posts = [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: document_requests.append(request.url) if "/api/documents" in request.url and request.method == "GET" else None)
        page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)
        page.goto(URL)
        page.wait_for_function("() => currentState && currentState.conversation_revision === 3 && document.querySelectorAll('.chat-message.assistant').length === 3")
        state_before = page.evaluate("() => currentState")
        assert state_before["session_id"] == SESSION and state_before["document_count"] == 12000
        assert page.locator("#memory-rows tr").count() == 0
        page.locator("#memory-details summary").click()
        page.wait_for_function("() => document.querySelectorAll('#memory-rows tr').length === 50")
        first_rows = page.locator("#memory-rows tr td:first-child").all_text_contents()
        first_summary = page.locator("#memory-page-summary").inner_text()
        page.locator("#memory-next").click()
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('51–100')")
        second_rows = page.locator("#memory-rows tr td:first-child").all_text_contents()
        second_summary = page.locator("#memory-page-summary").inner_text()
        assert not set(first_rows).intersection(second_rows)
        page.locator("#memory-previous").click()
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('1–50')")
        assert first_rows == page.locator("#memory-rows tr td:first-child").all_text_contents()
        page.evaluate("() => refreshDocuments(650)")
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('651–700')")
        source = page.locator("#memory-rows tr td:first-child").first.inner_text()
        assert "OpenAssistant/oasst2" in source and "Apache-2.0" in source
        page.evaluate("() => refreshDocuments(11950)")
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('11.951–12.000')")
        last_summary = page.locator("#memory-page-summary").inner_text()
        assert page.locator("#memory-next").is_disabled()
        page.reload()
        page.wait_for_function("() => currentState && currentState.conversation_revision === 3 && document.querySelectorAll('.chat-message.assistant').length === 3")
        assert page.evaluate("() => sessionStorage.getItem('freqai.conversation.id')") == SESSION
        if not page.locator("#memory-details").evaluate("element => element.open"):
            page.locator("#memory-details summary").click()
        page.wait_for_function("() => document.querySelector('#memory-page-summary').textContent.startsWith('1–50')")
        page.locator("#memory-details").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / "browser.final.png"))
        page.locator("#context-details").evaluate("element => element.open = true")
        page.wait_for_function("() => currentState.conversation_revision === 3 && currentState.context_wave && currentState.context_wave.energy > 0")
        first_wave = page.evaluate("() => currentState.context_wave")
        page.wait_for_function("before => currentState.conversation_revision === 3 && currentState.context_wave.time_s > before + .3", arg=first_wave["time_s"])
        second_wave = page.evaluate("() => currentState.context_wave")
        assert first_wave["displacement"] != second_wave["displacement"]
        assert abs(first_wave["energy"] - second_wave["energy"]) < 1e-8 * max(1, first_wave["energy"])
        assert second_wave["carrier_size"] == 44800 and second_wave["mode_count"] == 44614
        page.locator("#context-details").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / "browser.final.context.png"))
        page.locator("#ask-heading").scroll_into_view_if_needed()
        page.screenshot(path=str(OUT / "browser.final.chat.png"))
        page.screenshot(path=str(OUT / "browser.png"))
        process = json.loads((ROOT / "runtime/server-process.json").read_text(encoding="utf-8-sig"))
        assert process["pid"] == 29008
        assert not errors and not posts
        report = {
            "stage": "final", "url": URL, "session_id": SESSION,
            "new_empty_audit_session": True, "session_origin": "Fresh isolated browser session from initial audit; read-only continuation reused only that session",
            "document_count": 12000, "live_revision": state_before["live_revision"],
            "maximum_rendered_document_rows": 50, "first_page": first_summary,
            "second_page": second_summary, "last_page": last_summary,
            "first_and_second_disjoint": True, "previous_page_identical": True,
            "reload_resets_page_zero_preserves_session": True, "audit_conversation_survives_reload": True,
            "source": source, "document_requests": document_requests, "source_data_mutations": 0,
            "chat_posts_initial_run": 3, "chat_posts_read_only_continuation": 0,
            "answers": answers, "javascript_errors": errors, "server_process": process,
            "decoder_carrier_bins": 44800,
            "context_wave": {"conversation_revision": 3, "time_before": first_wave["time_s"], "time_after": second_wave["time_s"],
                             "energy_before": first_wave["energy"], "energy_after": second_wave["energy"],
                             "mode_count": second_wave["mode_count"], "carrier_size": second_wave["carrier_size"],
                             "motion_verified": True, "energy_stable": True},
            "harness_correction": "First browser attempt compared unversioned UI frames immediately after a new answer and failed its energy assertion. The completed check explicitly waits for conversation_revision=3 and compares two frames at that same revision. No application code was changed.",
        }
        for name in ("browser.final.json", "browser.json"):
            (OUT / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "answers"}, ensure_ascii=True, indent=2))
        print(json.dumps([{"prompt": row["prompt"], "answer": row["answer"]} for row in answers], ensure_ascii=True, indent=2))
        browser.close()


if __name__ == "__main__":
    run()
