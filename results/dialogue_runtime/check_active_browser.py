"""Verify the deployed dashboard; writes only ordinary isolated chat history."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from playwright.sync_api import expect, sync_playwright


OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]


def database_snapshot(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        documents = connection.execute("SELECT id,prompt,text,source FROM documents ORDER BY id").fetchall()
        archive_count = connection.execute("SELECT COUNT(*) FROM document_archive").fetchone()[0]
        history_count = connection.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {"documents_sha256": hashlib.sha256(json.dumps(documents, ensure_ascii=False).encode()).hexdigest(),
                "document_count": len(documents), "archive_count": archive_count,
                "history_count": history_count, "integrity_check": integrity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["V1", "V2"], default="V1")
    args = parser.parse_args()
    baseline_name = "before-deployment-v2.json" if args.stage == "V2" else "before-deployment.json"
    baseline = json.loads((OUTPUT / baseline_name).read_text(encoding="utf-8"))
    backup = database_snapshot(baseline["backup"])
    active_before = database_snapshot(PROJECT / "memory/memory.sqlite3")
    assert backup["documents_sha256"] == active_before["documents_sha256"] == baseline["documents_sha256"]
    assert backup["archive_count"] == active_before["archive_count"] == baseline["counts"]["document_archive"]
    errors, replies = [], []
    old_session_check = None
    v2_checks = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1160})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8765")
        expect(page.locator("#ask-button")).to_be_enabled()
        expect(page.locator("#connection")).to_contain_text("Berechnung läuft")
        initial = page.evaluate("fetch('/api/state').then(r => r.json())")
        original_session = page.evaluate("sessionStorage.getItem('freqai.conversation.id')")
        if args.stage == "V2":
            old_report = json.loads((OUTPUT / "V1/active-browser-report.json").read_text(encoding="utf-8"))
            old_session = old_report["replies"][0]["session_id"]
            old_expected = [turn for turn in old_report["replies"] if turn["session_id"] == old_session]
            old_history = page.evaluate("session => fetch('/api/conversation?session_id=' + session).then(r => r.json())", old_session)
            assert old_history["conversation_revision"] == len(old_expected)
            assert [(turn["prompt"], turn["answer"]) for turn in old_history["turns"]] == [
                (turn["prompt"], turn["answer"]) for turn in old_expected]
            old_session_check = {"session_id": old_session, "preserved_turns": len(old_expected),
                                 "all_prompts_and_answers_unchanged": True}

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
            replies.append({"prompt": prompt, "answer": result["answer"], "abstained": result["abstained"],
                            "generated": result.get("generated"), "method": result.get("method"),
                            "interpretation": result.get("interpretation"),
                            "response_acts": result.get("response_acts"),
                            "session_id": result["session_id"], "conversation_revision": result["conversation_revision"]})
            return result

        for prompt in ["Hallo", "Mir geht es gut, wie geht es dir denn?", "gut und dir?"]:
            assert not ask(page, prompt)["abstained"]
        page.reload()
        expect(page.locator("#ask-button")).to_be_enabled()
        expect(page.locator(".chat-message")).to_have_count(6)
        assert page.evaluate("sessionStorage.getItem('freqai.conversation.id')") == original_session
        reciprocal = ask(page, "Und dir?")
        assert not reciprocal["abstained"] and "bereit" in reciprocal["answer"].lower()
        assert reciprocal["conversation_revision"] == 4
        page.locator("#conversation").evaluate("element => { element.style.maxHeight = 'none'; element.scrollTop = 0; }")
        page.screenshot(path=str(OUTPUT / "active-reported-dialogue.png"), full_page=True)
        page.locator("#conversation").evaluate("element => element.style.maxHeight = ''")

        assert not ask(page, "Ich heiße Elena.")["abstained"]
        assert "Elena" in ask(page, "Wie heiße ich?")["answer"]
        other = context.new_page()
        other.on("pageerror", lambda error: errors.append(str(error)))
        other.goto("http://127.0.0.1:8765")
        expect(other.locator("#ask-button")).to_be_enabled()
        expect(other.locator(".chat-message")).to_have_count(0)
        other_session = other.evaluate("sessionStorage.getItem('freqai.conversation.id')")
        assert other_session != original_session
        assert "Elena" not in ask(other, "Wie heiße ich?")["answer"]
        assert not ask(other, "Ich heiße Sofia.")["abstained"]
        assert "Sofia" in ask(other, "Wie heiße ich?")["answer"]
        recall = ask(page, "Wie heiße ich?")["answer"]
        assert "Elena" in recall and "Sofia" not in recall
        for session, absent in [(original_session, "Sofia"), (other_session, "Elena")]:
            history = page.evaluate("session => fetch('/api/conversation?session_id=' + session).then(r => r.json())", session)
            assert absent not in json.dumps(history, ensure_ascii=False)

        if args.stage == "V2":
            negative = ask(page, "Gut geht es mir nicht, dir vielleicht?")
            assert not negative["abstained"]
            assert any(act["kind"] == "mood_statement" and act["negated"]
                       for act in negative["interpretation"]["acts"])
            assert "assistant_status" in negative["response_acts"]
            assert "klingt schön" not in negative["answer"].lower()
            correction = ask(page, "Mir geht es schlecht, nein, eigentlich gut.")
            assert not correction["abstained"]
            moods = [act for act in correction["interpretation"]["acts"] if act["kind"] == "mood_statement"]
            assert len(moods) == 1 and moods[0]["value"] == "positive" and not moods[0]["negated"]
            assert "klingt schön" in correction["answer"].lower()
            assert not ask(page, "Nenn mich bitte Mara.")["abstained"]
            changed = ask(page, "Wie heiße ich?")["answer"]
            assert "Mara" in changed and "Elena" not in changed
            unchanged_other = ask(other, "Wie heiße ich?")["answer"]
            assert "Sofia" in unchanged_other and "Mara" not in unchanged_other
            other_history = other.evaluate("session => fetch('/api/conversation?session_id=' + session).then(r => r.json())", other_session)
            assert "Mara" not in json.dumps(other_history, ensure_ascii=False)
            page.locator("#conversation").evaluate("element => { element.style.maxHeight = 'none'; element.scrollTop = 0; }")
            page.screenshot(path=str(OUTPUT / "active-v2-dialogue-details.png"), full_page=True)
            page.locator("#conversation").evaluate("element => element.style.maxHeight = ''")
            v2_checks = {"negation_with_reciprocal_handled": True,
                         "self_correction_replaces_old_mood": True,
                         "nenn_mich_changes_name_only_in_own_session": True}

        page.locator("#new-conversation").click()
        expect(page.locator(".chat-message")).to_have_count(0)
        assert page.evaluate("sessionStorage.getItem('freqai.conversation.id')") not in {original_session, other_session}
        assert ask(page, "Und dir?")["abstained"]
        assert "Elena" not in ask(page, "Wie heiße ich?")["answer"]
        final = page.evaluate("fetch('/api/state').then(r => r.json())")
        assert final["ticks"] > initial["ticks"] and final["time_s"] > initial["time_s"]
        assert final["displacement"] != initial["displacement"]
        assert final["running"] and final["error"] is None
        assert final["document_count"] == initial["document_count"] == 120
        assert final["live_revision"] == initial["live_revision"]
        assert not errors
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(OUTPUT / "active-conversation-mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        browser.close()
    active_after = database_snapshot(PROJECT / "memory/memory.sqlite3")
    assert active_after["documents_sha256"] == active_before["documents_sha256"] == backup["documents_sha256"]
    assert active_after["archive_count"] == active_before["archive_count"] == backup["archive_count"]
    assert active_after["integrity_check"] == "ok"
    assert active_after["history_count"] >= active_before["history_count"] + len(replies)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "deployment_stage": args.stage,
              "url": "http://127.0.0.1:8765", "reported_three_turn_dialogue_passed": True,
              "reload_restored_history_and_context": True, "contextual_reciprocal_after_reload_passed": True,
              "tabs_have_independent_sessions": True, "session_histories_have_no_name_leakage": True,
              "new_conversation_resets_context": True, "javascript_errors": errors,
              "v1_session_after_restart": old_session_check, "v2_checks": v2_checks,
              "wave_advanced_ticks": final["ticks"] - initial["ticks"],
              "wave_advanced_seconds": final["time_s"] - initial["time_s"],
              "live_revision_unchanged": final["live_revision"], "backup": backup,
              "active_before": active_before, "active_after": active_after,
              "documents_and_archive_unchanged": True, "ordinary_test_queries": len(replies), "replies": replies}
    (OUTPUT / "active-browser-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
