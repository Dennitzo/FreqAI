"""Read-only browser verification of the installed dashboard on localhost."""

from datetime import datetime, timezone
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    output = Path(__file__).resolve().parent
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1180})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8765")
        expect(page.locator("#ask-heading")).to_contain_text("Alltagsgespräch")
        before = page.evaluate("fetch('/api/memory').then(r => r.json())")
        initial = page.evaluate("fetch('/api/state').then(r => r.json())")
        expect(page.locator("#documents")).to_have_text(str(before["document_count"]))
        page.locator("#memory-details summary").click()
        expect(page.locator("#memory-rows tr")).to_have_count(before["document_count"])
        documents = page.evaluate("fetch('/api/documents').then(r => r.json())")
        assert before["document_count"] >= 120
        assert all(row["prompt"] for row in documents["documents"])
        page.wait_for_timeout(350)
        final = page.evaluate("fetch('/api/state').then(r => r.json())")
        after = page.evaluate("fetch('/api/memory').then(r => r.json())")
        assert final["ticks"] > initial["ticks"]
        assert final["time_s"] > initial["time_s"]
        assert final["displacement"] != initial["displacement"]
        assert after["history_count"] == before["history_count"]
        assert after["revision"] == before["revision"] == final["live_revision"]
        assert final["error"] is None and final["running"]
        assert not errors
        page.screenshot(path=str(output / "active_dashboard.png"), full_page=True)
        report = {"checked_at": datetime.now(timezone.utc).isoformat(),
                  "url": page.url, "memory": after, "all_active_documents_are_dialogue_pairs": True,
                  "no_writes": True, "ticks_advanced": final["ticks"] - initial["ticks"],
                  "simulation_time_advanced_seconds": final["time_s"] - initial["time_s"],
                  "wave_changed": True, "javascript_errors": errors}
        browser.close()
    (output / "active_dashboard_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
