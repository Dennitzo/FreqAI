"""Temporary probe: resolve the real de.wikipedia article titles behind curated
science terms that were rejected as disambiguation or missing pages.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://de.wikipedia.org/w/api.php"
UA = {"User-Agent": "FreqAI-science-extension-probe/1.0 (one-off title check)"}


def call(params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{API}?{query}", headers=UA)
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    candidates = [
        "Zelle (Biologie)",
        "Quark (Teilchenphysik)",
        "Virus (Biologie)",
        "Max Delbrück (Biophysiker)",
        "Jacques Lucien Monod",
        "Pierre Curie",
        "Elementarteilchen",
        "Thomas Young (Naturwissenschaftler)",
        "Standardmodell",
        "Christiaan Huygens",
        "Léon Foucault",
        "Antonie van Leeuwenhoek",
    ]
    query = "|".join(candidates)
    payload = call({"action": "query", "format": "json", "redirects": 1, "prop": "info",
                    "titles": query})
    pages = payload.get("query", {}).get("pages", {})
    for page_id, page in sorted(pages.items(), key=lambda item: str(item[1].get("title"))):
        print("INFO", page_id, page.get("title"), "missing" if "missing" in page else "ok",
              page.get("ds", ""))
    for term in ("Thomas Young", "Delbrück", "Monod", "Foucault", "Standardmodell"):
        found = call({"action": "opensearch", "format": "json", "search": term, "limit": 6})
        print("SEARCH", term, found[1] if len(found) > 1 else [])
        time.sleep(1.0)
    # Why did the API-side extraction fail for these specific titles?
    import re
    ending = re.compile(r"[.!?][\"'»“”’)]*$")
    empty_field = re.compile(r"\(\s*(?:[,;]\s*)?\)|\(\s*;", re.IGNORECASE)
    for title in ("Pierre Curie", "Christiaan Huygens", "Léon Foucault",
                  "Antonie van Leeuwenhoek", "Elementarteilchen", "Rosalind Franklin"):
        payload = call({"action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
                        "exintro": 1, "exsectionformat": "plain", "titles": title})
        for page in payload.get("query", {}).get("pages", {}).values():
            extract = str(page.get("extract", ""))
            blocks = [b.strip() for b in extract.split("\n") if b.strip()]
            first = blocks[0] if blocks else ""
            print("LEAD", title, "blocks", len(blocks), "lens", [len(b) for b in blocks][:4],
                  "ends", bool(ending.search(first)), "empty_field", bool(empty_field.search(first)),
                  "|", first[:90].encode("ascii", "backslashreplace").decode())
        time.sleep(1.0)


if __name__ == "__main__":
    main()
