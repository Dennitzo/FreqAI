"""Focused UI contracts for information-only ingestion and one text decoder."""
from html.parser import HTMLParser
from pathlib import Path
import re


DASHBOARD = Path(__file__).resolve().parents[1] / "freqai/dashboard.html"
HTML = DASHBOARD.read_text(encoding="utf-8")


class DashboardStructure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = {}
        self.forms = {}
        self.labels = []
        self.headers = []
        self.form = None
        self.header = None
        self.script = ""
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            assert attrs["id"] not in self.ids, "Duplicate DOM identifier"
            self.ids[attrs["id"]] = {"tag": tag, **attrs}
        if tag == "form":
            self.form = attrs["id"]
            self.forms[self.form] = []
        if self.form and tag in {"input", "textarea", "select", "button"}:
            self.forms[self.form].append(attrs.get("id"))
        if tag == "label":
            self.labels.append(attrs.get("for"))
        if tag == "th":
            self.header = ""
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "form":
            self.form = None
        if tag == "th":
            self.headers.append(self.header)
            self.header = None
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.header is not None:
            self.header += data
        if self.in_script:
            self.script += data


STRUCTURE = DashboardStructure()
STRUCTURE.feed(HTML)


def payload_keys(endpoint):
    match = re.search(r"request\('" + re.escape(endpoint) + r"',\s*\{([^{}]*)\}\)", STRUCTURE.script)
    assert match, f"No explicit request payload for {endpoint}"
    return set(re.findall(r"\b([a-z_]+)\s*:", match.group(1)))


def test_removed_fields_leave_no_broken_dom_or_label_references():
    references = set(re.findall(r"byId\('([^']+)'\)", STRUCTURE.script))
    assert references <= set(STRUCTURE.ids)
    assert set(STRUCTURE.labels) <= set(STRUCTURE.ids)
    assert not {"generation-mode", "cue", "matches"}.intersection(STRUCTURE.ids)


def test_knowledge_form_submits_only_information_fields():
    assert set(STRUCTURE.forms["ingest-form"]) == {"source", "document", "ingest-button"}
    assert payload_keys("/api/documents") == {"text", "source"}
    assert "document.prompt" not in STRUCTURE.script


def test_chat_uses_the_single_server_default_and_ignores_old_mode_storage():
    assert payload_keys("/api/ask") == {"prompt", "session_id"}
    assert "freqai.generation.mode" not in STRUCTURE.script
    assert "generation-mode" not in STRUCTURE.script
    assert not any(element["tag"] == "select" for element in STRUCTURE.ids.values())


def test_storage_has_two_information_columns_and_no_prompt_column():
    assert len(STRUCTURE.headers) == 2
    assert "Quelle" in STRUCTURE.headers[0]
    assert "Information" in STRUCTURE.headers[1]
    assert not any("Gesprächsanlass" in header for header in STRUCTURE.headers)


def test_knowledge_example_is_declarative_information_instead_of_a_chat_reply():
    example = STRUCTURE.ids["document"]["placeholder"]
    assert example.endswith(".") and "?" not in example
    assert "frequenz" in example.casefold() and "hertz" in example.casefold()
    assert "hallo" not in example.casefold()


def test_response_details_have_one_decoder_display_without_legacy_match_cards():
    assert {"answer-method", "token-details", "token-trace"} <= set(STRUCTURE.ids)
    assert "result.decoder" in STRUCTURE.script
    assert "result.matches" not in STRUCTURE.script
    assert "result.generated" not in STRUCTURE.script
    assert "match-header" not in HTML
    assert "Regeldialog" not in HTML


def test_history_context_and_pagination_controls_remain_available():
    assert {"new-conversation", "conversation", "context-details", "context-wave",
            "memory-details", "memory-rows", "memory-previous", "memory-next"} <= set(STRUCTURE.ids)
    assert "freqai.conversation.id" in STRUCTURE.script
    assert "/api/conversation?session_id=" in STRUCTURE.script
    assert re.search(r"documentPageSize\s*=\s*50\b", STRUCTURE.script)
    assert "result.next_offset" in STRUCTURE.script and "result.previous_offset" in STRUCTURE.script


def test_user_can_identify_the_grammar_and_unpaired_data_basis():
    assert "explizite Grammatikregeln" in HTML
    assert "keine Frage-Antwort-Trainingspaare" in HTML
