"""Download pinned public German text corpora and build auditable import files.

This script never opens the application's SQLite database or evaluation files.
OASST follow-up turns are preserved separately, because the current input encoder
does not understand role-marked conversation histories. GermanQuAD uses TRAIN
only; answer fragments become their original surrounding complete sentences.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "memory/sources"
IMPORTS = ROOT / "memory/imports"
OASST_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
QUAD_REVISION = "a2f3a59f0be843fc305d0417d7292ef0b1a66884"
QUAD_CARD_REVISION = "fff05ceaf2ffbe5b65c7e0c57e678f7b7e1a0581"
DOWNLOADS = {
    "oasst2": {
        "path": SOURCES / "openassistant-oasst2/2023-11-05_oasst2_all.messages.jsonl.gz",
        "url": f"https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/{OASST_REVISION}/2023-11-05_oasst2_all.messages.jsonl.gz",
        "sha256": "820146830e78634170f5a33d79d0b3e5022a7f169ce054886ad1f16e1d53a764",
    },
    "germanquad": {
        "path": SOURCES / "deepset-germanquad/train-0000.parquet",
        "url": f"https://huggingface.co/datasets/deepset/germanquad/resolve/{QUAD_REVISION}/plain_text/train/0000.parquet",
        "sha256": "319334092605ed9ade8a1e6535584debb2e99def474697417a06ad4a455bf0ec",
    },
}
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
GERMAN_MARKERS = set("der die das den dem des ein eine einer eines einem einen und oder aber ist sind war waren wird werden wurde wurden mit auf für von zu zum zur im in an am aus als sich nicht auch nach bei durch wie was wer warum welche welcher welches kann können ich du er sie es wir ihr man nur noch wenn dann diese dieser dieses diese haben hat hatte sein seine ihrer ihrem über unter zwischen vor bis mehr alle dass da um so sehr es".split())
CODE_RE = re.compile(r"```|`[^`]+`|:=|https?://|www\.|<\/?[A-Za-z][^>]*>|\\(?:frac|begin|end|sum)\b|^\s*(?:def |import |function |SELECT )", re.MULTILINE)
END_RE = re.compile(r"[.!?][\"'»“”’)]*$")
ABBREVIATIONS = set("bzw bspw ca dr prof vgl nr art abs st jr sr mr usw sog z b u a d h ggf inkl insb etc str".split())
DEPENDENT_START = re.compile(r"^(?:(?:Dies(?:e[rsnm]?)?|Dazu|Dabei|Dadurch|Dort|Deshalb|Deswegen|Hierbei|Hierzu|Er|Sie|Es|Sein(?:e[rsnm]?)?|Ihr(?:e[rsnm]?)?)\b|Das\s+(?:ist|war|wird|sind|waren|hat|hatte|bedeutet|heißt|führt|führte|kann|könnte|muss)\b)")
ROLEPLAY_RE = re.compile(r"\b(?:du bist|tu(?:e)? so,? als|spiel(?:e)? die rolle|übernimm die rolle|stelle dir vor,? (?:du|dass du))\b", re.IGNORECASE)
SELF_IDENTITY_RE = re.compile(r"\b(?:ich heiße|mein name (?:ist|lautet)|ich bin (?:chatgpt|open\s?assistant|gpt[- ]?\d)\b|ich (?:bin|wurde)[^.?!\n]{0,100}\b(?:sprachmodell|chatbot|künstliche? intelligenz|ki[- ](?:modell|assistent)|trainiert|programmiert|entwickelt))", re.IGNORECASE)
FILTER_CONFIGURATION = {
    "version": 3,
    "oasst_min_answer_quality": 0.5,
    "oasst_max_binary_adverse_label": 0.25,
    "oasst_max_toxicity_or_violence": 0.5,
    "oasst_answer_words": [6, 200],
    "oasst_prompt_words_max": 120,
    "oasst_active_root_prompts_only": True,
    "oasst_plain_prose_only": "Reject explicit lists and unfinished nonempty lines before whitespace normalization",
    "oasst_persona_filter": "Reject role-assignment prompts and foreign first-person assistant identities; objective questions about AI remain eligible",
    "oasst_allowed_tree_states": ["ready_for_export", "growing", "ranking"],
    "quad_split": "train",
    "quad_answer_chars_max": 900,
    "quad_answer_words": [6, 140],
    "quad_question_words": [3, 80],
    "quad_answer_is_verbatim_context_sentences": True,
    "quad_wiki_format_normalization": "Remove repeated apostrophe markers used for Wiki italics/bold; preserve original offsets in provenance",
    "german_check": "author language tag plus German function-word heuristic; no external language model",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_verified(spec: dict, offline: bool = False) -> Path:
    path = spec["path"]
    if not path.exists():
        if offline:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(spec["url"], timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if sha256(temporary) != spec["sha256"]:
            raise ValueError(f"Downloaded checksum mismatch: {temporary}")
        temporary.replace(path)
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Cached source checksum mismatch: {path}")
    return path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def plausible_german(text: str) -> bool:
    words = [w.casefold() for w in WORD_RE.findall(text)]
    if not words:
        return False
    required = 1 if len(words) < 15 else 2
    return sum(w in GERMAN_MARKERS for w in words) >= required


def is_plain_prose(text: str) -> bool:
    if re.search(r"^\s*(?:[-*•#]|\d+[.)])\s", text, flags=re.MULTILINE):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return all(END_RE.search(line) for line in lines)


def has_foreign_persona(prompt: str, answer: str) -> bool:
    if ROLEPLAY_RE.search(prompt) or SELF_IDENTITY_RE.search(answer):
        return True
    # A proper-name-style identity is a source persona, not the local FreqAI.
    return re.search(r"\b[Ii]ch bin (?:der |die )?[A-ZÄÖÜ][a-zäöüß]+(?:[.!?,]|\s+(?:und|dein|ein|eine)\b)", answer) is not None


def label_value(message: dict, label: str, default: float = 0.0) -> float:
    labels = message.get("labels") or {}
    return float((labels.get(label) or {}).get("value", default))


def oasst_message_rejection(message: dict) -> str | None:
    if message.get("lang") != "de":
        return "language_tag"
    if message.get("review_result") is not True or message.get("deleted"):
        return "not_accepted_or_deleted"
    if message.get("synthetic") is not False:
        return "synthetic_or_unknown"
    binary_labels = ("spam", "lang_mismatch", "pii", "not_appropriate", "hate_speech", "sexual_content", "fails_task")
    if any(label_value(message, key) > FILTER_CONFIGURATION["oasst_max_binary_adverse_label"] for key in binary_labels):
        return "adverse_review_label"
    if any(label_value(message, key) > FILTER_CONFIGURATION["oasst_max_toxicity_or_violence"] for key in ("toxicity", "violence")):
        return "toxicity_or_violence_label"
    if CODE_RE.search(message.get("text", "")):
        return "code_or_external_reference"
    if not plausible_german(message.get("text", "")):
        return "language_heuristic"
    return None


def parent_chain(message: dict, by_id: dict[str, dict]) -> list[dict]:
    chain: list[dict] = []
    seen: set[str] = set()
    parent_id = message.get("parent_id")
    expected_role = "prompter"
    while parent_id is not None:
        if parent_id in seen or parent_id not in by_id:
            raise ValueError("Missing parent or cyclic conversation")
        seen.add(parent_id)
        parent = by_id[parent_id]
        if parent.get("role") != expected_role or parent.get("message_tree_id") != message.get("message_tree_id"):
            raise ValueError("Role/tree mismatch")
        chain.append(parent)
        expected_role = "assistant" if expected_role == "prompter" else "prompter"
        parent_id = parent.get("parent_id")
    return list(reversed(chain))


def build_oasst(path: Path) -> tuple[list[dict], list[dict], dict]:
    # Keep German records only; non-German ancestors make a chain ineligible.
    by_id = {}
    total = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            total += 1
            if item.get("lang") == "de":
                by_id[item["message_id"]] = item
    active, followups = [], []
    rejected: Counter = Counter()
    seen = set()
    for message in sorted(by_id.values(), key=lambda item: item["message_id"]):
        if message.get("role") != "assistant":
            continue
        reason = oasst_message_rejection(message)
        if not reason and message.get("tree_state") not in FILTER_CONFIGURATION["oasst_allowed_tree_states"]:
            reason = "unfinished_or_rejected_tree"
        if not reason and label_value(message, "quality", -1.0) < FILTER_CONFIGURATION["oasst_min_answer_quality"]:
            reason = "answer_quality"
        if reason:
            rejected[reason] += 1
            continue
        try:
            chain = parent_chain(message, by_id)
        except ValueError:
            rejected["invalid_parent_chain"] += 1
            continue
        if not chain or any(oasst_message_rejection(parent) for parent in chain):
            rejected["parent_quality"] += 1
            continue
        prompt, answer = normalize(chain[-1]["text"]), normalize(message["text"])
        if has_foreign_persona(prompt, answer):
            rejected["foreign_persona_or_role_assignment"] += 1
            continue
        if not 6 <= len(WORD_RE.findall(answer)) <= 200 or len(WORD_RE.findall(prompt)) > 120 or len(answer) > 1600:
            rejected["length"] += 1
            continue
        if not END_RE.search(answer):
            rejected["unfinished_answer"] += 1
            continue
        if not is_plain_prose(message["text"]):
            rejected["list_or_unfinished_line"] += 1
            continue
        key = (prompt.casefold(), answer.casefold())
        if key in seen:
            rejected["duplicate_pair"] += 1
            continue
        seen.add(key)
        provenance = {
            "dataset": "OpenAssistant/oasst2", "revision": OASST_REVISION,
            "license": "Apache-2.0", "language": "de", "synthetic": False,
            "message_id": message["message_id"], "parent_id": message["parent_id"],
            "message_tree_id": message["message_tree_id"], "tree_state": message["tree_state"],
            "created_date": message.get("created_date"), "review_count": message.get("review_count"),
            "quality": label_value(message, "quality"), "rank": message.get("rank"),
            "labels": message.get("labels"),
            "parent_chain": [{key: parent.get(key) for key in ("message_id", "parent_id", "role", "text")} for parent in chain],
            "normalization": "Unicode NFC and whitespace only; no generated or paraphrased answer",
            "raw_source_sha256": DOWNLOADS["oasst2"]["sha256"],
        }
        row = {
            "id": f"oasst2-de-{message['message_id']}", "prompt": prompt, "text": answer,
            "source": f"https://huggingface.co/datasets/OpenAssistant/oasst2 | Apache-2.0 | revision={OASST_REVISION} | message={message['message_id']}",
            "provenance": provenance,
        }
        if len(chain) == 1:
            active.append(row)
        else:
            row["import_status"] = "inactive: requires a role-aware conversation encoder"
            followups.append(row)
    return active, followups, {"raw_messages": total, "german_messages": len(by_id), "active_root_pairs": len(active), "inactive_followup_pairs": len(followups), "rejections": dict(sorted(rejected.items()))}


@lru_cache(maxsize=8192)
def sentence_spans(context: str) -> list[tuple[int, int]]:
    """Conservative original-character sentence boundaries, without a model."""
    boundaries = {0, len(context)}
    for match in re.finditer(r"\n+", context):
        boundaries.update((match.start(), match.end()))
    for match in re.finditer(r"[.!?][\"'»“”’)]*(?=\s|$)", context):
        if context[match.start()] == ".":
            previous = re.search(r"([\w]+)$", context[:match.start()])
            token = previous.group(1).casefold() if previous else ""
            if token.isdigit() or len(token) == 1 or token in ABBREVIATIONS:
                continue
        boundaries.add(match.end())
    ordered = sorted(boundaries)
    spans = []
    for start, end in zip(ordered, ordered[1:]):
        while start < end and context[start].isspace():
            start += 1
        while end > start and context[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def answer_sentences(context: str, answer_start: int, answer_text: str) -> tuple[str, int, int] | None:
    answer_end = answer_start + len(answer_text)
    if answer_start < 0 or context[answer_start:answer_end] != answer_text:
        return None
    spans = sentence_spans(context)
    covering = [i for i, (start, end) in enumerate(spans) if start < answer_end and end > answer_start]
    if not covering:
        return None
    first, last = covering[0], covering[-1]
    # A sentence beginning with a backward reference gets its preceding sentence.
    for _ in range(2):
        if first == 0 or not DEPENDENT_START.match(context[spans[first][0]:spans[first][1]]):
            break
        previous = context[spans[first - 1][0]:spans[first - 1][1]]
        if not END_RE.search(previous):
            return None
        first -= 1
    start, end = spans[first][0], spans[last][1]
    text = context[start:end]
    if not start <= answer_start < answer_end <= end or not END_RE.search(text):
        return None
    return re.sub(r"'{2,}", "", normalize(text)), start, end


def build_germanquad(path: Path) -> tuple[list[dict], dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("Install import-only dependency: python -m pip install pyarrow==25.0.1") from error
    rows = parquet.read_table(path).to_pylist()
    active = []
    rejected: Counter = Counter()
    seen = set()
    for row in sorted(rows, key=lambda item: int(item["id"])):
        question, context = normalize(row["question"]), row["context"]
        if not 3 <= len(WORD_RE.findall(question)) <= 80 or not plausible_german(question):
            rejected["question_length_or_language"] += 1
            continue
        answers = row["answers"]
        candidates = []
        for text, start in zip(answers["text"], answers["answer_start"]):
            if text.strip():
                candidate = answer_sentences(context, start, text)
                if candidate is not None:
                    candidates.append((candidate, text, start))
        if not candidates:
            rejected["no_valid_complete_answer_sentence"] += 1
            continue
        (answer, start, end), gold, gold_start = min(candidates, key=lambda item: (len(item[0][0]), item[2]))
        if not 6 <= len(WORD_RE.findall(answer)) <= 140 or len(answer) > 900:
            rejected["answer_length"] += 1
            continue
        if CODE_RE.search(question + "\n" + answer) or not plausible_german(answer) or re.search(r"={2,}|\[\[|\]\]|\{\{|\}\}", answer):
            rejected["answer_format_or_language"] += 1
            continue
        if not re.match(r"[A-ZÄÖÜ„\"(]", answer):
            rejected["fragment_or_list_item"] += 1
            continue
        key = (question.casefold(), answer.casefold())
        if key in seen:
            rejected["duplicate_pair"] += 1
            continue
        seen.add(key)
        title = context.splitlines()[0].strip()
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        article_url = "https://de.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="_()")
        active.append({
            "id": f"germanquad-train-{row['id']}", "prompt": question, "text": answer,
            "source": f"https://huggingface.co/datasets/deepset/germanquad | CC-BY-4.0 annotations; Wikipedia contributors CC-BY-SA | train={row['id']} | article_title={title} | {article_url}",
            "provenance": {
                "dataset": "deepset/germanquad", "revision": QUAD_REVISION,
                "dataset_card_revision": QUAD_CARD_REVISION, "split": "train",
                "dataset_license": "CC-BY-4.0", "context_origin": "German Wikipedia",
                "context_attribution": "Wikipedia editors and contributors; original article text under Wikipedia's share-alike terms",
                "article_title": title, "article_url": article_url,
                "question_id": row["id"], "context_sha256": context_hash,
                "context_sentence_start": start, "context_sentence_end": end,
                "gold_answer_text": gold, "gold_answer_start": gold_start,
                "normalization": "Original context sentence(s), Unicode NFC/whitespace normalization and removal of repeated Wiki emphasis apostrophes; no generated answer",
                "raw_source_sha256": DOWNLOADS["germanquad"]["sha256"],
                "facts_as_of": "original dataset release (2021), not independently verified as current",
            },
        })
    return active, {"raw_train_questions": len(rows), "active_pairs": len(active), "distinct_contexts": len({row["provenance"]["context_sha256"] for row in active}), "rejections": dict(sorted(rejected.items()))}


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "records": len(rows), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require cached SHA-verified source files")
    args = parser.parse_args()
    paths = {name: download_verified(spec, args.offline) for name, spec in DOWNLOADS.items()}
    dialogue, followups, oasst_stats = build_oasst(paths["oasst2"])
    knowledge, quad_stats = build_germanquad(paths["germanquad"])
    outputs = {
        "dialogue": write_jsonl(IMPORTS / "public_dialogue_oasst2_de.jsonl", dialogue),
        "knowledge": write_jsonl(IMPORTS / "public_knowledge_germanquad_de.jsonl", knowledge),
        "combined": write_jsonl(IMPORTS / "public_dialogue_expansion.jsonl", dialogue + knowledge),
        "inactive_followups": write_jsonl(SOURCES / "openassistant-oasst2/followup_candidates_de.jsonl", followups),
    }
    manifest = {
        "schema_version": 1, "script_sha256": sha256(Path(__file__)),
        "filters": FILTER_CONFIGURATION, "source_statistics": {"oasst2": oasst_stats, "germanquad": quad_stats},
        "sources": {name: {"url": spec["url"], "sha256": spec["sha256"], "bytes": spec["path"].stat().st_size} for name, spec in DOWNLOADS.items()},
        "outputs": outputs, "sqlite_import_performed": False,
        "quality_note": "Filtered historical source data, not independently fact-checked. More corpus entries do not prove better generated answers. OASST alternative replies share roots; GermanQuAD questions share source passages.",
        "evaluation_files_read": [],
    }
    (IMPORTS / "public_dialogue_expansion.provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": outputs, "statistics": manifest["source_statistics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
