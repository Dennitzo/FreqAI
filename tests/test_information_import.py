"""Information imports preserve source prose and never fabricate QA pairs."""
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments import import_information_corpus as importer


SPEC = {"name": "sample.parquet", "sha256": "0" * 64}
PROSE = (
    "Der Garten ist ein abgegrenztes Stück Land, in dem Pflanzen wachsen. "
    "Die Menschen nutzen ihn zur Erholung und bauen dort auch Obst und Gemüse an."
)


def article(number="12", title="Garten", prose=PROSE):
    return {"id": number, "title": title,
            "url": "https://de.wikipedia.org/wiki/" + title,
            "text": prose}


def parquet_file(path, articles):
    pq.write_table(pa.Table.from_pylist(articles), path)
    return path, {**SPEC, "name": path.name}


def test_information_record_has_no_question_and_keeps_original_title():
    record, reason = importer.article_record(article(), SPEC)
    assert reason is None
    assert record["prompt"] == ""
    assert record["text"] == "Garten\n\n" + PROSE
    assert record["title"] == record["provenance"]["article_title"] == "Garten"
    assert record["provenance"]["prompt_created"] is False
    assert record["provenance"]["article_revision_id"] is None


def test_original_offsets_and_hash_keep_leading_whitespace_and_original_unicode():
    source = "\n\tGarten\n\n  " + PROSE.replace("Gemüse", "Gemu\u0308se") + "  \n\nGeschichte\n\nWeiterer Text.\n"
    record, reason = importer.article_record(article(prose=source), SPEC)
    assert reason is None
    provenance = record["provenance"]
    assert provenance["article_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    reconstructed = "\n\n".join(importer.normalized(source[start:end])
                                for start, end in provenance["paragraph_spans"])
    assert reconstructed == PROSE
    assert record["text"] == "Garten\n\n" + reconstructed
    assert provenance["paragraph_sha256"] == hashlib.sha256(reconstructed.encode()).hexdigest()


def test_two_short_adjacent_complete_paragraphs_are_kept_whole():
    first = "Der Garten ist ein grüner Ort für die Menschen in der Stadt."
    second = "Viele Menschen bauen dort Obst und Gemüse an und ruhen sich nach der Arbeit aus."
    prose, offsets, reason = importer.opening_paragraphs("Garten", first + "\n\n" + second)
    assert reason is None
    assert prose == first + "\n\n" + second
    assert len(offsets) == 2


def test_section_heading_is_never_skipped_to_create_a_false_contiguous_lead():
    first = "Der Garten ist ein grüner Ort für die Menschen in der Stadt."
    prose, offsets, reason = importer.opening_paragraphs("Garten", first + "\n\nGeschichte\n\n" + PROSE)
    assert (prose, offsets) == ("", [])
    assert reason == "opening_not_complete_prose"


def test_heading_separated_by_single_newline_is_not_flattened_into_prose():
    first = "Der Garten ist ein grüner Ort für die Menschen in der Stadt."
    prose, offsets, reason = importer.opening_paragraphs("Garten", first + "\n\nGeschichte\n" + PROSE)
    assert (prose, offsets) == ("", [])
    assert reason == "opening_embedded_linebreak"


@pytest.mark.parametrize("gap", ["()", "(; )", "(Stand: )", "(Stand )", "(; anderer Name: Garten)"])
def test_empty_fields_from_publisher_template_removal_are_rejected(gap):
    record, reason = importer.article_record(article(prose=PROSE + " " + gap + "."), SPEC)
    assert record is None
    assert reason == "empty_exported_field"


def test_empty_etymology_prefix_is_deleted_with_exact_source_evidence():
    source = PROSE.replace("Der Garten", "Der Garten (von; auch Park genannt)", 1)
    record, reason = importer.article_record(article(prose=source), SPEC)
    assert reason is None
    assert record["text"] == "Garten\n\n" + source.replace("(von; auch", "(auch")
    provenance = record["provenance"]
    assert provenance["removed_empty_template_fragments"] == ["von; "]
    assert provenance["article_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    start, end = provenance["paragraph_spans"][0]
    assert source[start:end] == source


def test_real_parenthetical_etymology_is_preserved():
    source = "Der Begriff (von lateinisch hortus; auch Garten genannt) bleibt unverändert."
    assert importer.normalized(source) == source


def test_long_paragraph_is_rejected_instead_of_truncated():
    record, reason = importer.article_record(article(prose=PROSE * 10), SPEC)
    assert record is None
    assert reason == "opening_too_long"


@pytest.mark.parametrize("prose", [PROSE[:-1], "* " + PROSE, PROSE + " https://example.com."])
def test_unfinished_prose_lists_and_raw_links_are_rejected(prose):
    record, reason = importer.article_record(article(prose=prose), SPEC)
    assert record is None
    assert reason == "opening_not_complete_prose"


def test_dependent_opening_cannot_lose_its_antecedent():
    record, reason = importer.article_record(article(prose="Diese sind besonders wichtig für die Menschen. " + PROSE), SPEC)
    assert record is None
    assert reason == "dependent_opening"


def test_disambiguation_pages_are_not_treated_as_a_single_definition():
    record, reason = importer.article_record(article(title="Garten (Begriffsklärung)"), SPEC)
    assert record is None
    assert reason == "list_or_disambiguation_title"


def test_selection_is_identical_after_row_order_changes(tmp_path):
    articles = [article(str(number), "Garten " + str(number)) for number in range(1, 16)]
    first, _ = importer.build([parquet_file(tmp_path / "a.parquet", articles)], count=5)
    second, _ = importer.build([parquet_file(tmp_path / "b.parquet", list(reversed(articles)))], count=5)
    assert len(first) == len(second) == 5
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert all(row["prompt"] == "" for row in first + second)


def test_article_ids_and_identical_title_prose_are_deduplicated(tmp_path):
    articles = [article(), article(), article("13"), article("14", "Park")]
    selected, stats = importer.build([parquet_file(tmp_path / "a.parquet", articles)], count=10)
    assert [row["provenance"]["article_id"] for row in selected] == ["12", "14"]
    assert stats["rejections"]["duplicate_article_id"] == 1
    assert stats["rejections"]["duplicate_title_and_prose"] == 1


def test_unexpected_qa_source_schema_is_rejected(tmp_path):
    path = tmp_path / "qa.parquet"
    pq.write_table(pa.Table.from_pylist([{"question": "Frage?", "answer": "Antwort."}]), path)
    with pytest.raises(ValueError, match="Unexpected source schema"):
        importer.build([(path, SPEC)])


def test_wrong_local_source_checksum_is_rejected_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "SOURCE_DIR", tmp_path)
    (tmp_path / "source.parquet").write_bytes(b"changed archive")
    spec = {"name": "source.parquet", "bytes": 15, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        importer.download(spec, offline=True)


def test_count_is_validated_before_accessing_sources():
    for value in (0, -1, True, 100001):
        with pytest.raises(ValueError, match="count must"):
            importer.build([], count=value)
