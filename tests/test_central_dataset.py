"""Integration checks for the central corpus; no exported dataset copies needed."""
import hashlib
import json
from central_information import connect


def test_source_receipts_describe_complete_local_wikipedia():
    with connect() as connection:
        rows = connection.execute('SELECT sha256,byte_count,record_count,metadata_json FROM information_imports').fetchall()
    shards = [row for row in rows if json.loads(row[3]).get('kind') == 'complete_wikipedia']
    assert len(shards) >= 4
    assert sum(row[2] for row in shards) >= 569062
    assert all(len(row[0]) == 64 and row[1] > 0 for row in shards)


def test_all_imported_article_addresses_exist_without_duplicate_leads():
    with connect() as connection:
        expected = sum(count for count, metadata in connection.execute('SELECT record_count,metadata_json FROM information_imports')
                       if json.loads(metadata).get('kind') == 'complete_wikipedia')
        prefix = 'wikipedia-de-20231101-'
        total, leads = connection.execute("SELECT COUNT(*),SUM(id LIKE '%-lead') FROM documents WHERE id>=? AND id<?",
                                         (prefix, prefix + '\uffff')).fetchone()
    assert total == expected
    assert leads == 0


def test_active_schema_and_archival_cleanup():
    with connect() as connection:
        columns = {row[1] for row in connection.execute('PRAGMA table_info(documents)')}
        assert not {'prompt', 'question', 'answer'}.intersection(columns)
        assert connection.execute('SELECT COUNT(*) FROM document_archive').fetchone()[0] == 0


def test_full_text_hashes_and_original_boundaries_at_both_ends_of_corpus():
    with connect() as connection:
        for direction in ('ASC', 'DESC'):
            rows = connection.execute(f'''SELECT d.text,p.metadata_json FROM documents d
                JOIN information_provenance p ON p.document_id=d.id
                WHERE d.id>=? AND d.id<? ORDER BY d.id {direction} LIMIT 10''',
                ('wikipedia-de-20231101-', 'wikipedia-de-20231101-\uffff')).fetchall()
            assert len(rows) == 10
            for text, encoded in rows:
                metadata = json.loads(encoded)
                title, separator, original = text.partition('\n\n')
                assert separator and title == metadata['article_title']
                assert hashlib.sha256(original.encode()).hexdigest() == metadata['article_sha256']
                assert hashlib.sha256(text.encode()).hexdigest() == metadata['text_sha256']


def test_authored_information_survived_consolidation():
    with connect() as connection:
        count = connection.execute('SELECT COUNT(*) FROM documents WHERE id>=? AND id<?',
                                   ('conversation-', 'conversation-\uffff')).fetchone()[0]
    assert count == 93


def test_addressability_probes_only_use_declared_subject_and_copula_sentences():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from understanding_eval import addressability_probes
    records = [{'text': 'Die Brillinge sind eine Gruppe kleiner Seevögel mit schwarzem Gefieder.'},
               {'text': 'Der Velaskörper rotiert langsam um seine Achse.'},
               {'text': 'Sie waren früher häufig, heute fehlen sie ganz.'}]
    probes = addressability_probes(records)
    assert len(probes) == 1
    assert probes[0]['prompt'].startswith('Was ist ')
    assert probes[0]['reference'].startswith('Die Brillinge')
