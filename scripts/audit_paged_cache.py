"""Independently compare every indexed passage with the central full texts."""
import argparse
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', type=Path)
    parser.add_argument('--memory', type=Path, default=Path('memory/memory.sqlite3'))
    parser.add_argument('--output', type=Path, default=Path('runtime/paged-cache-audit.json'))
    args = parser.parse_args()
    started = time.monotonic()
    source = sqlite3.connect(args.memory.resolve().as_uri()+'?mode=ro', uri=True)
    cache = sqlite3.connect(args.cache.resolve().as_uri()+'?mode=ro', uri=True)
    cursor = cache.execute('SELECT sequence,document_id,text,source FROM passages ORDER BY rowid')
    grouped = iter(groupby(cursor, key=lambda row: row[0]))
    count = byte_count = 0
    for seq, identifier, text, provenance in source.execute('SELECT sequence,id,text,source FROM documents ORDER BY sequence'):
        group_seq, rows = next(grouped)
        assert seq == group_seq, (seq, group_seq)
        parts = []
        for row in rows:
            assert row[1] == identifier and row[3] == provenance, identifier
            parts.append(row[2])
        assert ''.join(parts) == text, identifier
        count += 1
        byte_count += len(text.encode('utf-8'))
        if count % 25000 == 0:
            print(f'{count:,} vollständige Artikel/Texte verglichen', flush=True)
    assert next(grouped, None) is None
    packet_count, stored_bytes = cache.execute('SELECT count(*),sum(text_bytes) FROM packets').fetchone()
    assert packet_count == count and stored_bytes == byte_count
    result = {'documents_verified': count, 'full_text_bytes': byte_count,
              'all_passages_exact': True, 'packet_count': packet_count,
              'elapsed_s': time.monotonic()-started}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    source.close()
    cache.close()


if __name__ == '__main__':
    main()
