"""Verify the new prose bytes and the continuous wave at the expanded size."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqai.cli import read_documents
from freqai.codec import decode_text, encode_text
from freqai.memory import WaveMemory


def main():
    corpus = ROOT / "memory/imports/information_wikipedia_20000_v3.jsonl"
    base = ROOT / "results/information_corpus/evaluation/baseline_documents.jsonl"
    documents = read_documents(corpus)
    assert all(not doc.prompt.strip() for doc in documents)
    started = time.perf_counter()
    modes = checks = 0
    for doc in documents:
        text_packet = encode_text(doc.text)
        metadata_packet = WaveMemory._metadata_packet(doc)
        modes += len(text_packet.coefficients)+len(metadata_packet.coefficients)
        for time_s in (0.0, 86400.125):
            assert decode_text(text_packet, time_s) == doc.text, doc.id
            assert json.loads(decode_text(metadata_packet, time_s)) == {
                "id": doc.id, "source": doc.source, "prompt": doc.prompt}
            checks += 2
    report = {"corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
              "unpaired_records": len(documents), "codec_checks": checks, "all_exact": True,
              "new_text_metadata_modes": modes, "codec_seconds": time.perf_counter()-started,
              "scope": "Reversible bytes and numerical wave throughput, not language quality."}
    print(json.dumps(report), flush=True)
    combined = read_documents(base) + documents
    started = time.perf_counter()
    memory = WaveMemory(combined, feature_mode="morphology", retrieval_policy="coverage")
    report["combined_records"] = len(combined)
    report["wave_build_seconds"] = time.perf_counter()-started
    frames = []
    for time_s in (0.0, 13.75, 86400.125):
        started = time.perf_counter()
        frame = memory.snapshot(time_s, points=256)
        frames.append({"time_s": time_s, "seconds": time.perf_counter()-started,
                       "energy": frame["energy"], "modal_count": frame["modal_count"],
                       "key_mode_count": frame["key_mode_count"]})
    assert max(frame["energy"] for frame in frames)-min(frame["energy"] for frame in frames) < frames[0]["energy"]*1e-12
    report["frames"] = frames
    report["median_snapshot_seconds"] = statistics.median(frame["seconds"] for frame in frames)
    target = ROOT / "results/information_corpus/storage_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
