"""Check every imported UTF-8 text/metadata wave, separately from generation."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freqai.cli import read_documents
from freqai.codec import decode_text, encode_text
from freqai.memory import WaveMemory


def main():
    path = ROOT/"memory/imports/public_dialogue_expansion.jsonl"
    documents = read_documents(path)
    checks = 0
    modes = 0
    started = time.perf_counter()
    for document in documents:
        text_packet = encode_text(document.text)
        metadata_packet = WaveMemory._metadata_packet(document)
        modes += len(text_packet.coefficients)+len(metadata_packet.coefficients)
        for time_s in (0.0, 86400.125):
            assert decode_text(text_packet, time_s) == document.text, document.id
            metadata = json.loads(decode_text(metadata_packet, time_s))
            assert metadata == {"id": document.id, "source": document.source, "prompt": document.prompt}
            checks += 2
    report = {"corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "records": len(documents),
              "wave_packet_roundtrips": checks, "checked_times_s": [0.0, 86400.125],
              "text_and_metadata_modes": modes, "all_exact": True,
              "seconds": time.perf_counter()-started,
              "scope": "Reversible storage of every imported byte, not generated-answer quality."}
    target = ROOT/"results/public_corpus/public_codec_audit.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
