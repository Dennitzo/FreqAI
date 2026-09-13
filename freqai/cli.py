"""Command line access to the central text store and its live wave view."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

from .codec import decode_text, encode_text
from .memory import Document, WaveMemory
from .record_contract import validate_information_record
from .store import DEFAULT_MEMORY_PATH, MemoryStore, SnapshotRequired


def _document_id(text: str, source: str, prompt: str = "") -> str:
    encoded = json.dumps([prompt, text, source], ensure_ascii=False).encode("utf-8")
    return "text-" + hashlib.sha256(encoded).hexdigest()[:24]


def read_documents(path: str | Path, *, allow_legacy_pairs: bool = False) -> list[Document]:
    """Read an explicit input; content IDs avoid collisions across unnamed files."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        documents = WaveMemory.load(path).documents
        for document in documents:
            validate_information_record({"text": document.text, "prompt": document.prompt},
                                        allow_legacy_pairs=allow_legacy_pairs)
        return documents
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()]
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            validate_information_record(data, allow_legacy_pairs=allow_legacy_pairs)
        items = data["documents"] if isinstance(data, dict) else data
    else:
        text = path.read_text(encoding="utf-8-sig")
        source = str(path.resolve())
        return [Document(_document_id(text, source), text, source)]
    result = []
    for item in items:
        validate_information_record(item, allow_legacy_pairs=allow_legacy_pairs)
        text, source = item["text"], str(item.get("source", path.name))
        prompt = item.get("prompt", "")
        document_id = str(item["id"]) if "id" in item else _document_id(text, source, prompt)
        result.append(Document(document_id, text, source, prompt))
    return result


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="FreqAI: Text-Wellenspeicher und spektraler Wortgenerator")
    sub = parser.add_subparsers(dest="command", required=True)

    def memory_argument(command):
        command.add_argument("--memory", default=str(DEFAULT_MEMORY_PATH),
                             help="Zentrale SQLite-Datei; Standard unabhängig vom Arbeitsverzeichnis")

    ingest = sub.add_parser("import", help="Texte atomar und ohne Überschreiben ergänzen (JSONL/JSON/TXT/NPZ)")
    ingest.add_argument("input")
    memory_argument(ingest)
    add = sub.add_parser("add", help="Einen deklarativen Informationstext direkt ergänzen")
    add.add_argument("--text", required=True, help="Informationstext mit vollständigen Aussagen")
    add.add_argument("--source", default="Eigene Texte", help="Herkunft des Informationstexts")
    add.add_argument("--id", dest="document_id")
    memory_argument(add)
    archive = sub.add_parser("archive", help="Alttexte zentral aufbewahren, ohne sie als Antworten zu aktivieren")
    archive.add_argument("input")
    memory_argument(archive)
    build = sub.add_parser("build", help="Veralteter Alias für import; ergänzt vorhandene Daten")
    build.add_argument("input")
    memory_argument(build)
    export = sub.add_parser("export", help="Explizite NPZ-Kopie der Wissensdokumente exportieren")
    export.add_argument("--output", required=True)
    memory_argument(export)
    relation_import = sub.add_parser("import-relations", help="Explizite Beziehungen in die zentrale Memory ergänzen")
    relation_import.add_argument("input")
    memory_argument(relation_import)
    stats = sub.add_parser("stats", help="Inhalt und Revision der zentralen Memory anzeigen")
    memory_argument(stats)
    migrate = sub.add_parser("migrate-information", help="Frühere Paare und Antwortprior archivieren; reines Informationsschema aktivieren")
    memory_argument(migrate)
    ask = sub.add_parser("ask", help="Eine Gesprächsantwort aus Prompt, Kontext und Wellenspeicher berechnen")
    ask.add_argument("prompt")
    memory_argument(ask)
    ask.add_argument("--top-k", type=int, default=1)
    ask.add_argument("--time", type=float, default=0.0)
    ask.add_argument("--session", help="Gesprächs-ID für dauerhaften Kontext; ohne ID ist jede Abfrage unabhängig")
    ask.add_argument("--mode", choices=("wave",), default="wave",
                     help="Einheitliche Wortausgabe aus Informationsfeldern")
    ask.add_argument("--max-tokens", type=int, default=40)
    ask.add_argument("--seed", type=int, default=17)
    ask.add_argument("--json", action="store_true")
    generate = sub.add_parser("generate", help="Eine neue Tokenfolge autoregressiv aus dem Wellenfeld berechnen")
    generate.add_argument("prompt")
    memory_argument(generate)
    generate.add_argument("--time", type=float, default=0.0)
    generate.add_argument("--session", help="Gesprächs-ID für einen dauerhaft gespeicherten Kontextzustand")
    generate.add_argument("--max-tokens", type=int, default=40)
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument("--json", action="store_true")
    serve = sub.add_parser("serve", help="Dauerhafte Schwingungsberechnung mit lokalem Dashboard")
    memory_argument(serve)
    serve.add_argument("--data", help="Optionaler zusätzlicher Import beim Start, ersetzt niemals vorhandene Texte")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true")
    oscillate = sub.add_parser("oscillate", help="Alle Moden kontinuierlich berechnen, Ende mit Strg+C")
    memory_argument(oscillate)
    oscillate.add_argument("--duration", type=float, default=0.0, help="0 = bis Strg+C")
    oscillate.add_argument("--hz", type=float, default=10.0)
    codec = sub.add_parser("codec", help="Text-Welle-Text-Rückrechnung prüfen")
    codec.add_argument("text")
    codec.add_argument("--time", type=float, default=123.456)
    infer = sub.add_parser("infer", help="Aussage aus gespeicherten is_a-Relationen im Frequenzraum ableiten")
    infer.add_argument("subject")
    infer.add_argument("target")
    memory_argument(infer)
    infer.add_argument("--relations", help="Optional zuvor Beziehungen aus dieser JSON-Datei ergänzen")
    infer.add_argument("--max-hops", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.command == "codec":
            packet = encode_text(args.text)
            print(json.dumps({"text": decode_text(packet, args.time), "bytes": packet.byte_length,
                              "modes": len(packet.coefficients), "time_s": args.time,
                              "sha256": packet.sha256}, ensure_ascii=False, indent=2))
            return
        store = MemoryStore(args.memory)
        if args.command in {"import", "build", "add"}:
            if args.command == "build":
                print("Hinweis: 'build' ist ein veralteter Alias für 'import'; bestehende Daten "
                      "werden ergänzt und niemals ersetzt.", file=sys.stderr)
            if args.command == "add":
                if not args.text.strip():
                    raise ValueError("Der gespeicherte Text darf nicht leer sein")
                source = args.source
                document_id = args.document_id or _document_id(args.text, source)
                documents = [Document(document_id, args.text, source)]
            else:
                documents = read_documents(args.input)
            result = store.append_documents(documents)
            print(json.dumps({"path": str(store.path), "revision": result["revision"],
                              "added": len(result["added"]), "existing": len(result["existing"])},
                             ensure_ascii=False, indent=2))
        elif args.command == "archive":
            print(json.dumps(store.archive_documents(read_documents(args.input, allow_legacy_pairs=True)), ensure_ascii=False, indent=2))
        elif args.command == "export":
            output = Path(args.output).expanduser().resolve()
            if output.suffix.lower() != ".npz":
                raise ValueError("Export benötigt einen expliziten Dateinamen mit Endung .npz")
            memory = store.load_memory()
            memory.save(output)
            print(f"{len(memory.documents)} Wissensdokumente als NPZ exportiert: {output}")
        elif args.command == "import-relations":
            triples = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
            print(json.dumps(store.add_relations(triples), ensure_ascii=False, indent=2))
        elif args.command == "stats":
            print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        elif args.command == "migrate-information":
            print(json.dumps(store.migrate_information_only(), ensure_ascii=False, indent=2))
        elif args.command in {"ask", "generate"}:
            from .server import WaveEngine
            store.migrate_information_only()
            engine = WaveEngine(store.load_memory(), central_store=store)
            try:
                result = engine.ask(args.prompt, top_k=getattr(args, "top_k", 1), time_s=args.time,
                                    session_id=args.session, mode="wave" if args.command == "generate" else args.mode,
                                    max_tokens=args.max_tokens, seed=args.seed)
            finally:
                engine.close()
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["answer"])
        elif args.command == "serve":
            from .server import serve as serve_dashboard
            from .progress import phase
            with phase('Start: zentrales Informationsschema prüfen'):
                migration = store.migrate_information_only()
            if migration["changed"]:
                print(f"Informationsschema aktiviert; {migration['archived']} Alttexte zentral archiviert.", flush=True)
            if args.data:
                store.append_documents(read_documents(args.data))
            memory = store.load_memory()
            print(f"FreqAI läuft unter http://127.0.0.1:{args.port}; Memory: {store.path}", flush=True)
            serve_dashboard(memory, port=args.port, central_store=store, open_browser=args.open)
        elif args.command == "infer":
            from .reasoning import WaveReasoner
            if args.relations:
                triples = json.loads(Path(args.relations).read_text(encoding="utf-8-sig"))
                store.add_relations(triples)
            result = WaveReasoner(store.relations()).infer(args.subject, args.target, args.max_hops)
            store.record_query(f"is_a({args.subject}, {args.target})", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "oscillate":
            if not 0 < args.hz <= 1000 or args.duration < 0:
                raise ValueError("hz must be in (0,1000] and duration nonnegative")
            revision = store.revision()
            memory = store.load_memory()
            started = time.perf_counter()
            tick = 0
            while True:
                elapsed = time.perf_counter() - started
                if args.duration and elapsed >= args.duration:
                    break
                try:
                    current = store.revision()
                    if current != revision:
                        memory = store.load_memory()
                except SnapshotRequired:
                    current = store.revision()
                    memory = store.load_memory()
                revision = current
                snapshot = memory.snapshot(elapsed)
                print(json.dumps({**{key: snapshot[key] for key in
                                    ("time_s", "energy", "modal_count", "document_count")},
                                  "memory_revision": revision}))
                tick += 1
                time.sleep(max(0.0, started + tick / args.hz - time.perf_counter()))
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        parser.exit(2, f"Fehler: {error}\n")
    except KeyboardInterrupt:
        print("\nBerechnung beendet.")
