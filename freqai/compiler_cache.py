"""One atomic, non-executable cache of a derived information compiler.

The container is a standard uncompressed NPZ (JSON metadata as UTF-8 bytes and
numeric arrays only), followed by its SHA-256 trailer. No pickle, dynamic class
names, session state, or user database writes are involved. Invalid or obsolete
cache files are misses; invalid source records are still rejected before lookup.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import scipy

from .record_contract import validate_information_record

SCHEMA_VERSION = 1
CACHE_FILENAME = "compiled.npz"
_TRAILER = b"\nFREQAI-COMPILER-SHA256:"
_SOURCE_FILES = ("compiler_cache.py", "information.py", "unpaired.py", "spectral_storage.py",
                 "features.py", "spectral_ops.py", "waves.py", "codec.py", "record_contract.py", "parallel.py")
_CORPUS_ORIGINS = {"corpus_subject", "corpus_predicate", "corpus_predicate_argument", "corpus_lexical_fact"}
_JSON_FIELDS = ("concepts", "display_tokens", "context_forms", "groups", "lexicon", "lexicon_words",
                "lexicon_phrases", "lexicon_spellings", "action_roles", "state_vocabulary",
                "assistant_subjects", "assistant_aliases")
_SCALAR_FIELDS = ("order", "corpus_digest", "input_record_count", "record_count", "statement_count",
                  "unaddressed_statement_count", "size", "carrier_size", "edge_count", "data_gain", "resonance_floor")
_ARRAY_DTYPES = {"field_groups": "int64", "prefix_lengths": "uint8", "prefix_ids": "int64",
                 "offsets": "int64", "token_indices": "int64", "coefficients": "complex128",
                 "frequencies": "float64", "symbol_energy": "float64", "role_offsets": "int64",
                 "role_indices": "int64", "role_spectra": "complex128", "group_ranges": "int64",
                 "group_fingerprints": "uint8", "compilation_energy": "float64"}


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _validate_records(records, order):
    if type(order) is not int or not 1 <= order <= 5:
        raise ValueError("order must be an integer from 1 to 5")
    from .information import _record_fields
    items = list(records)
    for record in items:
        if isinstance(record, dict):
            validate_information_record(record)
            prompt = record.get("prompt", "")
        else:
            prompt = getattr(record, "prompt", "") if not isinstance(record, str) else ""
        if prompt:
            raise ValueError("UnpairedWaveModel accepts information declarations, not prompt/answer pairs")
        _record_fields(record)
    return items


def code_fingerprint():
    """Invalidate on implementation, schema, interpreter or numerical-version changes."""
    digest = hashlib.sha256()
    digest.update(_json_bytes([SCHEMA_VERSION, list(sys.version_info[:2]), np.__version__, scipy.__version__]))
    for name in _SOURCE_FILES:
        payload = Path(__file__).with_name(name).read_bytes()
        digest.update(name.encode()+b"\0"+len(payload).to_bytes(8, "little")+payload)
    return digest.hexdigest()


def _input_digest(records):
    # Ordered records and duplicates matter to the declaration compiler even
    # when the normalized information corpus digest happens to be unchanged.
    raw = []
    for record in records:
        if isinstance(record, str):
            raw.append(["str", record])
        elif isinstance(record, dict):
            raw.append(["dict", record])
        elif is_dataclass(record):
            raw.append(["dataclass", type(record).__module__, type(record).__qualname__, asdict(record)])
        else:
            # Unknown Python record classes have no safe exact raw encoding.
            # Their normal compiler path remains available without this cache.
            raise TypeError("Record type has no exact JSON cache representation")
    return hashlib.sha256(_json_bytes(raw)).hexdigest()


def _key(input_digest, order, fingerprint):
    return hashlib.sha256(_json_bytes([SCHEMA_VERSION, input_digest, order, fingerprint])).hexdigest()


def _auxiliary_digest(model):
    """Capture small mutable numerical state outside the packed transition bank."""
    digest = hashlib.sha256()

    def add(value):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode()+b"\0"+array.nbytes.to_bytes(8, "little"))
        digest.update(memoryview(array).cast("B"))

    add(model.symbol_energy)
    add(model.frequencies)
    add(model.token_frequencies)
    for role in model.lexical_roles:
        if role.origin in _CORPUS_ORIGINS:
            digest.update(role.origin.encode()+b"\0")
            add(role.indices)
            add(role.spectrum)
    return digest.hexdigest()


def cache_key(records, order=2):
    items = _validate_records(records, order)
    return _key(_input_digest(items), order, code_fingerprint())


def _encode(value):
    """Encode only fixed primitive containers, never a Python class identifier."""
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if isinstance(value, dict):
        return {"t": "dict", "v": [[_encode(key), _encode(item)] for key, item in value.items()]}
    kinds = {tuple: "tuple", list: "list", set: "set", frozenset: "frozenset"}
    kind = kinds.get(type(value))
    if kind:
        items = sorted(value, key=repr) if kind in {"set", "frozenset"} else value
        return {"t": kind, "v": [_encode(item) for item in items]}
    raise TypeError(f"Unsupported cache metadata type: {type(value).__name__}")


def _decode(value, depth=0):
    if depth > 32:
        raise ValueError("Cache metadata is too deeply nested")
    if value is None or type(value) in {str, int, float, bool}:
        return value
    if not isinstance(value, dict) or set(value) != {"t", "v"} or not isinstance(value["v"], list):
        raise ValueError("Invalid cache metadata container")
    kind, items = value["t"], value["v"]
    if kind == "dict":
        if any(not isinstance(item, list) or len(item) != 2 for item in items):
            raise ValueError("Invalid cache mapping")
        result = {_decode(key, depth+1): _decode(item, depth+1) for key, item in items}
        if len(result) != len(items):
            raise ValueError("Duplicate cache mapping key")
        return result
    factory = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}.get(kind)
    if factory is None:
        raise ValueError("Unknown cache metadata type")
    return factory(_decode(item, depth+1) for item in items)


def _pack(model, items, order):
    from .spectral_storage import PackedTransitionSpectra
    from .spectral_ops import BOS
    if model.order != order or model.input_record_count != len(items):
        raise ValueError("Compiler does not match the input record count/order")
    input_digest = _input_digest(items)
    if getattr(model, "_compiler_input_digest", None) != input_digest:
        raise ValueError("Compiler was built from different raw source records")
    if getattr(model, "_compiler_auxiliary_digest", None) != _auxiliary_digest(model):
        raise ValueError("Modified auxiliary coefficients cannot become a dataset cache")
    table = model.transition_spectra
    keys = table.field_keys if isinstance(table, PackedTransitionSpectra) else list(table)
    if isinstance(table, PackedTransitionSpectra):
        if not table.is_pristine():
            raise ValueError("Modified compiler buffers cannot become a dataset cache")
        for field in table._cache.values():
            if field._dense_override is not None or not np.shares_memory(field.local_spectrum, table.coefficients):
                raise ValueError("Modified diagnostic fields are not a compiler cache")
        offsets, indices, coefficients = table.offsets, table.token_indices, table.coefficients
    else:
        fields = [table[key] for key in keys]
        if any(field._dense_override is not None for field in fields):
            raise ValueError("Dense diagnostic overrides cannot be cached")
        offsets = np.concatenate(([0], np.cumsum([len(field.token_indices) for field in fields])))
        indices = np.concatenate([field.token_indices for field in fields]) if fields else np.zeros(0, dtype=np.int64)
        coefficients = np.concatenate([field.local_spectrum for field in fields]) if fields else np.zeros(0, dtype=np.complex128)
    field_groups = np.fromiter((key[0] for key in keys), dtype=np.int64, count=len(keys))
    prefix_lengths = np.fromiter((len(key[1]) for key in keys), dtype=np.uint8, count=len(keys))
    prefix_ids = np.full((len(keys), order), -1, dtype=np.int64)
    token_ids = {token: index for index, token in enumerate(model.vocabulary)}
    token_ids[BOS] = model.size
    if np.any(prefix_lengths > order):
        raise ValueError("Invalid cached prefix order")
    for position in range(order):
        prefix_ids[:, position] = np.fromiter(
            (token_ids[prefix[position]] if len(prefix) > position else -1 for _, prefix in keys),
            dtype=np.int64, count=len(keys))
    roles = []
    for role in model.lexical_roles:
        if role.origin not in _CORPUS_ORIGINS:
            break
        if role.surfaces is not None:
            raise ValueError("Session surfaces cannot enter the compiler cache")
        roles.append(role)
    if any(role.origin in _CORPUS_ORIGINS for role in model.lexical_roles[len(roles):]):
        raise ValueError("Corpus roles must precede generated grammar roles")
    arrays = {"field_groups": field_groups, "prefix_lengths": prefix_lengths, "prefix_ids": prefix_ids,
              "offsets": offsets, "token_indices": indices, "coefficients": coefficients,
              "frequencies": model.frequencies, "symbol_energy": model.symbol_energy,
              "role_offsets": np.concatenate(([0], np.cumsum([len(role.indices) for role in roles]))),
              "role_indices": np.concatenate([role.indices for role in roles]) if roles else np.zeros(0),
              "role_spectra": np.concatenate([role.spectrum for role in roles]) if roles else np.zeros(0),
              "group_ranges": np.asarray(getattr(model, "group_field_ranges", ()), dtype=np.int64).reshape(-1, 2),
              "compilation_energy": getattr(table, "mode_energy", None) if getattr(table, "mode_energy", None) is not None else np.zeros(0),
              "group_fingerprints": np.asarray([list(bytes.fromhex(value)) for value in
                                                 getattr(model, "group_fingerprints", ())], dtype=np.uint8).reshape(-1, 32)}
    arrays = {name: np.ascontiguousarray(value, dtype=_ARRAY_DTYPES[name]) for name, value in arrays.items()}
    fingerprint = code_fingerprint()
    metadata = {"schema": SCHEMA_VERSION, "code_fingerprint": fingerprint, "input_digest": input_digest,
                "key": _key(input_digest, order, fingerprint),
                "scalars": {name: getattr(model, name) for name in _SCALAR_FIELDS},
                "vocabulary": list(model.vocabulary), "attributes": {name: _encode(getattr(model, name)) for name in _JSON_FIELDS},
                "role_origins": [role.origin for role in roles],
                "facts": [[fact.subject, fact.verb, fact.subject_role, fact.verb_role, fact.object_role,
                           sorted(fact.attributes), fact.owner] for fact in model.facts]}
    return metadata, arrays


def _write_container(path, metadata, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".compiler-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            np.savez(stream, __meta__=np.frombuffer(_json_bytes(metadata), dtype=np.uint8), **arrays)
            end = stream.tell()
            stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
            stream.seek(end)
            stream.write(_TRAILER+digest.hexdigest().encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_container(path):
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        trailer_size = len(_TRAILER)+64
        if size <= trailer_size:
            raise ValueError("Truncated compiler cache")
        stream.seek(size-trailer_size)
        trailer = stream.read()
        if not trailer.startswith(_TRAILER):
            raise ValueError("Compiler cache checksum is missing")
        stream.seek(0)
        remaining = size-trailer_size
        digest = hashlib.sha256()
        while remaining:
            block = stream.read(min(1024*1024, remaining))
            if not block:
                raise ValueError("Truncated compiler cache")
            digest.update(block)
            remaining -= len(block)
        if digest.hexdigest().encode("ascii") != trailer[len(_TRAILER):]:
            raise ValueError("Compiler cache checksum mismatch")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            expected = {name+".npy" for name in (*_ARRAY_DTYPES, "__meta__")}
            if set(archive.namelist()) != expected or len(archive.infolist()) != len(expected):
                raise ValueError("Unexpected compiler cache entries")
            if any(entry.compress_type != zipfile.ZIP_STORED or entry.file_size > size for entry in archive.infolist()):
                raise ValueError("Invalid compiler cache container")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            meta = archive["__meta__"]
            if meta.dtype != np.uint8 or meta.ndim != 1:
                raise ValueError("Invalid compiler cache metadata bytes")
            metadata = json.loads(meta.tobytes().decode("utf-8"))
            arrays = {name: archive[name] for name in _ARRAY_DTYPES}
    return metadata, arrays


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid compiler cache integer")
    return value


def _string_collection(value, kind):
    return type(value) is kind and all(type(word) is str and word for word in value)


def _validate_attributes(attributes, role_count):
    for name, kind in (("concepts", set), ("state_vocabulary", set),
                       ("assistant_subjects", frozenset), ("assistant_aliases", set)):
        if not _string_collection(attributes[name], kind):
            raise ValueError("Invalid cached symbol collection")
    display = attributes["display_tokens"]
    if type(display) is not dict or any(type(key) is not str or type(value) is not str for key, value in display.items()):
        raise ValueError("Invalid cached display tokens")
    forms = attributes["context_forms"]
    if type(forms) is not dict or any(type(key) is not tuple or len(key) != 2
            or not _string_collection(key[0], tuple) or type(key[1]) is not str or type(value) is not str
            for key, value in forms.items()):
        raise ValueError("Invalid cached contextual spellings")
    lexicon = attributes["lexicon"]
    if type(lexicon) is not dict or any(type(key) is not str or type(values) is not list
            or any(type(value) is not int or not 0 <= value < role_count for value in values)
            for key, values in lexicon.items()):
        raise ValueError("Invalid cached lexical role indices")
    words = attributes["lexicon_words"]
    if type(words) is not dict or any(type(key) is not str or not _string_collection(values, set) for key, values in words.items()):
        raise ValueError("Invalid cached lexical words")
    phrases = attributes["lexicon_phrases"]
    if type(phrases) is not dict or any(type(key) is not str or type(values) is not set
            or any(not _string_collection(value, tuple) for value in values) for key, values in phrases.items()):
        raise ValueError("Invalid cached lexical phrases")
    spellings = attributes["lexicon_spellings"]
    if type(spellings) is not dict or any(not _string_collection(key, tuple) or not _string_collection(value, tuple)
                                        for key, value in spellings.items()):
        raise ValueError("Invalid cached lexical spellings")
    actions = attributes["action_roles"]
    if type(actions) is not dict or any(type(key) is not str or type(value) is not int or not 0 <= value < role_count
                                      for key, value in actions.items()):
        raise ValueError("Invalid cached action roles")


def _validate(metadata, arrays, order, expected_key=None):
    if not isinstance(metadata, dict) or set(metadata) != {
            "schema", "code_fingerprint", "input_digest", "key", "scalars", "vocabulary", "attributes", "role_origins", "facts"}:
        raise ValueError("Invalid compiler cache metadata schema")
    if type(metadata["schema"]) is not int or metadata["schema"] != SCHEMA_VERSION or metadata["code_fingerprint"] != code_fingerprint():
        raise ValueError("Obsolete compiler cache")
    for name in ("input_digest", "key"):
        if not isinstance(metadata[name], str) or len(metadata[name]) != 64 or len(bytes.fromhex(metadata[name])) != 32:
            raise ValueError("Invalid compiler cache fingerprint")
    if metadata["key"] != _key(metadata["input_digest"], order, metadata["code_fingerprint"]):
        raise ValueError("Compiler cache key mismatch")
    if expected_key is not None and metadata["key"] != expected_key:
        raise ValueError("Compiler input changed")
    scalars = metadata["scalars"]
    if not isinstance(scalars, dict) or set(scalars) != set(_SCALAR_FIELDS) or scalars["order"] != order:
        raise ValueError("Invalid compiler cache scalar schema")
    _integer(scalars["order"], 1)
    for name in ("input_record_count", "record_count", "statement_count", "unaddressed_statement_count", "edge_count"):
        _integer(scalars[name])
    size = _integer(scalars["size"], 1)
    _integer(scalars["carrier_size"], size)
    if not isinstance(scalars["corpus_digest"], str) or len(bytes.fromhex(scalars["corpus_digest"])) != 32:
        raise ValueError("Invalid corpus digest")
    if any(type(scalars[name]) not in {int, float} or not np.isfinite(scalars[name]) or scalars[name] < 0
           for name in ("data_gain", "resonance_floor")):
        raise ValueError("Invalid compiler cache gain")
    vocabulary = metadata["vocabulary"]
    if not isinstance(vocabulary, list) or len(vocabulary) != size or any(type(token) is not str for token in vocabulary) or len(set(vocabulary)) != size:
        raise ValueError("Invalid cached vocabulary")
    if not isinstance(metadata["attributes"], dict) or set(metadata["attributes"]) != set(_JSON_FIELDS):
        raise ValueError("Unexpected compiler cache attributes")
    attributes = {name: _decode(value) for name, value in metadata["attributes"].items()}
    for name, dtype in _ARRAY_DTYPES.items():
        array = arrays[name]
        if array.dtype != np.dtype(dtype) or array.dtype.hasobject or (array.dtype.kind in "fc" and not np.isfinite(array).all()):
            raise ValueError("Invalid compiler cache array type or value")
    groups = attributes["groups"]
    if type(groups) is not tuple or any(type(group) is not tuple or len(group) != 3
            or not isinstance(group[0], str) or not isinstance(group[1], str)
            or type(group[2]) is not tuple or any(type(word) is not str for word in group[2]) for group in groups):
        raise ValueError("Invalid cached information groups")
    n = len(arrays["field_groups"])
    if arrays["field_groups"].shape != (n,) or arrays["prefix_lengths"].shape != (n,) or arrays["prefix_ids"].shape != (n, order):
        raise ValueError("Invalid cached prefix dimensions")
    if np.any(arrays["field_groups"] < 0) or np.any(arrays["field_groups"] >= len(groups)) or np.any(arrays["prefix_lengths"] > order):
        raise ValueError("Invalid cached group or prefix index")
    mask = np.arange(order)[None, :] < arrays["prefix_lengths"][:, None]
    if np.any(arrays["prefix_ids"][mask] < 0) or np.any(arrays["prefix_ids"][mask] > size) or np.any(arrays["prefix_ids"][~mask] != -1):
        raise ValueError("Invalid cached prefix symbols")
    for offset_name, index_name, value_name, count in (
            ("offsets", "token_indices", "coefficients", n),
            ("role_offsets", "role_indices", "role_spectra", len(metadata["role_origins"]))):
        offsets, indices, values = arrays[offset_name], arrays[index_name], arrays[value_name]
        if offsets.shape != (count+1,) or offsets[0] != 0 or offsets[-1] != len(indices) or indices.ndim != 1 or values.shape != indices.shape:
            raise ValueError("Invalid cached coefficient boundaries")
        if np.any(offsets < 0) or np.any(offsets > len(indices)) or np.any(np.diff(offsets) < (1 if offset_name == "offsets" else 0)) or np.any(indices < 0) or np.any(indices >= size):
            raise ValueError("Invalid cached coefficient indices")
    if len(arrays["token_indices"]) != scalars["edge_count"]:
        raise ValueError("Cached edge count mismatch")
    if arrays["compilation_energy"].shape not in {(0,), arrays["coefficients"].shape} or np.any(arrays["compilation_energy"] < 0):
        raise ValueError("Invalid cached compilation energy")
    if arrays["frequencies"].shape != (size,) or arrays["symbol_energy"].shape != (size,) or np.any(arrays["symbol_energy"] < 0):
        raise ValueError("Invalid cached mode dimensions")
    origins = metadata["role_origins"]
    if not isinstance(origins, list) or any(origin not in _CORPUS_ORIGINS for origin in origins):
        raise ValueError("Session or grammar roles cannot enter this cache")
    _validate_attributes(attributes, len(origins))
    facts = metadata["facts"]
    if not isinstance(facts, list):
        raise ValueError("Invalid cached facts")
    for fact in facts:
        if not isinstance(fact, list) or len(fact) != 7 or any(type(fact[index]) is not str for index in (0, 1, 6)) \
                or any(type(fact[index]) is not int or not 0 <= fact[index] < len(origins) for index in (2, 3, 4)) \
                or not isinstance(fact[5], list) or any(type(word) is not str for word in fact[5]):
            raise ValueError("Invalid cached factual role")
    ranges, fingerprints = arrays["group_ranges"], arrays["group_fingerprints"]
    if ranges.shape not in {(0, 2), (len(groups), 2)} or fingerprints.shape not in {(0, 32), (len(groups), 32)} or len(ranges) != len(fingerprints):
        raise ValueError("Invalid cached group reuse metadata")
    if len(ranges) and (ranges[0, 0] != 0 or ranges[-1, 1] != n or np.any(ranges[:, 1] < ranges[:, 0])
                        or np.any(ranges[:-1, 1] != ranges[1:, 0])):
        raise ValueError("Invalid cached group field ranges")
    if len(ranges):
        changes = np.concatenate(([0], np.flatnonzero(np.diff(arrays["field_groups"]))+1, [n]))
        actual_ranges = np.column_stack((changes[:-1], changes[1:]))
        if not np.array_equal(ranges, actual_ranges) or not np.array_equal(
                arrays["field_groups"][changes[:-1]], np.arange(len(groups))):
            raise ValueError("Cached group ranges do not match the addressed fields")
    return attributes


def _restore(metadata, arrays, attributes):
    from .spectral_storage import PackedTransitionSpectra, PackedSupportMapping
    from .spectral_ops import BOS
    from .unpaired import Fact, LexicalRole, UnpairedWaveModel
    vocabulary = metadata["vocabulary"]
    prefix_vocabulary = vocabulary+[BOS]
    group_objects = list(range(len(attributes["groups"])))
    keys = []
    # Materialize scalar IDs in C once. Iterating millions of ndarray rows and
    # casting each NumPy scalar in Python dominated the original cache restore.
    for group, length, ids in zip(arrays["field_groups"].tolist(), arrays["prefix_lengths"].tolist(), arrays["prefix_ids"].tolist()):
        if length == 0:
            prefix = ()
        elif length == 1:
            prefix = (prefix_vocabulary[ids[0]],)
        elif length == 2:
            prefix = (prefix_vocabulary[ids[0]], prefix_vocabulary[ids[1]])
        else:
            prefix = tuple(prefix_vocabulary[index] for index in ids[:length])
        keys.append((group_objects[group], prefix))
    table = PackedTransitionSpectra(len(vocabulary), keys, arrays["offsets"], arrays["token_indices"], arrays["coefficients"],
                                    mode_energy=arrays["compilation_energy"] if len(arrays["compilation_energy"]) else None)
    model = UnpairedWaveModel.__new__(UnpairedWaveModel)
    model._compiler_input_digest = metadata["input_digest"]
    for name in _SCALAR_FIELDS:
        setattr(model, name, metadata["scalars"][name])
    for name in _JSON_FIELDS:
        setattr(model, name, attributes[name])
    model.vocabulary = vocabulary
    model.index = {token: index for index, token in enumerate(vocabulary)}
    model.frequencies = arrays["frequencies"]
    model.token_frequencies = model.frequencies
    model.symbol_energy = arrays["symbol_energy"]
    model.transition_spectra = table
    model.supports = PackedSupportMapping(table)
    model.group_fingerprints = tuple(row.tobytes().hex() for row in arrays["group_fingerprints"])
    model.group_field_ranges = tuple((int(start), int(stop)) for start, stop in arrays["group_ranges"])
    model.compilation_stats = {"cache_loaded": True, "compiled_groups": 0, "reused_groups": len(model.groups), "reused_fields": len(table)}
    model.compile_reuse_stats = {"groups_compiled": 0, "groups_reused": len(model.groups), "groups_total": len(model.groups),
                                 "compilation_energy_bytes": int(arrays["compilation_energy"].nbytes)}
    model._concept_heads = defaultdict(list)
    for concept in sorted(model.concepts):
        words = tuple(concept.split())
        model._concept_heads[words[0]].append((words, concept))
    model.group_indices_by_concept = defaultdict(set)
    for index, (_, _, concepts) in enumerate(model.groups):
        for concept in concepts:
            model.group_indices_by_concept[concept].add(index)
    model.lexical_roles = []
    model._role_ids = {}
    for row, origin in enumerate(metadata["role_origins"]):
        start, stop = arrays["role_offsets"][row:row+2]
        role = LexicalRole(arrays["role_indices"][start:stop], arrays["role_spectra"][start:stop], origin)
        model._role_ids[(origin, tuple(int(index) for index in role.indices))] = row
        model.lexical_roles.append(role)
    model.facts = [Fact(subject, verb, s, v, o, frozenset(words), owner) for subject, verb, s, v, o, words, owner in metadata["facts"]]
    model.facts_by_subject = defaultdict(list)
    for index, fact in enumerate(model.facts):
        model.facts_by_subject[fact.subject].append(index)
        if fact.owner and fact.owner != fact.subject:
            model.facts_by_subject[fact.owner].append(index)
    model.lexicon = defaultdict(list, model.lexicon)
    model.lexicon_words = defaultdict(set, model.lexicon_words)
    model.lexicon_phrases = defaultdict(set, model.lexicon_phrases)
    model._data_wave_cache = None
    model._grammar_roles = {}
    model._compiler_auxiliary_digest = _auxiliary_digest(model)
    return model


def _load(order, cache_dir, expected_key=None):
    if cache_dir is None:
        return None
    try:
        metadata, arrays = _read_container(Path(cache_dir)/CACHE_FILENAME)
        attributes = _validate(metadata, arrays, order, expected_key)
        return _restore(metadata, arrays, attributes)
    except (Exception, MemoryError):
        return None


def load_compiled(records, order=2, cache_dir=None):
    """Load only an exact ordered-input/code match; invalid input still raises."""
    items = _validate_records(records, order)
    if cache_dir is None:
        return None
    try:
        expected = _key(_input_digest(items), order, code_fingerprint())
    except Exception:
        return None
    return _load(order, cache_dir, expected)


def load_previous_compiled(order=2, cache_dir=None):
    """Load validated old derived groups for a fresh compiler's explicit reuse.

    This result is never a current-input cache hit. Callers must pass it only as
    previous_model to a fresh constructor, whose group fingerprints gate reuse.
    """
    if type(order) is not int or not 1 <= order <= 5:
        raise ValueError("order must be an integer from 1 to 5")
    return _load(order, cache_dir)


def save_compiled(model, records, order=2, cache_dir=None):
    """Atomically replace one derived snapshot; persistence failures return False."""
    if cache_dir is None:
        return False
    try:
        items = _validate_records(records, order)
        metadata, arrays = _pack(model, items, order)
        _validate(metadata, arrays, order, metadata["key"])
        _write_container(Path(cache_dir)/CACHE_FILENAME, metadata, arrays)
        return True
    except (Exception, MemoryError):
        return False


__all__ = ["cache_key", "code_fingerprint", "load_compiled", "load_previous_compiled", "save_compiled"]
