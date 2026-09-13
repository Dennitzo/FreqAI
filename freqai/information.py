"""Information-only, count-compiled Fourier token generation.

No question/answer pairs, document similarity ranking or response templates.
Generic German subject/facet rules supply explicit linguistic assumptions.
Observed transitions determine complex coefficients; these are statistical
weights even though no gradient optimization or pretrained model is involved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
import unicodedata

import numpy as np
from scipy.fft import next_fast_len

from .features import STOPWORDS, canonical_term
from .parallel import chunk_ranges, map as parallel_map, worker_count
from .spectral_ops import BOS, EOS, spectral_convolution, stable_frequency
from .spectral_storage import CompactSpectrum, PackedSupportMapping, PackedTransitionSpectra
from .waves import (DEFAULT_DATA_GAIN, VERIFY_TOLERANCE, WaveField, build_field,
                    interference_pair, probabilities, resonance)


TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W_]+(?:['’-][^\W_]+)*|[^\w\s]", re.UNICODE)
COPULA = r"ist|sind|war|waren|bezeichnet|bedeutet|beschreibt|nennt|entspricht|misst|wird|werden|liegt|liegen|besteht|bestehen|gilt|gelten|hat|haben"
ARTICLES = {"der", "die", "das", "ein", "eine", "einer", "eines", "einem", "einen"}
FACET_PATTERNS = {
    "definition": r"\b(?:definition|bedeutung|bedeutet|versteht)\b|\bwas\s+(?:ist|sind)\b",
    "unit": r"\b(?:einheit\w*|gemessen|maßeinheit\w*|masseinheit\w*)\b",
    "formula": r"\b(?:formel\w*|berechnet|berechnen|berechnung|berechne|ausrechnen)\b",
    "relation": r"\b(?:zusammenhang|zusammenhängen|zusammenhaengen|verhältnis|verhaeltnis|beziehung|abhängig|abhaengig|unterschied)\b|\bzusammen\b",
}
REFERENCES = re.compile(r"\b(?:sie|ihre|ihrer|ihr|er|seine|seiner|es|dessen|diese|dieser|dazu|dabei)\b", re.I)
QUERY_FILLERS = frozenset(canonical_term(word) for word in """
bitte etwas genau einfach kurz allgemein ausführlich erkläre erklären erläutere erläutern beschreibe beschreiben
erzähle erzählen sage sagen nenne nennen bekannt wissen wissenswertes information informationen verstehe verstehen
versteht bedeutet bedeutung definition einheit einheiten maßeinheit masseinheit si si-einheit basiseinheit gemessen messen angegeben formel formeln berechnet berechnen
berechnung berechne ausrechnen zusammen zusammenhang zusammenhängen hängt hängen verhältnis beziehung unterschied
groß gross klein länge lang lange viele viel aufgabe aufgaben zweck dies diese dieser dieses diese lösen lösung
welchem welchen welchem welcher welche welches hat haben sein und unter versteht funktioniert funktionieren
""".split())
# Grammatical predicates identify a request, not an unknown domain argument.
# This list contains verbs only, independently of corpus-specific subject names.
QUESTION_PREDICATES = frozenset(canonical_term(word) for word in (COPULA.split("|")+"""
gibt gebe geben misst messen steht stehen steckt stecken dient dienen nutzt nutzen benutzt benutzen verwendet verwenden
erfolgt erfolgen erfolgt weist weisen zeigt zeigen entsteht entstehen besteht bestehen bezeichnet bezeichnen
nennt nennen definiert definieren lautet lauten gilt gelten benötigt benötigen braucht brauchen bedeutet bedeuten
beschreibt beschreiben entspricht entsprechen hängt hängen unterscheidet unterscheiden versteht verstehen
""".split()))
ABBREVIATIONS = frozenset({"z","b","bzw","ca","dr","prof","u","a"})


def _relation_arguments(sentence):
    concepts = set()
    for match in re.finditer(r"\b(?:Kehrwert|Produkt|Quotient|Summe|Differenz)\s+(?:der|des|von|aus)\s+([^.;!?]+)", sentence, re.I):
        for phrase in re.split(r"\bund\b|,", match.group(1)):
            if concept := _concept(phrase):
                concepts.add(concept)
    return concepts


def _unit_owner(sentence):
    match = re.search(r"\b(?:SI[- ]?)?(?:Maß)?[Ee]inheit\s+(?:der|des|von)\s+(.+?)(?=\s+(?:ist|sind|wird|werden)\b|[.;!?]|$)", sentence)
    return _concept(match.group(1)) if match else ""


def tokenize(text):
    return TOKEN_RE.findall(unicodedata.normalize("NFC", str(text)).casefold())


def _terms(text):
    return tuple(canonical_term(word) for word in tokenize(text)
                 if (word.isalpha() or any(char.isdigit() for char in word)) and word not in STOPWORDS)


def _concept(text):
    words = list(_terms(re.sub(r"\([^)]*\)", "", str(text))))
    while words and len(words[-1]) == 1:
        words.pop()  # A trailing variable symbol is a notation, not an entity.
    return " ".join(words)


def _sentences(text):
    text = unicodedata.normalize("NFC", str(text)).replace("\r", "")
    protected = text
    protected = re.sub(r"(?<=\d)\.(?=\d|\s+[A-Za-zÄÖÜäöü])", "\uE000", protected)
    protected = re.sub(r"\b(?:z|b|bzw|ca|dr|prof|u|a)\.", lambda match: match.group().replace(".", "\uE000"), protected, flags=re.I)
    return [re.sub(r"\s+", " ", part.replace("\uE000", ".")).strip()
            for part in re.split(r"(?<=[.!?])\s+|\n\s*\n", protected) if part.strip()]


def _subject(sentence):
    owner = re.search(r"\b(?:Einheit|Formel|Definition|Berechnung|Bedeutung)\s+(?:der|des|von)\s+(.+?)\s+(?:"+COPULA+r")\b", sentence, re.I)
    if owner:
        return _concept(owner.group(1))
    formula_owner = re.match(r"\s*Für\s+(?:die|den|das)?\s*(.+?)\s+(?:gilt|gelten)\b", sentence, re.I)
    if formula_owner:
        return _concept(formula_owner.group(1))
    match = re.match(r"\s*(.+?)\s+(?:"+COPULA+r")\b", sentence, re.I)
    if not match or len(match.group(1).split()) > 8:
        return ""
    phrase = match.group(1).strip()
    if REFERENCES.match(phrase) or phrase.casefold().startswith(("für ", "bei ", "mit ", "wenn ", "dann ")):
        return ""
    words = [word for word in TOKEN_RE.findall(phrase) if word.casefold() not in ARTICLES]
    # German noun phrases normally contain a capitalized noun. Avoid registering
    # arbitrary verbal/adverbial prefixes as new concepts.
    if not any(word[:1].isupper() for word in words):
        return ""
    return _concept(" ".join(words))


def _statement_facets(sentence):
    value = sentence.casefold()
    result = {"about"}
    unit = bool(re.search(FACET_PATTERNS["unit"], value))
    formula = bool(re.search(FACET_PATTERNS["formula"], value) or "=" in value)
    relation = bool(re.search(r"\b(?:kehrwert|zusammenhang|proportional|abhängig|produkt|quotient|verhältnis|zusammenhängen)\b", value))
    if unit:
        result.add("unit")
    if formula or relation:
        result.add("formula")
    if relation:
        result.add("relation")
    if not unit and re.search(r"\b(?:ist|sind|bezeichnet|bedeutet|beschreibt|definiert|entspricht)\b", value):
        result.add("definition")
    return result


def _unit(values):
    values = np.asarray(values)
    norm = np.linalg.norm(values)
    return values/norm if norm > 0 else values.copy()


@dataclass
class InformationPrompt:
    token_amplitudes: np.ndarray
    feature_indices: np.ndarray
    feature_spectrum: np.ndarray

    def __mul__(self, scalar):
        return InformationPrompt(self.token_amplitudes*scalar, self.feature_indices.copy(), self.feature_spectrum*scalar)

    __rmul__ = __mul__


def _record_fields(item):
    """Text und Titel eines Eingabedatensatzes, inklusive der Feldprüfung.

    Die Prüfung läuft im Aufrufer und nicht im Worker, damit eine ungueltige
    Eingabe weiterhin denselben Fehler an derselben Stelle wirft wie zuvor.
    """
    if isinstance(item, str):
        text, title = item, ""
    elif isinstance(item, dict):
        text = item.get("text", "")
        title = item.get("title", item.get("provenance", {}).get("article_title", ""))
    else:
        text, title = getattr(item, "text", ""), getattr(item, "title", "")
    if not isinstance(text, str) or not isinstance(title, str):
        raise ValueError("Information records need string text and optional string title")
    return text, title


def _normalize_record(item):
    """Titelabspaltung, Identität und Satztrennung eines einzelnen Datensatzes.

    Leerer Text liefert ``None``; ein Datensatz ohne Satze behält seine Identität,
    genau wie im seriellen Pfad, damit Duplikate dieselben Verlierer bleiben.
    """
    text, title = _record_fields(item)
    if not text.strip():
        return None
    if "\n" in text:
        first, remainder = text.split("\n", 1)
        if first.strip() and len(first) <= 100 and not re.search(r"[.!?]", first) \
                and (not title or first.strip() == title.strip()):
            title, text = title or first.strip(), remainder
    text = text.strip()
    identity = (title.strip(), re.sub(r"\s+", " ", text))
    return identity, title, text, _sentences(text)


def normalize_records(records):
    """Alle Eingabedatensätze auf allen Kernen normalisieren, Reihenfolge getreu.

    Deduplizierung, Anker und Korpus-Hash bleiben im Aufrufer, weil sie von der
    globalen Reihenfolge abhängen. Gehaltvoll ist die Satztrennung, und die ist
    pro Datensatz unabhängig.
    """
    items = list(records)
    for item in items:
        _record_fields(item)
    results = parallel_map(_normalize_record, items, min_items=32)
    normalized, seen, digest = [], set(), hashlib.sha256()
    for result in results:
        if result is None:
            continue
        identity, title, text, sentences = result
        if identity in seen:
            continue
        seen.add(identity)
        if not sentences:
            continue
        anchor = _concept(title) or next((_subject(sentence) for sentence in sentences if _subject(sentence)), "")
        normalized.append((anchor, sentences))
        digest.update(json.dumps((title, text), ensure_ascii=False, separators=(",", ":")).encode())
    return normalized, digest.hexdigest(), len(items)


def _concept_set_of(record):
    """Konzepte eines Datensatzes plus die geordneten Begriffe jeder seiner Sätze.

    Die Begrifflisten brauchen die spätere Statements-Phase: dort wird pro Paket
    nur der Kopf genau jener Konzepte übergeben, deren Wörter in diesem Paket
    vorkommen. Sonst würde bei jedem Paket der komplette Konzeptkopf über die
    Prozessgrenze wandern.
    """
    anchor, sentences = record
    found = set()
    if anchor:
        found.add(anchor)
    terms_by_sentence = []
    for sentence in sentences:
        subject = _subject(sentence)
        if subject:
            found.add(subject)
        found.update(_relation_arguments(sentence))
        owner = _unit_owner(sentence)
        if owner:
            found.add(owner)
        terms_by_sentence.append(tuple(_terms(sentence)))
    return found, terms_by_sentence


def collect_concepts(normalized):
    """Konzeptmenge des Korpus sammeln, parallel, und Begriffe je Satz mitbringen."""
    concepts = set()
    terms_by_record = []
    for part, terms in parallel_map(_concept_set_of, normalized, min_items=32):
        concepts.update(part)
        terms_by_record.append(terms)
    return concepts, terms_by_record


class _StatementPass:
    """Statements eines Pakets; trägt die Konzeptköpfe genau dieses Pakets.

    Dieselbe Regel wie im seriellen ``_mentions``: vollständigeregistered
    Phrasen schlagen ihre zufälligen Teilphrasen. Der Kopf ist auf die Wörter
    beschränkt, die in diesem Paket überhaupt auftreten - gleiches Ergebnis,
    Bruchteil der Daten über die Prozessgrenze.
    """

    __slots__ = ("heads",)

    def __init__(self, heads):
        self.heads = dict(heads)

    def mentions(self, words):
        found = set()
        for position, word in enumerate(words):
            for phrase, concept in self.heads.get(word, ()):
                if words[position:position+len(phrase)] == phrase:
                    found.add(concept)
        return {concept for concept in found if not any(concept != other and
                (" "+concept+" ") in (" "+other+" ") for other in found)}

    def __call__(self, items):
        prepared = []
        surface_forms = defaultdict(Counter)
        statements = unaddressed = 0
        for anchor, sentences, terms_by_sentence in items:
            previous = anchor
            for sentence, words in zip(sentences, terms_by_sentence):
                subject = _subject(sentence) or previous or anchor
                if not subject:
                    unaddressed += 1
                previous = subject
                concepts = self.mentions(words) | ({subject} if subject else set())
                if anchor:
                    concepts.add(anchor)
                facets = _statement_facets(sentence)
                owner = _unit_owner(sentence)
                if owner:
                    concepts.add(owner)
                if owner and owner != subject and "unit" in facets:
                    prepared.append((owner, tuple(sorted(concepts)), {"unit"}, tokenize(sentence)))
                    facets = facets-{"unit"}
                    if re.search(r"\b(?:ist|sind|bezeichnet)\b", sentence, re.I):
                        facets.add("definition")
                prepared.append((subject, tuple(sorted(concepts)), facets, tokenize(sentence)))
                statements += 1
                for surface in TOKEN_RE.findall(sentence):
                    surface_forms[surface.casefold()][surface] += 1
        return (prepared, {token: dict(counts) for token, counts in surface_forms.items()},
                statements, unaddressed)


def _statement_bundle(bundle):
    records, heads = bundle
    return _StatementPass(heads)(records)


def compile_statements(normalized, terms_by_record, concept_heads, *, packets=None):
    """Aussagen, Facetten und Schreibweisen des Korpus auf allen Kernen kompilieren.

    Die Reihenfolge der Pakete und innerhalb der Pakete bleibt erhalten; die
    Elternseite fügt Zähler und Liste in genau dieser Reihenfolge zusammen. Damit
    sind Zählerstände, Tie-Breaks von ``most_common`` und die Statement-Zahlen
    bitgenau die des seriellen Pfads.
    """
    count = len(normalized)
    if not count:
        return [], defaultdict(Counter), 0, 0
    packets = int(packets) if packets else max(1, worker_count() * 4)
    ranges = chunk_ranges(count, packets)
    bundles = []
    for start, stop in ranges:
        heads = {}
        for terms in terms_by_record[start:stop]:
            for sentence_words in terms:
                for word in sentence_words:
                    if word in concept_heads and word not in heads:
                        heads[word] = list(concept_heads[word])
        records = [(*normalized[index], terms_by_record[index]) for index in range(start, stop)]
        bundles.append((records, heads))
    prepared, surface_forms = [], defaultdict(Counter)
    statements = unaddressed = 0
    for rows, forms, row_count, missing_count in parallel_map(
            _statement_bundle, bundles, min_items=2 if count >= 32 else count + 1):
        prepared.extend(rows)
        for token, counts in forms.items():
            surface_forms[token].update(counts)
        statements += row_count
        unaddressed += missing_count
    return prepared, surface_forms, statements, unaddressed


class ContextFormsPass:
    """Kontextabhängige Großschreibung für mehrdeutige Symbole eines Pakets."""

    __slots__ = ("ambiguous",)

    def __init__(self, ambiguous):
        self.ambiguous = set(ambiguous)

    def __call__(self, items):
        local_forms = defaultdict(Counter)
        for _anchor, sentences in items:
            for sentence in sentences:
                previous = [BOS, BOS]
                for surface in TOKEN_RE.findall(sentence):
                    token = surface.casefold()
                    if token in self.ambiguous:
                        local_forms[(tuple(previous[-2:]), token)][surface] += 1
                    previous.append(token)
        return [(key, dict(counts)) for key, counts in local_forms.items()]


class _TransitionPass:
    """Compile independent descriptor groups without merging partial counts.

    A group owns every statement contributing to its prefixes. Workers receive
    integer token sequences plus vocabulary strings only for those sequences;
    neither a model nor a complete corpus is copied into every work packet.
    """

    def __init__(self, order):
        self.order = order

    def __call__(self, item):
        keys, offsets, all_indices, all_coefficients, all_energy = [], [0], [], [], []
        group_ranges = []
        for group, sequences in item:
            group_start = len(keys)
            counts = {}
            for tokens, targets in sequences:
                extended = [BOS] * self.order + tokens + [EOS]
                for position, target in enumerate(targets, self.order):
                    for depth in range(self.order + 1):
                        prefix = tuple(extended[position-depth:position]) if depth else ()
                        counter = counts.get(prefix)
                        if counter is None:
                            counts[prefix] = {target: 1}
                        else:
                            counter[target] = counter.get(target, 0) + 1
            for prefix, counter in counts.items():
                indices = sorted(counter)
                if len(indices) == 1:
                    all_coefficients.append(1.0 + 0.0j)
                    all_energy.append(1.0)
                else:
                    values = np.array([counter[index] for index in indices], dtype=float)
                    amplitudes = np.sqrt(values / values.sum())
                    spectrum = np.fft.fft(amplitudes, norm="ortho")
                    all_coefficients.extend(spectrum)
                    all_energy.extend(np.abs(np.fft.ifft(spectrum, norm="ortho")) ** 2)
                keys.append((group, prefix))
                all_indices.extend(indices)
                offsets.append(len(all_indices))
            group_ranges.append((group, group_start, len(keys)))
        return (keys, np.asarray(offsets, dtype=np.intp), np.asarray(all_indices, dtype=np.intp),
                np.asarray(all_coefficients, dtype=np.complex128), np.asarray(all_energy, dtype=float), group_ranges)


def _group_fingerprint(order, sequences):
    digest = hashlib.sha256(bytes([order]))
    for tokens, _ in sequences:
        encoded = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _reused_group(group, previous_model, previous_group, index_remap):
    table = previous_model.transition_spectra
    start, stop = previous_model.group_field_ranges[previous_group]
    edge_start, edge_stop = table.offsets[start], table.offsets[stop]
    offsets = table.offsets[start:stop + 1] - edge_start
    indices = index_remap[table.token_indices[edge_start:edge_stop]]
    coefficients = table.coefficients[edge_start:edge_stop]
    if table.mode_energy is not None:
        energy = table.mode_energy[edge_start:edge_stop]
    else:
        energy = np.ones(len(indices))
        for position in np.flatnonzero(np.diff(offsets) > 1):
            left, right = offsets[position:position + 2]
            energy[left:right] = np.abs(np.fft.ifft(coefficients[left:right], norm="ortho")) ** 2
    keys = (table.field_keys[start:stop] if group == previous_group else
            [(group, prefix) for _, prefix in table.field_keys[start:stop]])
    return keys, offsets, indices, coefficients, energy, [(group, 0, len(keys))]


class InformationWaveModel:
    """Compile information declarations into sparse complex transition fields."""

    def __init__(self, records, order=2, *, previous_model=None):
        if type(order) is not int or not 1 <= order <= 5:
            raise ValueError("order must be an integer from 1 to 5")
        records = list(records)
        total_chars = sum(len(record if isinstance(record, str) else record.get('text', '')
                              if isinstance(record, dict) else getattr(record, 'text', '')) for record in records)
        if total_chars > 4*1024**2:
            raise ValueError('Informationscompiler benötigt begrenzte Textabschnitte; '
                             'große Bestände über MemoryStore.load_memory() verwenden.')
        self.order = order
        from .progress import report
        report('Compiler 1/5: Texte normalisieren')
        normalized, self.corpus_digest, self.input_record_count = normalize_records(records)
        self.record_count = len(normalized)
        report('Compiler 2/5: Begriffe und Themen erfassen')
        self.concepts, terms_by_record = collect_concepts(normalized)
        self._concept_heads = defaultdict(list)
        for concept in sorted(self.concepts):
            words = tuple(concept.split())
            self._concept_heads[words[0]].append((words, concept))
        report('Compiler 3/5: Informationsaussagen aufbauen')
        prepared, surface_forms, self.statement_count, self.unaddressed_statement_count = \
            compile_statements(normalized, terms_by_record, self._concept_heads)
        self.vocabulary = sorted({token for _, _, _, tokens in prepared for token in tokens} | {EOS})
        self.index = {token: index for index, token in enumerate(self.vocabulary)}
        self.size = len(self.vocabulary)
        self.frequencies = np.array([stable_frequency(token) for token in self.vocabulary])
        self.token_frequencies = self.frequencies
        self.carrier_size = next_fast_len(self.size)
        self.display_tokens = {token: counts.most_common(1)[0][0]
                               for token, counts in surface_forms.items()}
        ambiguous = {token for token, counts in surface_forms.items()
                     if len(counts) > 1 and token not in STOPWORDS}
        local_forms = defaultdict(Counter)
        if ambiguous:
            packets = [normalized[start:stop] for start, stop in
                       chunk_ranges(len(normalized), max(1, worker_count() * 4))]
            for forms in parallel_map(ContextFormsPass(ambiguous), packets,
                                      min_items=2 if len(normalized) >= 32 else len(normalized) + 1):
                for key, counts in forms:
                    local_forms[key].update(counts)
        self.context_forms = {key: sorted(counts, key=lambda form: (-counts[form], form))[0]
                              for key, counts in local_forms.items()}
        self.groups = tuple(sorted({(subject, facet, concepts)
                                    for subject, concepts, facets, _ in prepared for facet in facets}))
        group_index = {descriptor: index for index, descriptor in enumerate(self.groups)}
        self.group_indices_by_concept = defaultdict(set)
        for index, (_, _, concepts) in enumerate(self.groups):
            for concept in concepts:
                self.group_indices_by_concept[concept].add(index)
        sequences_by_group = defaultdict(list)
        for subject, concepts, facets, tokens in prepared:
            targets = [self.index[token] for token in tokens] + [self.index[EOS]]
            for facet in sorted(facets):
                sequences_by_group[group_index[(subject, facet, concepts)]].append((tokens, targets))
        self.group_fingerprints = tuple(_group_fingerprint(order, sequences_by_group[group])
                                        for group in range(len(self.groups)))
        previous_groups = {}
        if previous_model is not None and previous_model.order == order \
                and isinstance(previous_model.transition_spectra, PackedTransitionSpectra) \
                and len(getattr(previous_model, "group_fingerprints", ())) == len(previous_model.groups) \
                and len(getattr(previous_model, "group_field_ranges", ())) == len(previous_model.groups) \
                and previous_model.transition_spectra.is_pristine():
            previous_groups = {descriptor: group for group, descriptor in enumerate(previous_model.groups)}
        index_remap = (np.array([self.index.get(token, -1) for token in previous_model.vocabulary], dtype=np.intp)
                       if previous_groups else None)
        packets, fresh = [], []
        reused_groups = reused_fields = 0
        for group, sequences in sorted(sequences_by_group.items()):
            old_group = previous_groups.get(self.groups[group])
            if old_group is not None and previous_model.group_fingerprints[old_group] == self.group_fingerprints[group]:
                packet = _reused_group(group, previous_model, old_group, index_remap)
                packets.append(packet)
                reused_groups += 1
                reused_fields += len(packet[0])
            else:
                fresh.append((group, sequences))
        work = [fresh[start:stop] for start, stop in chunk_ranges(len(fresh), max(1, worker_count() * 4))]
        report('Compiler 4/5: Spektrale Übergänge berechnen')
        packets.extend(parallel_map(_TransitionPass(order), work, min_items=2 if len(fresh) >= 32 else len(work) + 1))
        # Mixed reuse/rebuild packets can span nonconsecutive group IDs. Flatten
        # group slices before assembling in the original deterministic order.
        ordered = []
        for packet in packets:
            keys, offsets, indices, coefficients, energy, group_ranges = packet
            for group, left, right in group_ranges:
                edge_left, edge_right = offsets[left], offsets[right]
                ordered.append((group, keys[left:right], offsets[left:right + 1] - edge_left,
                                indices[edge_left:edge_right], coefficients[edge_left:edge_right], energy[edge_left:edge_right]))
        ordered.sort(key=lambda item: item[0])
        field_count = sum(len(item[1]) for item in ordered)
        edge_count = sum(len(item[3]) for item in ordered)
        field_keys = []
        offsets = np.empty(field_count + 1, dtype=np.intp)
        offsets[0] = 0
        packed_indices = np.empty(edge_count, dtype=np.intp)
        packed_coefficients = np.empty(edge_count, dtype=np.complex128)
        packed_energy = np.empty(edge_count, dtype=float)
        ranges = []
        field_position = edge_position = 0
        for group, keys, local_offsets, indices, coefficients, energy in ordered:
            field_stop, edge_stop = field_position + len(keys), edge_position + len(indices)
            ranges.append((field_position, field_stop))
            field_keys.extend(keys)
            offsets[field_position + 1:field_stop + 1] = local_offsets[1:] + edge_position
            packed_indices[edge_position:edge_stop] = indices
            packed_coefficients[edge_position:edge_stop] = coefficients
            packed_energy[edge_position:edge_stop] = energy
            field_position, edge_position = field_stop, edge_stop
        self.transition_spectra = PackedTransitionSpectra(self.size, field_keys, offsets, packed_indices,
                                                         packed_coefficients, mode_energy=packed_energy)
        self.supports = PackedSupportMapping(self.transition_spectra)
        self.group_field_ranges = tuple(ranges)
        self.symbol_energy = np.zeros(self.size)
        np.add.at(self.symbol_energy, packed_indices, packed_energy)
        self.edge_count = len(packed_indices)
        self.compilation_stats = {"reused_groups": reused_groups, "compiled_groups": len(fresh),
                                  "reused_fields": reused_fields}
        self.compile_reuse_stats = {"groups_reused": reused_groups, "groups_compiled": len(fresh),
                                    "groups_total": len(self.groups),
                                    "compilation_energy_bytes": int(packed_energy.nbytes)}
        self.data_gain = DEFAULT_DATA_GAIN
        self.resonance_floor = 0.25
        self._data_wave_cache = None
        # Full statements and temporary real counts are discarded after compilation.

    def _extra_data_energy(self) -> np.ndarray:
        """Additional data energy per symbol mode; declarative roles override this."""
        return np.zeros(self.size)

    def data_wave(self) -> WaveField:
        """The oscillation of all compiled data: every excited symbol mode."""
        cached = self._data_wave_cache
        key = (self.corpus_digest, self.size, len(self.transition_spectra))
        if cached is None or cached[0] != key:
            energy = self.symbol_energy + self._extra_data_energy()
            cached = (key, build_field("data", energy, self.frequencies).normalized())
            self._data_wave_cache = cached
        return cached[1]

    def prompt_wave(self, prompt_field) -> WaveField:
        """The oscillation of the current prompt plus its persisted session context."""
        amplitudes = np.asarray(prompt_field.token_amplitudes, dtype=float)
        if amplitudes.shape != (self.size,):
            raise ValueError("Prompt field does not match the model")
        return build_field("prompt", amplitudes, self.frequencies).normalized()

    def _mode_prior(self, wave: WaveField, active: np.ndarray, gain: float):
        """Unit-normalized all-data amplitudes on the candidate modes, scaled by ``gain``."""
        if gain <= 0:
            return None
        values = wave.restrict(active)
        norm = float(np.linalg.norm(values))
        if norm <= 0.0:
            return None
        return (float(gain)/norm)*values

    def _field_modes(self, key, active: np.ndarray) -> np.ndarray:
        """Complex mode values of one compiled transition field on ``active`` modes."""
        spectrum = self.transition_spectra[key]
        if isinstance(spectrum, CompactSpectrum) and spectrum._dense_override is None:
            support = self.supports[key]
            modes = (spectrum.local_spectrum if len(spectrum.local_spectrum) == 1 else
                     np.fft.ifft(spectrum.local_spectrum, norm="ortho"))
            result = np.zeros(len(active), dtype=complex)
            result[np.searchsorted(active, support)] = modes
            return result
        return np.fft.ifft(np.asarray(spectrum), norm="ortho")[active]

    def _mentions(self, text):
        words = _terms(text)
        found = set()
        for position, word in enumerate(words):
            for phrase, concept in self._concept_heads.get(word, ()):
                if words[position:position+len(phrase)] == phrase:
                    found.add(concept)
        # Prefer the complete registered phrase to its incidental subphrase.
        return {concept for concept in found if not any(concept != other and
                (" "+concept+" ") in (" "+other+" ") for other in found)}

    def analyze_question(self, prompt, context=None):
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        concepts = self._mentions(prompt)
        query_terms = set(_terms(prompt))
        # Interrogative/facet words are not the requested entity merely because
        # an unrelated source sentence happened to register them as a subject.
        concepts = {concept for concept in concepts if not set(concept.split()) <= (QUERY_FILLERS|QUESTION_PREDICATES)}
        used_context = False
        if not concepts and REFERENCES.search(prompt):
            previous = (context or {}).get("subjects", [])
            concepts = {term for term in previous if isinstance(term, str) and term in self.concepts}
            used_context = bool(concepts)
        found_facets = [(match.start(), facet) for facet, pattern in FACET_PATTERNS.items()
                        if (match := re.search(pattern, prompt, re.I))]
        facets = [facet for _, facet in sorted(found_facets)] or ["about"]
        # Explicit multiple concepts require a shared relation, unless the user
        # explicitly asks for their separate definitions or units.
        if len(concepts) > 1 and facets == ["about"]:
            facets = ["relation"]
        groups_by_facet = {}
        if concepts:
            compatible = set.intersection(*(self.group_indices_by_concept[concept] for concept in concepts))
            for facet in facets:
                groups_by_facet[facet] = sorted(group for group in compatible
                    if self.groups[group][1] == facet and
                    (facet == "relation" or self.groups[group][0] in concepts))
        covered = {word for concept in concepts for word in concept.split()}
        unknown = sorted(query_terms-covered-QUERY_FILLERS-QUESTION_PREDICATES)
        if used_context:
            unknown = [word for word in unknown if not REFERENCES.fullmatch(word)]
        negated = bool(re.search(r"(?<![\w-])(?:nicht|kein|keine|keinen|keinem|keiner|keines)(?![\w-])",prompt,re.I))
        supported = bool(concepts) and not unknown and not negated and all(groups_by_facet.get(facet) for facet in facets)
        return {"supported": supported, "subjects": sorted(concepts), "facets": facets,
                "groups_by_facet": groups_by_facet, "used_context": used_context,
                "unbound_terms":unknown,"unsupported_negation":negated,
                "reason": "" if supported else "no_supported_information_binding"}

    def can_handle(self, prompt, context=None):
        return self.analyze_question(prompt, context)["supported"]

    def prompt_field(self, prompt, context=None, context_field=None, *, facet=None):
        analysis = self.analyze_question(prompt, context)
        groups = sorted({group for name, values in analysis["groups_by_facet"].items()
                         if facet is None or name == facet for group in values})
        token_amplitudes = np.zeros(self.size)
        for token in tokenize(prompt):
            if token in self.index and token not in STOPWORDS and (token.isalpha() or token[:1].isdigit()):
                token_amplitudes[self.index[token]] += 1
        token_amplitudes = _unit(token_amplitudes)
        if context_field is not None:
            prior = np.asarray(context_field, dtype=float)
            if prior.shape != (self.size,) or not np.isfinite(prior).all() or (prior < 0).any():
                raise ValueError("context_field needs one finite nonnegative amplitude per token")
            token_amplitudes = _unit(token_amplitudes + _unit(prior))
        features = np.ones(len(groups))/math.sqrt(len(groups)) if groups else np.array([])
        spectrum = np.fft.fft(features, norm="ortho") if groups else np.array([], dtype=complex)
        return InformationPrompt(token_amplitudes, np.array(groups, dtype=np.intp), spectrum), analysis

    def next_distribution(self, prefix_tokens, prompt_field, *, method="operator", time_s=0.0,
                          phase_error=0.0, prefix_enabled=True, prompt_gain=1.0, data_gain=None,
                          order_policy="deepest"):
        """Read the all-data wave out in the presence of the prompt wave.

        One symbol step of one answer. The compatible transition fields are
        superposed coherently and weighted by how strongly each field resonates
        with the prompt wave. Two modulations then act on that data oscillation:
        the prompt wave boosts every candidate mode it shares, and the all-data
        wave adds a corpus-frequency prior over the candidate modes. The measured
        intensity gives the next-symbol probabilities.

        ``order_policy="deepest"`` keeps the fields of the longest available
        context (the tested recipe). ``order_policy="all"`` superposes every order;
        it is exposed for experiments and needs length-normalized decoding, since
        early termination otherwise wins on cumulative log probability.
        """
        if method not in {"operator", "direct"}:
            raise ValueError("method must be operator or direct")
        if order_policy not in {"deepest", "all"}:
            raise ValueError("order_policy must be deepest or all")
        if data_gain is None:
            data_gain = self.data_gain
        if not np.isfinite([time_s, phase_error, prompt_gain, data_gain]).all() \
                or prompt_gain < 0 or data_gain < 0:
            raise ValueError("Invalid interference controls")
        if not isinstance(prompt_field, InformationPrompt):
            raise ValueError("Expected numerical InformationPrompt")
        field = prompt_field
        if (field.token_amplitudes.shape != (self.size,) or not np.isfinite(field.token_amplitudes).all()
                or field.feature_indices.ndim != 1 or field.feature_spectrum.shape != field.feature_indices.shape
                or not np.isfinite(field.feature_spectrum).all()
                or np.any(field.feature_indices < 0) or np.any(field.feature_indices >= len(self.groups))):
            raise ValueError("Information prompt field does not match the model")
        feature_modes = np.abs(np.fft.ifft(field.feature_spectrum, norm="ortho")) if len(field.feature_spectrum) else np.array([])
        active_features = [(int(group), float(value)) for group, value in zip(field.feature_indices, feature_modes) if value > 1e-12]
        if not active_features:
            raise ValueError("No nonzero numerical information binding")
        prompt_wave = self.prompt_wave(field)
        prefix = [BOS]*self.order + list(prefix_tokens)
        addresses = []
        depths = range(self.order + 1) if order_policy == "all" else \
            ([0] if not prefix_enabled else range(self.order, -1, -1))
        for depth in depths:
            key = tuple(prefix[-depth:]) if depth else ()
            for group, value in active_features:
                if (group, key) in self.transition_spectra:
                    addresses.append((int(depth), (group, key), float(value)))
            if addresses and order_policy == "deepest":
                break
        if not prefix_enabled and addresses:
            addresses = [item for item in addresses if item[0] == 0]
        if not addresses:
            raise ValueError("No data wave for the numerical information binding")
        active = np.unique(np.concatenate([self.supports[key] for _, key, _ in addresses]))
        amplitudes = np.zeros(len(active), dtype=complex)
        field_resonance = {}
        for depth, key, value in addresses:
            modes = self._field_modes(key, active)
            support = self.supports[key]
            local_overlap = float(np.abs(np.vdot(modes[np.searchsorted(active, support)],
                                                 prompt_wave.restrict(support))))
            field_resonance["%d|%d|%s" % (depth, key[0], "|".join(key[1]))] = local_overlap
            amplitudes += value*(self.resonance_floor + local_overlap)*modes
        energy = float(np.vdot(amplitudes, amplitudes).real)
        if not np.isfinite(energy) or energy <= 1e-24:
            raise ValueError("Data wave has no supported finite energy")
        amplitudes /= math.sqrt(energy)
        data_wave_field = self.data_wave()
        carrier_size = next_fast_len(len(active))
        operator_readout, direct_readout, difference = interference_pair(
            active, amplitudes, prompt_wave, coupling=prompt_gain, phase_error=phase_error,
            mode_prior=self._mode_prior(data_wave_field, active, data_gain),
            carrier_size=carrier_size)
        if difference > VERIFY_TOLERANCE:
            raise ValueError("Spectral operator and direct readout disagree")
        readout = operator_readout if method == "operator" else direct_readout
        local_probabilities, total_intensity = probabilities(readout)
        probabilities_out = np.zeros(self.size)
        probabilities_out[active] = local_probabilities
        trace = {"runtime_source":"imported_complex_information_transition_spectra",
                 "method":method,"prefix":list(prefix_tokens[-self.order:]),
                 "active_token_indices":active.tolist(), "carrier_size":carrier_size,
                 "known_vocabulary_size":self.size, "all_supported_modes_included":True,
                 "support_rule":"coherent_superposition_of_bound_fields_"+order_policy+"_order",
                 "field_addresses":[{"group":key[0],"prefix":list(key[1]),"order":depth}
                                    for depth, key, _ in addresses],
                 "field_resonance":{name:float(value) for name, value in field_resonance.items()},
                 "feature_field_energy":float(np.vdot(field.feature_spectrum,field.feature_spectrum).real),
                 "mixed_data_energy_before_normalization":energy,
                 "data_wave":{"mode_count":data_wave_field.mode_count,"carrier_size":data_wave_field.carrier_size,
                              "energy":data_wave_field.energy,"fingerprint":data_wave_field.fingerprint(),
                              "gain":float(data_gain)},
                 "prompt_wave":{"mode_count":prompt_wave.mode_count,"carrier_size":prompt_wave.carrier_size,
                                "energy":prompt_wave.energy,"fingerprint":prompt_wave.fingerprint(),
                                "gain":float(prompt_gain)},
                 "wave_resonance":resonance(data_wave_field,prompt_wave),
                 "interference_verification":{"max_error":difference,"tolerance":VERIFY_TOLERANCE,
                                              "agrees":bool(difference<=VERIFY_TOLERANCE)},
                 "total_intensity":float(total_intensity),
                 "fft_roundtrip_max_error":difference,
                 "time_s":float(time_s),"phase_error":float(phase_error),
                 "coefficient_rule":"square_root_observed_transition_probability_then_unit_energy_superposition"}
        return probabilities_out, trace


    def render(self, tokens):
        result = ""
        sentence_start = True
        previous = [BOS,BOS]
        for token in tokens:
            if token in {BOS, EOS}:
                sentence_start = True
                previous = [BOS,BOS]
                continue
            word = token if token in STOPWORDS else self.context_forms.get(
                (tuple(previous[-2:]),token),self.display_tokens.get(token,token))
            if sentence_start and word.isalpha() and len(word) > 1:
                word = word[:1].upper()+word[1:]
            if not result or token in ".,!?;:%)]}":
                result += word
            elif result[-1:] in "([{":
                result += word
            else:
                result += " "+word
            abbreviation_dot = token=="." and (previous[-1] in ABBREVIATIONS or previous[-1].isdigit())
            sentence_start = token in ".!?" and not abbreviation_dot
            previous.append(token)
        return result

    def generate(self, prompt, *, context=None, context_field=None, time_s=0.0, max_tokens=64,
                 decoding="beam", beam_width=4, seed=17, method="operator", phase_error=0.0,
                 prefix_enabled=True, prompt_gain=1.0, data_gain=None, order_policy="deepest"):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 256:
            raise ValueError("max_tokens must be an integer from 1 to 256")
        if type(beam_width) is not int or not 1 <= beam_width <= 16:
            raise ValueError("beam_width must be an integer from 1 to 16")
        if decoding not in {"beam", "greedy", "sample"}:
            raise ValueError("decoding must be beam, greedy or sample")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        analysis = self.analyze_question(prompt, context)
        state = {"subjects":analysis["subjects"],"facets":analysis["facets"]}
        if not analysis["supported"]:
            return {"text":"","tokens":[],"trace":[],"ended":True,"reason":analysis["reason"],
                    "analysis":analysis,"information_state":state,"uses_answer_candidates":False}
        rng = np.random.default_rng(seed)
        width = beam_width if decoding == "beam" else 1
        tokens, traces, complete = [], [], True
        for facet in analysis["facets"]:
            field, _ = self.prompt_field(prompt, context, context_field, facet=facet)
            beams = [(0.0, [], [], False)]
            remaining = max_tokens-len(tokens)
            for step in range(remaining):
                candidates = []
                for score, prefix, history, ended in beams:
                    if ended:
                        candidates.append((score,prefix,history,ended))
                        continue
                    distribution, trace = self.next_distribution(prefix,field,method=method,time_s=time_s,
                        phase_error=phase_error,prefix_enabled=prefix_enabled,prompt_gain=prompt_gain,
                        data_gain=data_gain,order_policy=order_policy)
                    # An explicit finite-state cycle guard, not a fitted score
                    # penalty: do not traverse the identical trigram twice.
                    for index in trace["active_token_indices"]:
                        triple = tuple(prefix[-2:]+[self.vocabulary[index]])
                        if len(triple)==3 and any(tuple(prefix[i:i+3])==triple for i in range(len(prefix)-2)):
                            distribution[index]=0
                    if distribution.sum() <= 0:
                        continue
                    distribution /= distribution.sum()
                    active = np.flatnonzero(distribution)
                    if decoding=="sample":
                        choices = [int(rng.choice(self.size,p=distribution))]
                    else:
                        choices = active[np.argsort(-distribution[active],kind="stable")[:width]].tolist()
                    for selected in choices:
                        token = self.vocabulary[selected]
                        next_prefix = prefix+([token] if token!=EOS else [])
                        selected_trace = {**trace,"step":len(traces)+step,"facet":facet,
                            "selected_token":token,"selected_bin":selected,
                            "selected_frequency_hz":float(self.frequencies[selected]),
                            "probability":float(distribution[selected])}
                        # Sentence completion is observed EOS, not every period:
                        # abbreviation and ordinal dots must retain continuations.
                        finished = token==EOS
                        candidates.append((score+math.log(distribution[selected]),next_prefix,
                                           history+[selected_trace],finished))
                if not candidates:
                    break
                candidates.sort(key=lambda item:(-round(item[0],12),tuple(item[1])))
                beams = candidates[:width]
                if all(item[3] for item in beams):
                    break
            finished = [beam for beam in beams if beam[3]]
            _, selected_tokens, selected_traces, ended = (finished or beams)[0]
            tokens.extend(selected_tokens)
            traces.extend(selected_traces)
            complete = complete and ended
            if len(tokens)>=max_tokens:
                complete = complete and facet==analysis["facets"][-1]
                break
        return {"text":self.render([trace["selected_token"] for trace in traces]),"tokens":tokens,"trace":traces,"ended":complete,
                "reason":"" if complete else "token_budget_or_cycle_limit",
                "analysis":analysis,"information_state":state,"uses_answer_candidates":False,
                "coefficient_adaptation":"observed word-transition counts; no gradient optimization",
                "decoding":decoding,"corpus_digest":self.corpus_digest}

    def context_snapshot(self, context_field, time_s=0.0, points=128):
        values = np.asarray(context_field,dtype=float)
        if values.shape!=(self.size,) or not np.isfinite(values).all() or (values<0).any():
            raise ValueError("Context needs one finite nonnegative token amplitude per symbol")
        if not np.isfinite(time_s) or type(points) is not int or points<2:
            raise ValueError("Invalid context display controls")
        values = _unit(values)
        active = np.flatnonzero(values)
        if not len(active):
            return {"displacement":[],"quadrature":[],"energy":0.0,"mode_count":0,"carrier_size":0}
        length = next_fast_len(len(active))
        phases = np.exp(2j*np.pi*np.remainder(self.frequencies[active]*time_s,1.0))
        signal = np.fft.fft(values[active]*phases,n=length,norm="ortho")
        selection = np.linspace(0,length-1,min(points,length)).astype(int)
        return {"time_s":float(time_s),"displacement":signal.real[selection].tolist(),
                "quadrature":signal.imag[selection].tolist(),"energy":float(np.vdot(values,values).real),
                "mode_count":len(active),"carrier_size":length,"active_token_indices":active.tolist(),
                "known_vocabulary_size":self.size,
                "representation":"exact_nonzero_context_modes_before_sentence_specific_support_projection"}

    def storage_stats(self):
        if isinstance(self.transition_spectra, PackedTransitionSpectra):
            coefficients = self.transition_spectra.coefficient_nbytes
            indices = self.transition_spectra.token_indices.nbytes
        else:
            coefficients = sum(field.nbytes for field in self.transition_spectra.values())
            indices = sum(support.nbytes for support in self.supports.values())
        return {"record_count":self.record_count,"statement_count":self.statement_count,"vocabulary_size":self.size,
                "group_count":len(self.groups),"prefix_fields":len(self.transition_spectra),"observed_edges":self.edge_count,
                "unaddressed_statement_count":self.unaddressed_statement_count,
                "context_surface_forms":len(self.context_forms),
                "complex_coefficient_bytes":coefficients,"support_index_bytes":indices,
                "numerical_payload_bytes":coefficients+indices,
                "no_gradient_optimization":True,"no_qa_pairs":True,
                "excludes":"Python objects, token strings and temporary compilation counts"}

    def model_signature(self):
        digest = hashlib.sha256()
        digest.update(json.dumps(self.vocabulary,ensure_ascii=False,separators=(",",":")).encode())
        for key in sorted(self.transition_spectra):
            digest.update(json.dumps(key,separators=(",",":")).encode())
            if isinstance(self.transition_spectra, PackedTransitionSpectra):
                support, coefficients = self.transition_spectra.raw_field(key)
                digest.update(support.tobytes())
                digest.update(coefficients.tobytes())
            else:
                field = self.transition_spectra[key]
                digest.update(self.supports[key].tobytes())
                digest.update(field.local_spectrum.tobytes() if isinstance(field,CompactSpectrum) else np.asarray(field).tobytes())
        rendering_digest = hashlib.sha256(json.dumps(sorted(self.context_forms.items()),
            ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        return {"corpus_digest":self.corpus_digest,"coefficient_sha256":digest.hexdigest(),
                "rendering_sha256":rendering_digest,
                "order":self.order,"coefficient_rule":"sqrt(count/sum(count))","storage":self.storage_stats()}
