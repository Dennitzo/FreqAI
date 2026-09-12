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
from .generative import BOS, EOS, spectral_convolution, stable_frequency
from .spectral_storage import CompactSpectrum


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


class InformationWaveModel:
    def __init__(self, records, order=2):
        if type(order) is not int or not 1 <= order <= 5:
            raise ValueError("order must be an integer from 1 to 5")
        self.order = order
        normalized = []
        seen_records = set()
        self.input_record_count = 0
        digest = hashlib.sha256()
        for item in records:
            self.input_record_count += 1
            if isinstance(item, str):
                text, title = item, ""
            elif isinstance(item, dict):
                text = item.get("text", "")
                title = item.get("title", item.get("provenance", {}).get("article_title", ""))
            else:
                text, title = getattr(item, "text", ""), getattr(item, "title", "")
            if not isinstance(text, str) or not isinstance(title, str):
                raise ValueError("Information records need string text and optional string title")
            if not text.strip():
                continue
            if "\n" in text:
                first, remainder = text.split("\n", 1)
                if first.strip() and len(first) <= 100 and not re.search(r"[.!?]", first) and (not title or first.strip()==title.strip()):
                    title, text = title or first.strip(), remainder
            text = text.strip()
            identity = (title.strip(), re.sub(r"\s+", " ", text))
            if identity in seen_records:
                continue
            seen_records.add(identity)
            sentences = _sentences(text)
            if not sentences:
                continue
            anchor = _concept(title) or next((_subject(sentence) for sentence in sentences if _subject(sentence)), "")
            normalized.append((anchor, sentences))
            digest.update(json.dumps((title, text), ensure_ascii=False, separators=(",", ":")).encode())
        self.corpus_digest = digest.hexdigest()
        self.record_count = len(normalized)
        self.concepts = set()
        for anchor, sentences in normalized:
            if anchor:
                self.concepts.add(anchor)
            self.concepts.update(subject for sentence in sentences if (subject := _subject(sentence)))
            self.concepts.update(concept for sentence in sentences for concept in _relation_arguments(sentence))
            self.concepts.update(owner for sentence in sentences if (owner := _unit_owner(sentence)))
        self._concept_heads = defaultdict(list)
        for concept in sorted(self.concepts):
            words = tuple(concept.split())
            self._concept_heads[words[0]].append((words, concept))
        prepared = []
        surface_forms = defaultdict(Counter)
        self.statement_count = 0
        self.unaddressed_statement_count = 0
        for anchor, sentences in normalized:
            previous = anchor
            for sentence in sentences:
                subject = _subject(sentence) or previous or anchor
                if not subject:
                    self.unaddressed_statement_count += 1
                previous = subject
                concepts = self._mentions(sentence) | ({subject} if subject else set())
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
                self.statement_count += 1
                for surface in TOKEN_RE.findall(sentence):
                    surface_forms[surface.casefold()][surface] += 1
        self.vocabulary = sorted({token for _, _, _, tokens in prepared for token in tokens} | {EOS})
        self.index = {token: i for i, token in enumerate(self.vocabulary)}
        self.size = len(self.vocabulary)
        self.frequencies = np.array([stable_frequency(token) for token in self.vocabulary])
        self.token_frequencies = self.frequencies
        self.carrier_size = next_fast_len(self.size)  # Only the full context display.
        self.display_tokens = {token: counts.most_common(1)[0][0] for token, counts in surface_forms.items()}
        # Preserve context-dependent capitalization only for ambiguous symbols.
        # These are short token-prefix statistics, never stored sentence pieces.
        ambiguous = {token for token, counts in surface_forms.items() if len(counts)>1 and token not in STOPWORDS}
        local_forms = defaultdict(Counter)
        for _, sentences in normalized:
            for sentence in sentences:
                previous = [BOS,BOS]
                for surface in TOKEN_RE.findall(sentence):
                    token = surface.casefold()
                    if token in ambiguous:
                        local_forms[(tuple(previous[-2:]),token)][surface] += 1
                    previous.append(token)
        self.context_forms = {key:sorted(counts,key=lambda form:(-counts[form],form))[0]
                              for key,counts in local_forms.items()}
        descriptors = sorted({(subject, facet, concepts) for subject, concepts, facets, _ in prepared for facet in facets})
        self.groups = tuple(descriptors)
        group_index = {descriptor: i for i, descriptor in enumerate(self.groups)}
        self.group_indices_by_concept = defaultdict(set)
        for i, (subject, facet, concepts) in enumerate(self.groups):
            for concept in concepts:
                self.group_indices_by_concept[concept].add(i)
        counts = defaultdict(Counter)
        for subject, concepts, facets, tokens in prepared:
            groups = [group_index[(subject, facet, concepts)] for facet in facets]
            extended = [BOS]*order + tokens + [EOS]
            for position in range(order, len(extended)):
                target = self.index[extended[position]]
                for depth in range(order+1):
                    prefix = tuple(extended[position-depth:position]) if depth else ()
                    for group in groups:
                        counts[(group, prefix)][target] += 1
        self.transition_spectra = {}
        self.supports = {}
        for key, counter in counts.items():
            indices = np.array(sorted(counter), dtype=np.intp)
            values = np.array([counter[int(i)] for i in indices], dtype=float)
            amplitudes = np.sqrt(values/values.sum())
            self.supports[key] = indices
            self.transition_spectra[key] = CompactSpectrum(self.size, indices, np.fft.fft(amplitudes, norm="ortho"))
        self.edge_count = sum(len(indices) for indices in self.supports.values())
        # No complete statements, document IDs, real transition counts or answer
        # candidates survive compilation. Group descriptors contain concepts only.

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
                          phase_error=0.0, prefix_enabled=True, prompt_gain=1.0):
        if method not in {"operator", "direct"}:
            raise ValueError("method must be operator or direct")
        if not np.isfinite([time_s, phase_error, prompt_gain]).all() or prompt_gain < 0:
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
        prefix = [BOS]*self.order + list(prefix_tokens)
        fields = []
        for depth in range(self.order if prefix_enabled else 0, -1, -1):
            key = tuple(prefix[-depth:]) if depth else ()
            fields = [((group, key), value) for group, value in active_features if (group, key) in self.transition_spectra]
            if fields:
                break
        if not fields:
            raise ValueError("No data wave for the numerical information binding")
        active = np.unique(np.concatenate([self.supports[key] for key, _ in fields]))
        amplitudes = np.zeros(len(active), dtype=complex)
        for key, value in fields:
            spectrum = self.transition_spectra[key]
            if isinstance(spectrum, CompactSpectrum) and spectrum._dense_override is None:
                modes = np.fft.ifft(spectrum.local_spectrum, norm="ortho")
                amplitudes[np.searchsorted(active, spectrum.token_indices)] += value*modes
            else:
                amplitudes += value*np.fft.ifft(np.asarray(spectrum), norm="ortho")[active]
        energy = float(np.vdot(amplitudes, amplitudes).real)
        if not np.isfinite(energy) or energy <= 1e-24:
            raise ValueError("Data wave has no supported finite energy")
        amplitudes /= math.sqrt(energy)
        carrier_size = next_fast_len(len(active))
        phases = np.exp(2j*np.pi*np.remainder(self.frequencies[active]*time_s, 1.0))
        guide = field.token_amplitudes[active]
        if method == "operator":
            data = np.fft.fft(amplitudes, n=carrier_size, norm="ortho")
            carrier = np.fft.fft(phases, n=carrier_size, norm="ortho")
            moving = spectral_convolution(data, carrier)
            prompt = np.fft.fft(guide, n=carrier_size, norm="ortho")
            result = moving + prompt_gain*np.exp(1j*phase_error)*spectral_convolution(moving, prompt)
        else:
            result = np.fft.fft(amplitudes*phases*(1+prompt_gain*guide*np.exp(1j*phase_error)),
                                n=carrier_size, norm="ortho")
        decoded = np.fft.ifft(result, norm="ortho")
        scores = np.abs(decoded[:len(active)])**2
        if not np.isfinite(scores).all() or scores.sum() <= 1e-24:
            raise ValueError("Interference has no supported finite energy")
        local_probabilities = np.round(scores/scores.sum(), 12)
        local_probabilities /= local_probabilities.sum()
        probabilities = np.zeros(self.size)
        probabilities[active] = local_probabilities
        trace = {"runtime_source":"imported_complex_information_transition_spectra",
                 "method":method,"prefix":list(prefix_tokens[-self.order:]),
                 "active_token_indices":active.tolist(), "carrier_size":carrier_size,
                 "known_vocabulary_size":self.size, "all_supported_modes_included":True,
                 "support_rule":"union_of_all_edges_of_longest_available_conjunctive_prefix_fields",
                 "field_addresses":[{"group":key[0],"prefix":list(key[1])} for key, _ in fields],
                 "feature_field_energy":float(np.vdot(field.feature_spectrum,field.feature_spectrum).real),
                 "mixed_data_energy_before_normalization":energy,
                 "fft_roundtrip_max_error":float(np.max(np.abs(np.fft.fft(decoded,norm="ortho")-result))),
                 "time_s":float(time_s),"phase_error":float(phase_error),
                 "coefficient_rule":"square_root_observed_transition_probability_then_unit_energy_superposition"}
        return probabilities, trace

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
                 prefix_enabled=True, prompt_gain=1.0):
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
                        phase_error=phase_error,prefix_enabled=prefix_enabled,prompt_gain=prompt_gain)
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
            field = self.transition_spectra[key]
            digest.update(json.dumps(key,separators=(",",":")).encode())
            digest.update(self.supports[key].tobytes())
            digest.update(field.local_spectrum.tobytes() if isinstance(field,CompactSpectrum) else np.asarray(field).tobytes())
        rendering_digest = hashlib.sha256(json.dumps(sorted(self.context_forms.items()),
            ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        return {"corpus_digest":self.corpus_digest,"coefficient_sha256":digest.hexdigest(),
                "rendering_sha256":rendering_digest,
                "order":self.order,"coefficient_rule":"sqrt(count/sum(count))","storage":self.storage_stats()}
