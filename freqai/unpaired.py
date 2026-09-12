"""A single information-only Fourier decoder with an explicit German grammar.

The corpus supplies declarations and lexical facts, never dialogue pairs.
Linguistic rules below provide question analysis, person inflection and a small
finite-state conversation grammar. Those rules are authored knowledge, not
emergent intelligence. Complex coefficients are calculated, not optimized.

An utterance is a sequence of grammatical *roles*. No complete response string
is constructed before decoding. At every position the compatible lexical modes
of those roles are superposed, interfered with the prompt and inverse transformed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np
from scipy.fft import next_fast_len

from .features import STOPWORDS, canonical_term
from .information import (InformationPrompt, InformationWaveModel, TOKEN_RE,
                          QUERY_FILLERS, QUESTION_PREDICATES, FACET_PATTERNS,
                          REFERENCES, ABBREVIATIONS, _concept, _sentences, _terms, _unit, tokenize)
from .parallel import map as parallel_map
from .spectral_storage import PackedTransitionSpectra
from .spectral_ops import BOS, EOS, spectral_convolution, stable_frequency
from .waves import (VERIFY_TOLERANCE, WaveField, interference_pair, probabilities,
                    resonance)


# Explicit function-word grammar, not example responses or topic vocabulary.
FUNCTION_WORDS = "ich du bin bist ist sind war waren habe hast hat heiße heißt kann kannst können mir dir mich dich es und aber auch keine keinen kein eigenen eigene dein deine deiner deines deinen deinem du wie was geht heute jetzt mag magst möchte möchtest hätte hättest suche suchst gern beim da ? . , !".split()
BYTE_TOKENS = [f"<byte:{value:02x}>" for value in range(256)]
PERSON_ONE = {"ist":"bin", "sind":"bin", "hat":"habe", "heißt":"heiße",
              "heisst":"heiße", "kann":"kann", "besitzt":"besitze"}
PERSON_TWO = {"ist":"bist", "sind":"bist", "hat":"hast", "heißt":"heißt",
              "heisst":"heißt", "kann":"kannst", "besitzt":"besitzt"}
CLAUSE = re.compile(r"^\s*(?P<subject>.+?)\s+(?P<verb>ist|sind|hat|heißt|heisst|kann|besitzt|beträgt)\s+(?P<object>.+?)[.!?]?$", re.I)
LEXICON = re.compile(r"\b(?:Das|Die|Der)\s+(?:Wort|Ausdruck|Wendung)\s+[\"„]?([^\"“]+?)[\"“]?\s+(?:ist|gilt|wird)\b.+?\b(Begrüßungswort|Begrüßungsformel|Grußwort|Dankeswort|Dankesformel|Dankwort|Höflichkeitswort|Höflichkeitsformel|Abschiedswort|Abschiedsformel|Zustimmungswort|Zustimmungsformel|Verneinungswort|Verneinungsformel)\b", re.I)
LEXICAL_CATEGORY = {"begrüssungswort":"greeting", "grusswort":"greeting",
                    "dankeswort":"thanks", "dankwort":"thanks", "höflichkeitswort":"courtesy",
                    "abschiedswort":"farewell", "zustimmungswort":"affirmation", "verneinungswort":"negation"}
LEXICAL_CATEGORY.update({"begrüssungsformel":"greeting","dankesformel":"thanks",
    "höflichkeitsformel":"courtesy","abschiedsformel":"farewell","zustimmungsformel":"affirmation",
    "verneinungsformel":"negation"})
DEIXIS = {"ich":"du","mich":"dich","mir":"dir","mein":"dein","meine":"deine",
          "meiner":"deiner","meines":"deines","meinen":"deinen","meinem":"deinem"}
REGULAR_USER_VERBS = {"komme":"kommst","gehe":"gehst","höre":"hörst","lese":"liest",
                     "sehe":"siehst","freue":"freust","fühle":"fühlst","suche":"suchst"}
QUERY_GRAMMAR = frozenset(canonical_term(w) for w in """
was wer wen wem wie wieso warum weshalb wo wann welches welche welcher welchen welchem
mit in von für auf an durch unter über man bitte doch denn eigentlich darunter versteht verstehen
erklärt erklärt mir uns du kann kannst können könnte möchtest sollte soll brauche braucht brauchen
besitzt betraegt beträgt wert wertes gehört gehoert gehören gehoeren zugeordnet zuordnung genannt
größe grösse größe groesse masse maß misst gemessen si zu dessen deren
""".split()) | QUERY_FILLERS | QUESTION_PREDICATES


def _parse_declarations(text):
    """Parse one independent prose record; role IDs are assigned in corpus order."""
    if "\n" in text:
        first, rest = text.split("\n", 1)
        if len(first) < 100 and not re.search(r"[.!?]", first):
            text = rest
    declarations = []
    for sentence in _sentences(text):
        lexical = LEXICON.search(sentence)
        if lexical:
            words = tokenize(lexical.group(1).strip())
            if 0 < len(words) <= 6:
                category = LEXICAL_CATEGORY[lexical.group(2).casefold()]
                spellings = tuple(TOKEN_RE.findall(lexical.group(1).strip()))
                declarations.append(("lexical", category, words, spellings))
        match = CLAUSE.match(sentence)
        if not match:
            continue
        subject_text, verb, object_text = match.group("subject", "verb", "object")
        subject_tokens, object_tokens = tokenize(subject_text), tokenize(object_text)
        if len(subject_tokens) > 10 or len(object_tokens) > 64:
            continue
        subject = _concept(subject_text)
        if not subject or REFERENCES.match(subject_text.strip()) or subject_text.lower().startswith(
                ("wenn ", "dann ", "für ", "bei ")):
            continue
        while object_tokens and object_tokens[-1] in ".!?":
            object_tokens.pop()
        if not object_tokens:
            continue
        attributes = frozenset(_terms(object_text))
        ownership = re.match(r"^(?:(?:der|die|das)\s+)?(.+?)\s+(?:von|des|der)\s+(.+)$", subject_text, re.I)
        owner = ""
        if ownership and verb.casefold() in {"beträgt", "ist", "hat"}:
            owner = _concept(ownership.group(2))
            attributes = attributes | frozenset(_terms(ownership.group(1)))
        action = bool(verb.casefold() == "ist" and
                      re.match(r"das\s+\w+(?:en|eln|ern)\b", object_text, re.I) and len(subject_tokens) == 1)
        state = any(term in subject for term in ("befinden", "zustand", "gefühl", "gefuehl"))
        declarations.append(("fact", subject, verb.casefold(), subject_tokens, object_tokens,
                             attributes, owner, action, state))
    return declarations


@dataclass
class LexicalRole:
    """Ordered symbols of one factual role, stored in complex coordinates.

    A role is an entity label or predicate argument, never a complete answer.
    Positions are distinct modes, including repeated lexical symbols.
    """
    indices: np.ndarray
    spectrum: np.ndarray
    origin: str
    surfaces: dict | None = None

    def amplitudes(self):
        return np.fft.ifft(self.spectrum, norm="ortho") if len(self.spectrum) else np.array([], dtype=complex)


@dataclass(frozen=True)
class Fact:
    subject: str
    verb: str
    subject_role: int
    verb_role: int
    object_role: int
    attributes: frozenset
    owner: str = ""


@dataclass
class UnpairedPrompt(InformationPrompt):
    segments: tuple = ()
    evidence: tuple = ()
    numerical_gate: complex = 1.0
    temporary_roles: dict | None = None

    def __mul__(self, scalar):
        return UnpairedPrompt(self.token_amplitudes*scalar, self.feature_indices.copy(),
                              self.feature_spectrum*scalar, self.segments, self.evidence,
                              self.numerical_gate*scalar,self.temporary_roles)

    __rmul__ = __mul__


class UnpairedWaveModel(InformationWaveModel):
    """Compile all text declarations once; support fact and social utterances."""

    def __init__(self, records, order=2, *, previous_model=None):
        records = list(records)
        from .compiler_cache import _input_digest
        try:
            self._compiler_input_digest = _input_digest(records)
        except (TypeError, ValueError):
            self._compiler_input_digest = None
        for record in records:
            if isinstance(record,dict) and any(key in record for key in {
                "question","answer","response","instruction","messages","conversations","turns",
                "user","assistant","pairs","pair","dialogue","dialog","cue","input","output"}):
                raise ValueError("UnpairedWaveModel accepts information declarations, not question/answer or dialogue fields")
            cue = record.get("prompt", "") if isinstance(record, dict) else getattr(record, "prompt", "")
            if cue:
                raise ValueError("UnpairedWaveModel accepts information declarations, not prompt/answer pairs")
        super().__init__(records, order=order, previous_model=previous_model)
        missing = sorted(set(FUNCTION_WORDS + BYTE_TOKENS + list(PERSON_ONE.values()) + list(PERSON_TWO.values()))-set(self.vocabulary))
        self.vocabulary.extend(missing)
        self.index = {token:index for index,token in enumerate(self.vocabulary)}
        self.size = len(self.vocabulary)
        self.frequencies = np.array([stable_frequency(token) for token in self.vocabulary])
        self.token_frequencies = self.frequencies
        self.carrier_size = next_fast_len(self.size)
        # The function-word and byte grammar extends the medium after compilation;
        # extend the all-data energy vector with zero modes, not a new one.
        if len(self.symbol_energy) != self.size:
            extended = np.zeros(self.size)
            extended[:len(self.symbol_energy)] = self.symbol_energy
            self.symbol_energy = extended
        if isinstance(self.transition_spectra, PackedTransitionSpectra):
            self.transition_spectra.resize(self.size)
        else:
            for spectrum in self.transition_spectra.values():
                spectrum.size = self.size
        self.lexical_roles = []
        self._role_ids = {}
        self._role_spectrum_by_length = {}
        self.facts = []
        self.facts_by_subject = defaultdict(list)
        self.lexicon = defaultdict(list)
        self.lexicon_words = defaultdict(set)
        self.lexicon_phrases = defaultdict(set)
        self.lexicon_spellings = {}
        self.action_roles = {}
        self.state_vocabulary = set()
        texts = [record if isinstance(record, str) else record.get("text", "")
                 if isinstance(record, dict) else getattr(record, "text", "") for record in records]
        for declarations in parallel_map(_parse_declarations, texts, min_items=32):
            for declaration in declarations:
                if declaration[0] == "lexical":
                    _, category, words, spellings = declaration
                    role = self._role(words, "corpus_lexical_fact")
                    self.lexicon[category].append(role)
                    self.lexicon_words[category].update(words)
                    self.lexicon_phrases[category].add(tuple(words))
                    self.lexicon_spellings[tuple(words)] = spellings
                    continue
                _, subject, verb, subject_tokens, object_tokens, attributes, owner, action, state = declaration
                subject_role = self._role(subject_tokens, "corpus_subject")
                verb_role = self._role(tokenize(verb), "corpus_predicate")
                object_role = self._role(object_tokens, "corpus_predicate_argument")
                fact = Fact(subject, verb.casefold(), subject_role, verb_role, object_role, attributes,owner)
                fact_id = len(self.facts)
                self.facts.append(fact)
                self.facts_by_subject[subject].append(fact_id)
                if owner and owner!=subject:
                    self.facts_by_subject[owner].append(fact_id)
                if action:
                    self.action_roles[subject] = subject_role
                if state:
                    self.state_vocabulary.update(token for token in object_tokens if token.isalpha() and token not in STOPWORDS)
        for key in self.lexicon:
            self.lexicon[key] = sorted(set(self.lexicon[key]))
        self.assistant_subjects = frozenset(subject for subject in self.facts_by_subject
            if subject in {"assistent", "assistenzprogramm", "assistant"})
        # Declared aliases are data-derived: renaming the assistant changes both
        # the name response and the subject's mention dictionary.
        self.assistant_aliases = {"assistent", "assistenzprogramm"}
        for subject in self.assistant_subjects:
            for fact_id in self.facts_by_subject[subject]:
                fact = self.facts[fact_id]
                if fact.verb in {"heißt","heisst"}:
                    self.assistant_aliases.update(self._role_words(fact.object_role))
        self._grammar_roles = {}
        from .compiler_cache import _auxiliary_digest
        self._compiler_auxiliary_digest = _auxiliary_digest(self)

    def _extra_data_energy(self) -> np.ndarray:
        """Energy per symbol mode carried by the declarative corpus roles.

        Only corpus-derived roles count as data. Function-word grammar and
        session-only temporary roles are rules about an utterance, not corpus
        facts, so they must not change the all-data wave of a conversation.
        """
        energy = np.zeros(self.size)
        corpus_origins = {"corpus_subject","corpus_predicate","corpus_predicate_argument",
                          "corpus_lexical_fact"}
        for role in self.lexical_roles:
            if role.origin not in corpus_origins or not len(role.indices):
                continue
            np.add.at(energy, role.indices, 1.0/len(role.indices))
        return energy

    def _role(self, tokens, origin):
        ids = tuple(self.index[token] for token in tokens if token in self.index)
        key = (origin, ids)
        if key not in self._role_ids:
            self._role_ids[key] = len(self.lexical_roles)
            cache = getattr(self, "_role_spectrum_by_length", None)
            if cache is None:
                cache = self._role_spectrum_by_length = {}
            if len(ids) not in cache:
                values = np.ones(len(ids))/math.sqrt(len(ids)) if ids else np.array([])
                cache[len(ids)] = np.fft.fft(values,norm="ortho") if ids else np.array([],dtype=complex)
            self.lexical_roles.append(LexicalRole(np.array(ids,dtype=np.intp), cache[len(ids)].copy(), origin))
        return self._role_ids[key]

    def _encode_words(self, words):
        result = []
        previous_bytes = False
        for word in words:
            if word.casefold() in self.index:
                result.append(self.index[word.casefold()])
                previous_bytes = False
            else:
                if previous_bytes:
                    result.append(self.index["<byte:20>"])
                result.extend(self.index[f"<byte:{byte:02x}>"] for byte in word.encode("utf-8"))
                previous_bytes = True
        return np.array(result,dtype=np.intp)

    def _temporary_role(self, words):
        indices = self._encode_words(words)
        modes = np.ones(len(indices))/math.sqrt(len(indices)) if len(indices) else np.array([])
        surfaces = {}
        position = 0
        previous_bytes = False
        for word in words:
            if word.casefold() in self.index:
                surfaces[position] = word
                position += 1
                previous_bytes = False
            else:
                position += len(word.encode("utf-8")) + int(previous_bytes)
                previous_bytes = True
        return LexicalRole(indices,np.fft.fft(modes,norm="ortho") if len(modes) else np.array([],dtype=complex),"session_prompt_only",surfaces)

    def _field_role(self, role_id, field):
        return self.lexical_roles[role_id] if role_id>=0 else (field.temporary_roles or {})[role_id]

    def _role_words(self, role_id):
        return [self.vocabulary[int(index)] for index in self.lexical_roles[role_id].indices]

    def _grammar(self, *words):
        """A grammatical constituent, never a complete response sentence."""
        key = tuple(words)
        if key not in self._grammar_roles:
            if any(word not in FUNCTION_WORDS and word not in PERSON_ONE.values() and word not in PERSON_TWO.values() for word in words):
                raise ValueError("Grammar may only introduce declared function words")
            self._grammar_roles[key] = self._role(words, "explicit_function_grammar")
        return (self._grammar_roles[key],)

    def _assistant_facts(self, *, verb=None, terms=()):
        candidates = [index for subject in self.assistant_subjects for index in self.facts_by_subject[subject]]
        if verb is not None:
            candidates = [index for index in candidates if self.facts[index].verb in verb]
        if terms:
            wanted = {canonical_term(term) for term in terms}
            candidates = [index for index in candidates if self.facts[index].attributes & wanted]
        return candidates

    def _fact_segments(self, fact_id, person=3):
        fact = self.facts[fact_id]
        if person==1:
            subject = self._grammar("ich")
            verb = self._grammar(PERSON_ONE.get(fact.verb, fact.verb))
        elif person==2:
            subject = self._grammar("du")
            verb = self._grammar(PERSON_TWO.get(fact.verb, fact.verb))
        else:
            subject = (fact.subject_role,)
            verb = (fact.verb_role,)
        return [subject, verb, (fact.object_role,), self._grammar(".")]

    def _user_state(self, prompt, context):
        saved = dict((context or {}).get("user_facts", {}))
        availability = re.search(r"\bes\s+sind\s+(.+?)\s+da\b",prompt,re.I)
        if availability and len(TOKEN_RE.findall(availability.group(1)))<=16:
            saved["available"] = TOKEN_RE.findall(availability.group(1))
            saved["available_scope"] = "session_prompt_only"
        name_match = re.search(r"\b(?:ich\s+heiße|ich\s+heisse|mein\s+name\s+ist)\s+([^.!?,]+)",prompt,re.I)
        if name_match:
            name = re.split(r"\bund\b",name_match.group(1),maxsplit=1,flags=re.I)[0].strip()
            words = TOKEN_RE.findall(name)
            if 0<len(words)<=5 and len(name)<=80:
                saved["name"] = words
                saved["name_scope"] = "session_prompt_only"
                return saved,"name"
        owned = re.search(r"\b(mein(?:e|en|er|em|es)?)\s+(.+?)\s+(ist|sind|war|waren)\s+([^.!?,]+)",prompt,re.I)
        if owned and len(tokenize(owned.group(2)))<=5:
            complement = tokenize(re.split(r"\b(?:und|aber)\b",owned.group(4),maxsplit=1,flags=re.I)[0])
            if 0<len(complement)<=12:
                saved["owned_state"] = {"possessive":DEIXIS[owned.group(1).casefold()],
                    "subject":TOKEN_RE.findall(owned.group(2)),"verb":owned.group(3).casefold(),"object":complement}
                saved["owned_state_scope"] = "session_prompt_only"
                return saved,"owned_state"
        state_start = r"ich\s+bin|bin\s+ich|mir\s+geht(?:\s+es)?(?:\s+heute)?|geht(?:\s+es)?\s+mir|bei\s+mir\s+ist(?:\s+heute)?(?:\s+alles)?"
        if re.search(r"\bich\b",prompt,re.I):
            state_start += r"|und\s+bin"
        match = re.search(r"\b(?:"+state_start+r")\s+([^.!?,]+)", prompt,re.I)
        if not match:
            match = re.match(r"\s*([^,.!?]+?)\s+und\s+(?:dir|du)\b",prompt,re.I)
        if match:
            words = tokenize(re.split(r"\b(?:und|aber)\b",match.group(1),maxsplit=1,flags=re.I)[0])
            if 0<len(words)<=12:
                saved["state"] = words
                saved["state_scope"] = "session_prompt_only"
                saved["state_predicate"] = "wellbeing" if re.search(r"\b(?:mir|dir)\b",prompt,re.I) else "state"
                return saved,"state"
        preference = re.search(r"\b(?:ich\s+(mag|möchte|moechte|hätte|haette|suche)|(mag|möchte|moechte|hätte|haette|suche)\s+ich)\s+([^.!?,]+)",prompt,re.I)
        if preference:
            words = TOKEN_RE.findall(preference.group(3))
            if preference.group(2):
                prefix = re.split(r"[.!?,]",prompt[:preference.start()])[-1].strip()
                words = TOKEN_RE.findall(prefix)+words
            if 0<len(words)<=20:
                saved["preference"] = words
                verb = (preference.group(1) or preference.group(2)).casefold()
                saved["preference_verb"] = {"mag":"magst","möchte":"möchtest","moechte":"möchtest",
                    "hätte":"hättest","haette":"hättest","suche":"suchst"}[verb]
                saved["preference_scope"] = "session_prompt_only"
                return saved,"preference"
        verbs = "|".join(REGULAR_USER_VERBS)
        events = []
        for clause in re.split(r"[.!?,]",prompt):
            match = re.search(r"\b(?:ich\s+("+verbs+r")|("+verbs+r")\s+ich)\s+(.+)",clause,re.I)
            if match:
                verb = (match.group(1) or match.group(2)).casefold()
                words = TOKEN_RE.findall(match.group(3))
                prefix = TOKEN_RE.findall(clause[:match.start()])
                # Retain an explicit temporal/reference adjunct of V2 clauses.
                if match.group(2) and 0<len(prefix)<=2:
                    words += prefix
                if 0<len(words)<=16:
                    events.append({"verb":REGULAR_USER_VERBS[verb],"object":[DEIXIS.get(word.casefold(),word) for word in words]})
        if events:
            saved["events"] = events[-2:]
            saved["events_scope"] = "session_prompt_only"
            return saved,"events"
        return saved,""

    def _conversation_analysis(self, prompt, context=None):
        words = set(tokenize(prompt))
        prompt_tokens = tokenize(prompt)
        # A polite request wrapper does not change a domain question into a
        # question about the assistant's own abilities.
        if re.search(r"\b(?:erklär\w*|erklaer\w*|beschreib\w*|definier\w*)\b",prompt,re.I):
            domain = self._mentions(prompt)-self.assistant_subjects
            if domain:
                return None
        user_facts, asserted = self._user_state(prompt,context)
        state = {"user_facts":user_facts}
        segments, evidence = [], []
        temporary = {}
        def session_role(words):
            role_id = -len(temporary)-1
            temporary[role_id] = self._temporary_role(words)
            return role_id
        def contains(category):
            return any(any(tuple(prompt_tokens[i:i+len(phrase)])==phrase for i in range(len(prompt_tokens)-len(phrase)+1))
                       for phrase in self.lexicon_phrases.get(category,()))
        def assertion_segments(kind):
            result, sources = [],[]
            if kind=="owned_state":
                fact = user_facts["owned_state"]
                subject = session_role(fact["subject"])
                argument = session_role(fact["object"])
                result = [self._grammar(fact["possessive"]),(subject,),self._grammar(fact["verb"]),(argument,),self._grammar(".")]
                sources = [subject,argument]
            elif kind=="state":
                role = session_role(user_facts["state"])
                result = ([self._grammar("dir"),self._grammar("geht"),self._grammar("es")] if user_facts.get("state_predicate")=="wellbeing" else [self._grammar("du"),self._grammar("bist")])
                result += [(role,),self._grammar(".")]
                sources = [role]
            elif kind=="events":
                for event in user_facts["events"]:
                    verb = session_role([event["verb"]])
                    argument = session_role(event["object"])
                    result += [self._grammar("du"),(verb,),(argument,),self._grammar(".")]
                    sources += [verb,argument]
            return result,sources
        act = ""
        is_question = bool("?" in prompt or re.search(r"\b(?:wie|was|wer|welch\w*|kannst|hast|bist|erinner\w*)\b",prompt,re.I))
        asks_assistant = bool(words & {"du","dir","dich","dein","deine","deinen"} or words & self.assistant_aliases)
        greeting = contains("greeting") and asserted!="name"
        farewell = contains("farewell") and asserted!="name"
        thanks = contains("thanks")
        wellbeing = bool(re.search(r"\b(?:wie\s+geht|wie\s+gehts|befinden|geht\s+es\s+dir|und\s+(?:dir|du))\b",prompt,re.I))
        facts = []
        if asks_assistant and (is_question or wellbeing):
            if re.search(r"\b(?:heiß\w*|heiss\w*|name\w*)\b",prompt,re.I):
                facts = self._assistant_facts(verb={"heißt","heisst"})
            elif re.search(r"\b(?:gefühl\w*|gefuehl\w*|fühl\w*)\b",prompt,re.I):
                facts = self._assistant_facts(terms={"Gefühle","Empfindungen"})
            elif wellbeing:
                facts = self._assistant_facts(terms={"Befinden"}) or self._assistant_facts(terms={"Gefühle"})
            elif re.search(r"\b(?:kannst|könn\w*|aufgabe\w*|hilf\w*|helf\w*)\b",prompt,re.I):
                facts = self._assistant_facts(verb={"kann"})
            elif re.search(r"\b(?:wer|was|system|programm)\b|\bbist\s+du\b",prompt,re.I):
                facts = self._assistant_facts(verb={"ist"})
            if facts:
                # The discourse rule chooses a predicate, never an answer string.
                # Multiple compatible facts of the same role remain alternatives.
                fact_id = min(facts)
                segments += self._fact_segments(fact_id,person=1)
                evidence += [self.facts[index].object_role for index in facts]
                segments[-2] = tuple(self.facts[index].object_role for index in facts if self.facts[index].verb==self.facts[fact_id].verb)
                act = "answer_declared_assistant_fact"
                if asserted in {"owned_state","state","events"}:
                    acknowledgments, sources = assertion_segments(asserted)
                    segments = acknowledgments+segments
                    evidence += sources
        if not segments and farewell:
            segments += [tuple(self.lexicon["farewell"]),self._grammar("!")]
            evidence += self.lexicon["farewell"]
            act = "reciprocal_farewell"
        elif not segments and thanks and self.lexicon.get("courtesy"):
            segments += [tuple(self.lexicon["courtesy"]),self._grammar("!")]
            evidence += self.lexicon["thanks"] + self.lexicon["courtesy"]
            act = "acknowledge_thanks"
        elif not segments and greeting:
            segments += [tuple(self.lexicon["greeting"]),self._grammar("!")]
            evidence += self.lexicon["greeting"]
            # An open question is licensed only by a declared concept of human
            # wellbeing; the corpus does not contain the emitted question.
            if any(canonical_term("befinden") in subject for subject in self.facts_by_subject):
                segments += [self._grammar("wie"), self._grammar("geht"),self._grammar("es"),self._grammar("dir"),self._grammar("?")]
            act = "reciprocal_greeting"
        if not segments and asserted=="name":
            name_role = session_role(user_facts["name"])
            segments += [self._grammar("du"),self._grammar("heißt"),(name_role,),self._grammar(".")]
            evidence += [name_role]
            act = "acknowledge_session_name"
        if not segments and asserted=="preference":
            actions = [self.action_roles[canonical_term(word)] for word in user_facts["preference"] if canonical_term(word) in self.action_roles]
            negative = bool({word.casefold() for word in user_facts["preference"]} & {"nicht","kein","keine","keinen","keinem","keiner"})
            if actions and not negative:
                # A request involving a known activity leaves its object open:
                # interrogative object + modal + person + corpus action role.
                segments += [self._grammar("was"),self._grammar("möchtest"),self._grammar("du"),tuple(sorted(set(actions))),self._grammar("?")]
                evidence += actions
                act = "clarify_activity_object"
            else:
                preference_role = session_role(user_facts["preference"])
                segments += [self._grammar("du"),self._grammar(user_facts["preference_verb"]),(preference_role,),self._grammar(".")]
                evidence += [preference_role]
                act = "acknowledge_session_preference"
                if user_facts.get("available"):
                    available = session_role(user_facts["available"])
                    segments = [self._grammar("es"),self._grammar("sind"),(available,),self._grammar("da"),self._grammar(".")]+segments
                    evidence += [available]
        if not segments and asserted in {"owned_state","events"}:
            segments, evidence = assertion_segments(asserted)
            act = "acknowledge_session_"+asserted
        if not segments and asserted=="state":
            state_role = session_role(user_facts["state"])
            segments += ([self._grammar("dir"),self._grammar("geht"),self._grammar("es")] if user_facts.get("state_predicate")=="wellbeing" else [self._grammar("du"),self._grammar("bist")])
            segments += [(state_role,),self._grammar(".")]
            evidence += [state_role]
            act = "acknowledge_session_state"
        if not segments and user_facts.get("state") and re.search(r"\b(?:wie\s+geht\s+es\s+mir|wie\s+gehts\s+mir|wie\s+(?:ist|war)\s+mein\s+befinden|was\s+habe\s+ich\s+(?:gesagt|erzählt))\b",prompt,re.I):
            state_role = session_role(user_facts["state"])
            segments += ([self._grammar("dir"),self._grammar("geht"),self._grammar("es")] if user_facts.get("state_predicate")=="wellbeing" else [self._grammar("du"),self._grammar("bist")])
            segments += [(state_role,),self._grammar(".")]
            evidence += [state_role]
            act = "recall_session_state"
        if not segments and user_facts.get("name") and re.search(r"\b(?:wie\s+heiße\s+ich|wie\s+heisse\s+ich|mein\s+name)\b",prompt,re.I):
            name_role = session_role(user_facts["name"])
            segments += [self._grammar("du"),self._grammar("heißt"),(name_role,),self._grammar(".")]
            evidence += [name_role]
            act = "recall_session_name"
        # A deontic utterance constrains this conversation. It is not a negated
        # factual question. Its original wording remains a scoped user statement.
        directive = not is_question and (contains("courtesy") or bool(re.search(r"\b(?:soll|sollen)\b",prompt,re.I)))
        if not segments and directive and self.lexicon.get("affirmation"):
            user_facts["conversation_constraints"] = [prompt]
            user_facts["conversation_constraints_scope"] = "session_prompt_only"
            segments += [tuple(self.lexicon["affirmation"]),self._grammar(".")]
            evidence += self.lexicon["affirmation"]
            act = "acknowledge_conversation_constraint"
        if not segments:
            return None
        state.update({"subjects":["assistent"] if facts else [], "facets":["conversation"],"speech_act":act})
        return {"supported":True,"subjects":state["subjects"],"facets":["conversation"],
                "groups_by_facet":{},"reason":"","kind":"role_grammar","segments":tuple(segments),
                "evidence":tuple(evidence),"speech_act":act,"state":state,
                "user_facts":user_facts,"used_context":bool(context),"temporary_roles":temporary}

    def _attribute_analysis(self, prompt, context=None):
        """Bind concrete declaration roles without comparing any question pair."""
        words = set(_terms(prompt))
        # Numerical values are arguments, not statistical semantic similarity.
        values = {word for word in tokenize(prompt) if any(char.isdigit() for char in word)}
        subjects = {subject for subject in self.facts_by_subject
                    if set(subject.split()) <= words and subject not in self.assistant_subjects}
        subjects = {subject for subject in subjects if any(self.facts[index].verb in {"hat","beträgt"}
                    for index in self.facts_by_subject[subject])}
        if not subjects and REFERENCES.search(prompt):
            subjects = {subject for subject in (context or {}).get("subjects",[]) if subject in self.facts_by_subject}
        inverse = bool(re.search(r"\b(?:welch\w*|wer|was)\b",prompt,re.I)) and bool(values) and not subjects
        candidates = []
        ids = range(len(self.facts)) if inverse else sorted({i for subject in subjects for i in self.facts_by_subject[subject]})
        for index in ids:
            fact = self.facts[index]
            entity = fact.owner or fact.subject
            type_terms = set()
            for definition_id in self.facts_by_subject.get(entity,()):
                definition = self.facts[definition_id]
                if definition.subject==entity and definition.verb in {"ist","sind"}:
                    type_terms.update(definition.attributes)
            requested = words-QUERY_GRAMMAR-set(entity.split())-type_terms-values
            if inverse:
                requested -= set(fact.subject.split())
            if fact.verb not in {"hat","beträgt"}:
                continue
            object_words = set(self._role_words(fact.object_role))
            if values and not values <= object_words:
                continue
            if requested and not requested <= fact.attributes:
                continue
            if not requested and not inverse:
                continue
            candidates.append(index)
        if len(candidates)!=1:
            return None
        index = candidates[0]
        fact = self.facts[index]
        entity = fact.owner or fact.subject
        state = {"subjects":[entity],"facets":["attribute"],
                 "user_facts":dict((context or {}).get("user_facts",{}))}
        return {"supported":True,"subjects":[entity],"facets":["attribute"],
                "groups_by_facet":{},"reason":"","kind":"role_grammar",
                "segments":tuple(self._fact_segments(index)),"evidence":(fact.subject_role,fact.object_role),
                "speech_act":"answer_declarative_attribute","state":state,"used_context":bool(context)}

    def analyze_question(self, prompt, context=None):
        if not isinstance(prompt,str):
            raise ValueError("prompt must be a string")
        if not prompt.strip():
            return {"supported":False,"subjects":[],"facets":[],"groups_by_facet":{},"reason":"empty_prompt"}
        if self.record_count==0:
            return {"supported":False,"subjects":[],"facets":[],"groups_by_facet":{},"reason":"empty_information_corpus"}
        if analysis := self._conversation_analysis(prompt,context):
            return analysis
        # Questions containing explicit negation are not silently transformed to
        # positive assertions; a truth-conditional logic engine would be needed.
        if re.search(r"\b(?:nicht|kein|keine|keinen|keinem|keiner|keines)\b",prompt,re.I):
            return {"supported":False,"subjects":[],"facets":[],"groups_by_facet":{},"reason":"unsupported_negation"}
        if analysis := self._attribute_analysis(prompt,context):
            return analysis
        analysis = super().analyze_question(prompt,context)
        # Treat general question syntax as grammar rather than unknown entities.
        unknown = [term for term in analysis["unbound_terms"] if term not in QUERY_GRAMMAR]
        analysis["unbound_terms"] = unknown
        analysis["supported"] = bool(analysis["subjects"]) and not unknown and all(analysis["groups_by_facet"].get(facet) for facet in analysis["facets"])
        analysis["reason"] = "" if analysis["supported"] else "no_supported_information_binding"
        analysis["kind"] = "declarative_transition"
        analysis["state"] = {"subjects":analysis["subjects"],"facets":analysis["facets"],
                             "user_facts":dict((context or {}).get("user_facts",{}))}
        return analysis

    def prompt_field(self, prompt, context=None, context_field=None, *, facet=None):
        analysis = self.analyze_question(prompt,context)
        if analysis.get("kind")!="role_grammar":
            return super().prompt_field(prompt,context,context_field,facet=facet)
        amplitudes = np.zeros(self.size)
        for index in self._encode_words(TOKEN_RE.findall(prompt)):
            amplitudes[index] += 1
        amplitudes = _unit(amplitudes)
        if context_field is not None:
            prior = np.asarray(context_field,dtype=float)
            if prior.shape!=(self.size,) or not np.isfinite(prior).all() or (prior<0).any():
                raise ValueError("context_field needs one finite nonnegative amplitude per token")
            amplitudes = _unit(amplitudes+_unit(prior))
        evidence = analysis["evidence"]
        modes = np.ones(len(evidence))/math.sqrt(len(evidence)) if evidence else np.array([])
        spectrum = np.fft.fft(modes,norm="ortho") if evidence else np.array([],dtype=complex)
        return UnpairedPrompt(amplitudes,np.array(evidence,dtype=np.intp),spectrum,
                              analysis["segments"],evidence,1.0,analysis.get("temporary_roles",{})),analysis

    def _grammar_modes(self, prefix, field):
        # NFA state = grammatical segment, source-role alternative, role position.
        # Epsilon transitions only advance between grammatical constituents.
        segments = field.segments
        def enter(segment):
            if segment>=len(segments):
                return {(segment,-1,0)}
            return {(segment,role,0) for role in segments[segment] if len(self._field_role(role,field).indices)}
        states = enter(0)
        for token in prefix:
            token_id = self.index.get(token,-1)
            following = set()
            for segment, role_id, position in states:
                if segment>=len(segments):
                    continue
                role = self._field_role(role_id,field)
                if int(role.indices[position])!=token_id:
                    continue
                if position+1<len(role.indices):
                    following.add((segment,role_id,position+1))
                else:
                    following.update(enter(segment+1))
            states = following
            if not states:
                raise ValueError("Prefix is outside the supported role grammar")
        values = defaultdict(complex)
        for segment, role_id, position in states:
            if segment>=len(segments):
                values[self.index[EOS]] += 1
            else:
                role = self._field_role(role_id,field)
                values[int(role.indices[position])] += role.amplitudes()[position]
        return values,states

    def next_distribution(self, prefix_tokens, prompt_field, *, method="operator",time_s=0.0,
                          phase_error=0.0,prefix_enabled=True,prompt_gain=1.0,data_gain=None,
                          order_policy="deepest"):
        """Role-grammar step: declaration roles gate a grammar, then the two waves interfere.

        The compatible lexical modes of the current grammatical state form the
        data oscillation; the prompt wave and the all-data wave modulate it in the
        same readout as the declarative path, so both utterance kinds are answers
        to the interaction of the same two fields.
        """
        if not isinstance(prompt_field,UnpairedPrompt):
            return super().next_distribution(prefix_tokens,prompt_field,method=method,time_s=time_s,
                phase_error=phase_error,prefix_enabled=prefix_enabled,prompt_gain=prompt_gain,
                data_gain=data_gain,order_policy=order_policy)
        if method not in {"operator","direct"}:
            raise ValueError("method must be operator or direct")
        if data_gain is None:
            data_gain = self.data_gain
        if not np.isfinite([time_s,phase_error,prompt_gain,data_gain]).all() or prompt_gain<0 or data_gain<0:
            raise ValueError("Invalid interference controls")
        field = prompt_field
        if field.token_amplitudes.shape!=(self.size,) or not np.isfinite(field.token_amplitudes).all():
            raise ValueError("Prompt field does not match the model")
        feature_modes = np.fft.ifft(field.feature_spectrum,norm="ortho") if len(field.feature_spectrum) else np.array([])
        # Evidence coefficients really gate the grammar, including function words.
        gate = sum(abs(mode)**2 * float(np.vdot(self._field_role(int(role),field).spectrum,
                   self._field_role(int(role),field).spectrum).real)
                   for role,mode in zip(field.feature_indices,feature_modes)) * abs(field.numerical_gate)**2
        if gate<=1e-24:
            raise ValueError("No nonzero declaration wave licenses this grammar")
        values,states = self._grammar_modes(prefix_tokens if prefix_enabled else [],field)
        active = np.array(sorted(values),dtype=np.intp)
        modes = np.array([values[int(index)] for index in active],dtype=complex)
        energy = float(np.vdot(modes,modes).real)
        if energy<=1e-24:
            raise ValueError("Role data wave has no supported finite energy")
        modes *= math.sqrt(gate/energy)
        prompt_wave = self.prompt_wave(field)
        data_wave_field = self.data_wave()
        carrier_size = next_fast_len(len(active))
        operator_readout, direct_readout, difference = interference_pair(
            active, modes, prompt_wave, coupling=prompt_gain, phase_error=phase_error,
            mode_prior=self._mode_prior(data_wave_field, active, data_gain),
            carrier_size=carrier_size)
        if difference > VERIFY_TOLERANCE:
            raise ValueError("Spectral operator and direct readout disagree")
        readout = operator_readout if method=="operator" else direct_readout
        local_probabilities, total_intensity = probabilities(readout)
        probabilities_out = np.zeros(self.size)
        probabilities_out[active] = local_probabilities
        return probabilities_out,{"runtime_source":"complex_declaration_roles_and_explicit_grammar",
                "method":method,"prefix":list(prefix_tokens[-self.order:]),"active_token_indices":active.tolist(),
                "carrier_size":carrier_size,"known_vocabulary_size":self.size,"all_supported_modes_included":True,
                "support_rule":"all_compatible_lexical_modes_of_current_grammatical_role",
                "grammar_states":[list(state) for state in sorted(states)],"field_addresses":list(field.evidence),
                "feature_field_energy":gate,"mixed_data_energy_before_normalization":energy,
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
                "coefficient_rule":"direct_unit_energy_declaration_roles_no_fitted_gains"}


    def generate(self,prompt,*,context=None,context_field=None,time_s=0.0,max_tokens=64,
                 decoding="beam",beam_width=4,seed=17,method="operator",phase_error=0.0,
                 prefix_enabled=True,prompt_gain=1.0,data_gain=None,order_policy="deepest"):
        if type(max_tokens) is not int or not 1<=max_tokens<=256:
            raise ValueError("max_tokens must be an integer from 1 to 256")
        if type(beam_width) is not int or not 1<=beam_width<=16:
            raise ValueError("beam_width must be an integer from 1 to 16")
        if decoding not in {"beam","greedy","sample"}:
            raise ValueError("decoding must be beam, greedy or sample")
        if type(seed) is not int or not 0<=seed<2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        analysis = self.analyze_question(prompt,context)
        if analysis.get("kind")!="role_grammar":
            try:
                result = super().generate(prompt,context=context,context_field=context_field,time_s=time_s,
                    max_tokens=max_tokens,decoding=decoding,beam_width=beam_width,seed=seed,method=method,
                    phase_error=phase_error,prefix_enabled=prefix_enabled,prompt_gain=prompt_gain,
                    data_gain=data_gain,order_policy=order_policy)
            except ValueError as exc:
                if not any(term in str(exc) for term in ("wave","energy","binding")):
                    raise
                result = {"text":"","tokens":[],"trace":[],"ended":True,"reason":"no_supported_wave_energy","analysis":analysis}
            result["information_state"] = analysis.get("state",{"subjects":[],"facets":[],"user_facts":dict((context or {}).get("user_facts",{}))})
            result["state"] = result["information_state"]
            result["uses_answer_candidates"] = False
            return self._annotate(result)
        field,_ = self.prompt_field(prompt,context,context_field)
        width = beam_width if decoding=="beam" else 1
        beams = [(0.0,[],[],False)]
        rng = np.random.default_rng(seed)
        reason = ""
        for step in range(max_tokens):
            candidates = []
            for score,prefix,history,ended in beams:
                if ended:
                    candidates.append((score,prefix,history,True))
                    continue
                try:
                    distribution,trace = self.next_distribution(prefix,field,method=method,time_s=time_s,
                        phase_error=phase_error,prefix_enabled=prefix_enabled,prompt_gain=prompt_gain,
                        data_gain=data_gain)
                except ValueError as exc:
                    if not any(term in str(exc) for term in ("wave","energy","Prefix")):
                        raise
                    reason = "no_supported_wave_energy"
                    continue
                active = np.flatnonzero(distribution)
                choices = [int(rng.choice(self.size,p=distribution))] if decoding=="sample" else active[np.argsort(-distribution[active],kind="stable")[:width]].tolist()
                for index in choices:
                    token = self.vocabulary[index]
                    trace_value = {**trace,"step":step,"facet":analysis["facets"][0],"selected_token":token,
                        "selected_bin":index,"selected_frequency_hz":float(self.frequencies[index]),
                        "probability":float(distribution[index])}
                    candidates.append((score+math.log(distribution[index]),prefix+([] if token==EOS else [token]),
                                       history+[trace_value],token==EOS))
            if not candidates:
                return self._annotate({"text":"","tokens":[],"trace":[],"ended":True,"reason":reason or "no_supported_grammar",
                        "analysis":self._public_analysis(analysis),"information_state":analysis["state"],"state":analysis["state"],"uses_answer_candidates":False})
            candidates.sort(key=lambda item:(-round(item[0],12),tuple(item[1])))
            beams = candidates[:width]
            if all(beam[3] for beam in beams):
                break
        _,tokens,traces,ended = ([beam for beam in beams if beam[3]] or beams)[0]
        return self._annotate({"text":self._render_roles(tokens,traces,field),"tokens":tokens,"trace":traces,"ended":ended,
                "reason":"" if ended else "token_budget_or_cycle_limit","analysis":self._public_analysis(analysis),
                "information_state":analysis["state"],"state":analysis["state"],"uses_answer_candidates":False,
                "coefficient_adaptation":"direct declarative role counts and explicit grammar; no optimizer",
                "decoding":decoding,"corpus_digest":self.corpus_digest})

    def _annotate(self,result):
        _, truncated_bytes = self._render_byte_tokens(result["tokens"])
        if result["ended"] and truncated_bytes:
            raise ValueError("A completed role emitted an incomplete UTF-8 sequence")
        result["tokenization"] = {"budget_unit":"spectral_symbols_including_punctuation_utf8_bytes_and_eos",
            "decoder_steps":len(result["trace"]),"emitted_symbols":len(result["tokens"]),
            "utf8_byte_symbols":sum(token.startswith("<byte:") for token in result["tokens"]),
            "utf8_truncated_bytes":truncated_bytes,"utf8_complete":truncated_bytes==0,
            "rendered_lexical_tokens":len(tokenize(result["text"])),
            "unknown_prompt_words":"lossless_fixed_utf8_byte_modes_scoped_to_session"}
        return result

    @staticmethod
    def _public_analysis(analysis):
        return {key:value for key,value in analysis.items() if key not in {"segments","evidence","temporary_roles"}}

    @staticmethod
    def _render_byte_tokens(tokens):
        decoded, pending = [],bytearray()
        omitted = 0
        def flush(final=False):
            nonlocal omitted
            if pending:
                try:
                    word = pending.decode("utf-8",errors="strict")
                except UnicodeDecodeError as exc:
                    if final and exc.reason=="unexpected end of data" and exc.end==len(pending):
                        word = pending[:exc.start].decode("utf-8",errors="strict")
                        omitted += len(pending)-exc.start
                    else:
                        raise ValueError("Invalid UTF-8 byte sequence in a decoded lexical role") from exc
                if word:
                    decoded.append(word)
                pending.clear()
        for token in tokens:
            if token.startswith("<byte:") and token.endswith(">"):
                pending.append(int(token[6:-1],16))
            else:
                flush()
                decoded.append(token)
        flush(final=True)
        return decoded,omitted

    def _render_roles(self,tokens,traces,field):
        # Renderer positions refer to decoded lexical symbols, while the trace
        # can contain several UTF-8 byte symbols for one such lexical symbol.
        position_map = {}
        compact_position, pending = 0,False
        for position,token in enumerate(tokens):
            if token.startswith("<byte:"):
                pending = True
                continue
            if pending:
                compact_position += 1
                pending = False
            position_map[position] = compact_position
            compact_position += 1
        grammatical, surfaces = set(),{}
        emitted = 0
        for trace in traces:
            if trace["selected_token"]==EOS:
                continue
            if emitted in position_map:
                position = position_map[emitted]
                for segment,role_id,role_position in trace.get("grammar_states",[]):
                    if segment>=len(field.segments):
                        continue
                    role = self._field_role(role_id,field)
                    if role.origin=="explicit_function_grammar":
                        grammatical.add(position)
                    elif role.origin=="session_prompt_only" and role.surfaces and role_position in role.surfaces:
                        surfaces[position] = role.surfaces[role_position]
            emitted += 1
        return self.render(tokens,grammar_positions=grammatical,prompt_spellings=surfaces)

    def render(self,tokens,*,grammar_positions=None,prompt_spellings=None):
        decoded,_ = self._render_byte_tokens(tokens)
        if grammar_positions is None:
            return super().render(decoded)
        spelling = {}
        for position in range(len(decoded)):
            if position and decoded[position-1] not in {".","!","?"}:
                continue
            for phrase,surfaces in self.lexicon_spellings.items():
                if len(phrase)>1 and tuple(decoded[position:position+len(phrase)])==phrase:
                    spelling.update({position+offset:surface for offset,surface in enumerate(surfaces)})
        result = ""
        sentence_start = True
        previous = [BOS,BOS]
        modal_clause = False
        for position,token in enumerate(decoded):
            if token in {BOS,EOS}:
                sentence_start = True
                previous = [BOS,BOS]
                modal_clause = False
                continue
            if token in {"möchtest","kannst","kann","möchte"}:
                modal_clause = True
            functional = position in grammar_positions or token in STOPWORDS
            infinitive = modal_clause and canonical_term(token) in self.action_roles
            word = token if functional or infinitive else self.context_forms.get(
                (tuple(previous[-2:]),token),self.display_tokens.get(token,token))
            word = spelling.get(position,word)
            word = (prompt_spellings or {}).get(position,word)
            if sentence_start and word.isalpha() and len(word)>1:
                word = word[:1].upper()+word[1:]
            if not result or token in ".,!?;:%)]}":
                result += word
            elif result[-1:] in "([{":
                result += word
            else:
                result += " "+word
            abbreviation_dot = token=="." and (previous[-1] in ABBREVIATIONS or previous[-1].isdigit())
            sentence_start = token in ".!?" and not abbreviation_dot
            if sentence_start:
                modal_clause = False
            previous.append(token)
        return result

    def storage_stats(self):
        stats = super().storage_stats()
        role_bytes = sum(role.indices.nbytes+role.spectrum.nbytes for role in getattr(self,"lexical_roles",[])
                         if role.origin!="explicit_function_grammar")
        stats.update({"declarative_facts":len(getattr(self,"facts",[])),"lexical_role_bytes":role_bytes,
                      "numerical_payload_bytes":stats["numerical_payload_bytes"]+role_bytes,
                      "explicit_grammar":True,"optimizer_steps":0,"requires_question_answer_pairs":False})
        return stats

    def model_signature(self):
        base = super().model_signature()
        digest = hashlib.sha256()
        for role in self.lexical_roles:
            if role.origin in {"explicit_function_grammar","session_prompt_only"}:
                continue
            digest.update(role.origin.encode())
            digest.update(role.indices.tobytes())
            digest.update(role.spectrum.tobytes())
        base["declaration_role_sha256"] = digest.hexdigest()
        base["grammar_version"] = "unpaired_roles_v2"
        return base
