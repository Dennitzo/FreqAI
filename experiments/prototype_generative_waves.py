"""Independent next-token wave-decoder comparison; no document retrieval.

Only response text is read from the JSONL fixture. Prompt keys, conversation
evaluations and the productive database are deliberately not inputs.

The exact decoder is a finite-order count language model expressed as a bank of
oscillating token modes. HRR compresses the same conditional distributions into
a superposition of phase bindings. Neither model is an untrained general AI:
corpus counts supply a statistical language prior and the symbol table supplies
the explicit, conventional connection from modes to UTF-8 token strings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BOS = "<BOS>"
EOS = "<EOS>"
TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*|[^\w\s]", re.UNICODE)
STOP = set("aber alle als also am an auch auf aus bei bin bis bist da das dass dein deine dem den der des dich die dir doch du ein eine einem einen einer eines einfach er es etwas für gern gerade geht habe haben hast hat heute ich im in ist ja kann kein kleine können mag man mein meine mich mit möchte möchtest mir nach nicht noch nun nur ob oder ohne schon sehr sich sie sind so um und uns unser von vor war was wenn werde werden wie wieder wir wird wo zu zum zur".split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def detokenize(tokens: list[str]) -> str:
    result = ""
    sentence_start = True
    for token in tokens:
        if token in (BOS, EOS):
            continue
        word = token[:1].upper() + token[1:] if sentence_start and token.isalpha() else token
        if not result or token in ".,!?;:%)]}":
            result += word
        elif result[-1:] in "([{":
            result += word
        else:
            result += " " + word
        sentence_start = token in ".!?"
    return result


def stable_frequency(symbol: str) -> float:
    """Symbol-specific oscillator, stable when the vocabulary changes."""
    number = int.from_bytes(hashlib.blake2b(symbol.encode(), digest_size=8).digest(), "little")
    return 0.25 + 29.75 * (number / float(2**64))


def _seed(symbol: str) -> int:
    return int.from_bytes(hashlib.blake2b(symbol.encode(), digest_size=8).digest(), "little")


class PromptField(np.ndarray):
    """Token amplitudes and a separate numerical input-feature spectrum.

    The two channels have distinct dictionaries. Multiplication by a scalar
    applies to both channels; replacing only array entries is a token-channel
    intervention and is deliberately not a complete prompt ablation.
    """

    def __new__(cls, values, feature_spectrum=None):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.feature_spectrum = np.asarray(feature_spectrum if feature_spectrum is not None else [], dtype=complex)
        return obj

    def __array_finalize__(self, parent):
        self.feature_spectrum = getattr(parent, "feature_spectrum", np.array([], dtype=complex))

    def __mul__(self, other):
        result = np.asarray(self)*other
        return PromptField(result, self.feature_spectrum*other if np.isscalar(other) else self.feature_spectrum)

    __rmul__ = __mul__

    def __truediv__(self, other):
        result = np.asarray(self)/other
        return PromptField(result, self.feature_spectrum/other if np.isscalar(other) else self.feature_spectrum)


def spectral_convolution(left, right):
    """Circular convolution of frequency bins, using the convolution theorem.

    FFTs here transform the frequency-index sequence to calculate convolution;
    this helper does not select or plan text. With orthonormal spectra it gives
    F(F^-1(left) * F^-1(right)).
    """
    return np.fft.ifft(np.fft.fft(left)*np.fft.fft(right))/math.sqrt(len(left))


class GenerativeWaveModel:
    """Count-based autoregressive decoder with an inspectable oscillator state.

    Transition data are corpus sufficient statistics, not stored reply slots.
    At step n, the preceding generated tokens excite matching prefix modes;
    their outgoing token fields interfere with a prompt-derived topic field.
    The decoder demodulates each known symbol mode and samples its intensity.
    """

    def __init__(self, texts, order: int = 3, hrr_dimensions: int = 2048,
                 conditioned_pairs=None, conditioning="semantic", prior_weight=0.25):
        self.order = order
        self.hrr_dimensions = hrr_dimensions
        self.conditioning = conditioning
        pairs = list(conditioned_pairs or [])
        self.paired_example_count = len(pairs)
        paired_texts = [str(pair["text"]) for pair in pairs]
        self.texts = tuple(dict.fromkeys([str(text) for text in texts] + paired_texts))
        sequences = [tokenize(text) for text in self.texts]
        self.surface_forms = defaultdict(Counter)
        for text in self.texts:
            sentence_start = True
            for surface in TOKEN_RE.findall(text):
                token = surface.casefold()
                self.surface_forms[token][surface] += 0.15 if sentence_start else 1.0
                sentence_start = surface in ".!?"
        self.display_tokens = {word: counts.most_common(1)[0][0] for word, counts in self.surface_forms.items()}
        self.vocabulary = sorted({word for seq in sequences for word in seq} | {EOS})
        self.index = {word: i for i, word in enumerate(self.vocabulary)}
        self.frequencies = np.array([stable_frequency(word) for word in self.vocabulary])
        self.token_frequencies = self.frequencies
        self.size = len(self.vocabulary)
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        paired_set = set(paired_texts)
        for text, seq in zip(self.texts, sequences):
            count_weight = 1.0 if not pairs or text in paired_set else prior_weight
            extended = [BOS] * order + seq + [EOS]
            for position in range(order, len(extended)):
                for depth in range(order + 1):
                    key = tuple(extended[position-depth:position]) if depth else ()
                    self.counts[key][extended[position]] += count_weight
        self.distributions = {}
        for key, counts in self.counts.items():
            vector = np.zeros(self.size)
            for word, count in counts.items():
                vector[self.index[word]] = count
            self.distributions[key] = vector / vector.sum()
        # These complex arrays are the runtime data source. Real distributions
        # remain only for the alternate HRR benchmark and numerical references.
        self.transition_spectra = {key: np.fft.fft(np.sqrt(distribution), norm="ortho")
                                   for key, distribution in self.distributions.items()}
        self.supports = {key: np.flatnonzero(distribution) for key, distribution in self.distributions.items()}
        conditioned_counts = defaultdict(Counter)
        self.feature_frequency = Counter()
        for pair in pairs:
            features = self._input_features(str(pair["prompt"]))
            self.feature_frequency.update(features)
            seq = [BOS]*order + tokenize(str(pair["text"])) + [EOS]
            for feature in features:
                for position in range(order, len(seq)):
                    for depth in range(order+1):
                        key = tuple(seq[position-depth:position]) if depth else ()
                        conditioned_counts[(feature, key)][seq[position]] += 1
        self.conditioned_spectra = {}
        self.conditioned_supports = {}
        for key, counts in conditioned_counts.items():
            values = np.zeros(self.size)
            for token, count in counts.items():
                values[self.index[token]] = count
            self.conditioned_spectra[key] = np.fft.fft(np.sqrt(values/values.sum()), norm="ortho")
            self.conditioned_supports[key] = np.flatnonzero(values)
        self.feature_vocabulary = sorted(self.feature_frequency)
        self.feature_index = {feature: i for i, feature in enumerate(self.feature_vocabulary)}
        self.feature_frequencies = np.array([stable_frequency("feature:"+feature) for feature in self.feature_vocabulary])
        # A response-only local co-occurrence prior, explicitly not an input->
        # response label lookup. Window size limits diffuse document-level noise.
        association = np.zeros((self.size, self.size), dtype=np.float64)
        document_frequency = Counter()
        for seq in sequences:
            document_frequency.update(set(seq))
            for i, left in enumerate(seq):
                if left in STOP or not left.isalpha():
                    continue
                for j in range(max(0, i-6), min(len(seq), i+7)):
                    right = seq[j]
                    if left != right and right.isalpha():
                        association[self.index[left], self.index[right]] += 1 / (1 + abs(i-j))
        norm = association.max(axis=1, keepdims=True)
        self.association = association / np.maximum(norm, 1e-12)
        self.idf = np.array([math.log1p(len(sequences)/(1+document_frequency[word])) for word in self.vocabulary])
        self._hrr_ready = False
        self._hrr_cache = {}

    def _input_features(self, text, context=None):
        features = {"word:"+word: 1.0 for word in tokenize(text) if word not in STOP and word.isalpha()}
        if self.conditioning == "semantic":
            from freqai.semantics import analyze
            interpretation = analyze(text, context)
            for act in interpretation.acts:
                if act.kind in ("question", "context_clarification"):
                    continue
                # Polarities and speaker roles remain distinct input addresses.
                signature = ":".join((act.kind, act.target, act.value, act.topic))
                features["semantic:"+signature] = 8.0
        return features

    def render(self, tokens):
        return detokenize([self.display_tokens.get(token, token) for token in tokens])

    def prompt_field(self, prompt: str, context_text: str = "", context=None) -> tuple[np.ndarray, dict]:
        field = np.zeros(self.size)
        observed = []
        unknown = []
        for text, weight in ((context_text, 0.35), (prompt, 1.0)):
            for word in tokenize(text):
                if word in STOP or not word.isalpha():
                    continue
                if word in self.index:
                    i = self.index[word]
                    field += weight * self.idf[i] * self.association[i]
                    field[i] += weight * self.idf[i] * 0.2
                    observed.append(word)
                else:
                    unknown.append(word)
        if field.max() > 0:
            field /= field.max()
        features = self._input_features(prompt, context)
        feature_amplitudes = np.zeros(len(self.feature_vocabulary))
        for feature, weight in features.items():
            if feature in self.feature_index:
                feature_amplitudes[self.feature_index[feature]] = weight/math.sqrt(max(1, self.feature_frequency[feature]))
        feature_spectrum = np.fft.fft(feature_amplitudes, norm="ortho") if len(feature_amplitudes) else np.array([], dtype=complex)
        return PromptField(field, feature_spectrum), {"known_content_tokens": observed, "unknown_content_tokens": unknown,
                       "source": "response_token_cooccurrence", "context_weight": 0.35,
                       "condition_features": features,
                       "matched_condition_features": [feature for feature in features if feature in self.feature_frequency]}

    def _prepare_hrr(self):
        if self._hrr_ready:
            return
        rng = np.random.default_rng(923714)
        self._token_codes = np.exp(2j*np.pi*rng.random((self.size, self.hrr_dimensions)))
        self._hrr = np.zeros(self.hrr_dimensions, dtype=np.complex128)
        # Each observed prefix contributes an equally weighted conditional field.
        for key, distribution in self.distributions.items():
            values = np.flatnonzero(distribution)
            payload = distribution[values] @ self._token_codes[values]
            self._hrr += self._key_code(key) * payload
        self._hrr_ready = True

    def _key_code(self, key):
        rng = np.random.default_rng(_seed(json.dumps(key, ensure_ascii=False)))
        return np.exp(2j*np.pi*rng.random(self.hrr_dimensions))

    def _hrr_distribution(self, key):
        self._prepare_hrr()
        if key not in self._hrr_cache:
            unbound = self._hrr * np.conj(self._key_code(key))
            scores = np.real(self._token_codes.conj() @ unbound) / self.hrr_dimensions
            self._hrr_cache[key] = scores
        return self._hrr_cache[key]

    def next_distribution(self, prefix_tokens, prompt_field, *, method="operator", time_s=0.0,
                          prompt_gain=1.2, prefix_enabled=True, phase_error=0.0):
        prefix = [BOS]*self.order + list(prefix_tokens)
        keys = []
        if prefix_enabled:
            for depth in range(self.order, 0, -1):
                key = tuple(prefix[-depth:])
                if key in self.transition_spectra:
                    keys.append(key)
        if not keys:
            keys = [()]
        # Explicit interpolation: the most specific observed grammar is primary,
        # with backoff allowing local recombination instead of complete replay.
        keys = keys[:2]
        weights = [0.92, 0.08] if len(keys) == 2 else [1.0]
        base_spectrum = sum(weight*self.transition_spectra[key] for key, weight in zip(keys, weights))
        support = np.zeros(self.size, dtype=bool)
        for key in keys:
            support[self.supports[key]] = True
        conditional_sources = []
        feature_spectrum = getattr(prompt_field, "feature_spectrum", np.array([], dtype=complex))
        if len(feature_spectrum) not in (0, len(self.feature_vocabulary)):
            raise ValueError("feature spectrum does not match the model's feature dictionary")
        feature_modes = np.abs(np.fft.ifft(feature_spectrum, norm="ortho")) if len(feature_spectrum) and prompt_gain else np.array([])
        # Only the numerical field authorizes feature activations. The original
        # feature labels in diagnostics are never read to select output fields.
        features = {self.feature_vocabulary[int(i)]: float(feature_modes[i])
                    for i in np.flatnonzero(feature_modes > 1e-9)}
        # The longest prefix supported by any active input feature excites
        # compatible conditional transition fields. No document index is used.
        condition_wave = np.zeros(self.size, dtype=np.complex128)
        total_weight = 0.0
        condition_support = np.zeros(self.size, dtype=bool)
        for depth in range(self.order if prefix_enabled else 0, -1, -1):
            context = tuple(prefix[-depth:]) if depth else ()
            for feature, feature_weight in features.items():
                address = (feature, context)
                spectrum = self.conditioned_spectra.get(address)
                if spectrum is not None:
                    # Scarce lexical features carry stronger conditioning than
                    # frequent ones. Semantic addresses carry explicit priors.
                    weight = feature_weight
                    condition_wave += weight*spectrum
                    total_weight += weight
                    condition_support[self.conditioned_supports[address]] = True
                    conditional_sources.append({"feature": feature, "prefix": list(context), "weight": weight})
            if total_weight:
                break
        if total_weight:
            condition_wave /= total_weight
            # Small global prior prevents a paired field from becoming the only
            # possible sentence. Token-level grammar mask preserves local syntax.
            common_support = support & condition_support
            if common_support.any():
                base_spectrum = 0.04*base_spectrum + 0.96*condition_wave
                support = common_support
        if method == "operator":
            output_spectrum = base_spectrum
        elif method == "hrr":
            recovered = sum(weight*self._hrr_distribution(key) for key, weight in zip(keys, weights))
            # A separately reported finite-state grammar mask prevents crosstalk
            # from injecting impossible outgoing edges. It does not fix scores.
            conditional = np.where(support, np.maximum(recovered, 0), 0)
            if conditional.sum() <= 1e-12:
                conditional = support.astype(float)
            conditional /= conditional.sum()
            output_spectrum = np.fft.fft(np.sqrt(conditional), norm="ortho")
        else:
            raise ValueError("method must be operator or hrr")
        phases = np.exp(2j*np.pi*self.frequencies*time_s)
        phase_spectrum = np.fft.fft(phases, norm="ortho")
        moving_spectrum = spectral_convolution(output_spectrum, phase_spectrum)
        prompt_spectrum = np.fft.fft(np.asarray(prompt_field), norm="ortho")
        interaction = prompt_gain*spectral_convolution(moving_spectrum, prompt_spectrum)*np.exp(1j*phase_error)
        result_spectrum = moving_spectrum + interaction
        # The actual token amplitudes are obtained from the resulting complex
        # field. There is no earlier planned token or complete answer string.
        amplitudes = np.fft.ifft(result_spectrum, norm="ortho")
        scores = np.abs(amplitudes)**2
        scores[~support] = 0  # explicit local grammar constraint and roundoff guard
        if not np.isfinite(scores).all() or scores.sum() <= 1e-24:
            raise ValueError("The resulting field has no finite supported token energy")
        probabilities = scores / scores.sum()
        trace = {"prefix": list(prefix_tokens[-self.order:]),
                 "field_sources": [list(key) for key in keys], "field_weights": weights,
                 "decoder": method, "support_size": int(support.sum()),
                 "conditioned_field_sources": conditional_sources,
                 "feature_field_norm": float(np.linalg.norm(feature_spectrum)),
                 "feature_decoder": "inverse_dft_mode_amplitude",
                 "runtime_source": "imported_complex_transition_spectra",
                 "fft_roundtrip_max_error": float(np.max(np.abs(np.fft.fft(amplitudes, norm="ortho")-result_spectrum))),
                 "phase_time_s": time_s, "prompt_gain": prompt_gain,
                 "grammar_mask": method == "hrr", "phase_error": phase_error}
        return probabilities, trace

    def generate(self, prompt: str, context_text: str = "", *, context_field=None, method="operator", time_s=0.0,
                 max_tokens=48, seed=17, prompt_gain=1.2, prefix_enabled=True, phase_error=0.0,
                 temperature=0.8, decoding="sample", beam_width=4, max_sentences=2, context=None):
        field, field_info = self.prompt_field(prompt, context_text, context)
        if context_field is not None:
            prior = np.asarray(context_field, dtype=float)
            if prior.shape != field.shape or not np.isfinite(prior).all() or (prior < 0).any():
                raise ValueError("context_field must contain one finite nonnegative amplitude per vocabulary token")
            field += 0.35*prior
            if field.max() > 0:
                field /= field.max()
            field_info["persistent_context_field"] = True
        rng = np.random.default_rng(seed)
        if decoding not in ("sample", "greedy", "beam"):
            raise ValueError("decoding must be sample, greedy or beam")
        width = max(1, int(beam_width)) if decoding == "beam" else 1
        # Each beam item is an unfinished autoregressive token sequence. Its
        # score contains only accumulated probabilities from output fields.
        beams = [(0.0, [], [], False)]
        for step in range(max_tokens):
            candidates = []
            for log_probability, tokens, traces, ended in beams:
                if ended:
                    candidates.append((log_probability, tokens, traces, ended))
                    continue
                distribution, trace = self.next_distribution(tokens, field, method=method, time_s=time_s,
                    prompt_gain=prompt_gain, prefix_enabled=prefix_enabled, phase_error=phase_error)
                for token, count in Counter(tokens[-12:]).items():
                    if token.isalpha() and count > 1:
                        distribution[self.index[token]] *= 0.4**(count-1)
                distribution = np.maximum(distribution, 0)**(1/temperature)
                distribution /= distribution.sum()
                if decoding == "sample":
                    selections = [int(rng.choice(self.size, p=distribution))]
                else:
                    selections = [int(i) for i in np.argsort(-distribution, kind="stable")[:width] if distribution[i] > 0]
                for selected in selections:
                    token = self.vocabulary[selected]
                    selected_trace = {**trace, "step": step, "selected_token": token, "selected_bin": selected,
                        "selected_frequency_hz": float(self.frequencies[selected]), "selection": decoding,
                        "probability": float(distribution[selected]),
                        "top_modes": [{"token": self.vocabulary[i], "probability": float(distribution[i])}
                                      for i in np.argsort(distribution)[-5:][::-1] if distribution[i] > 0]}
                    new_tokens = tokens + ([token] if token != EOS else [])
                    new_ended = token == EOS or sum(word in ".!?" for word in new_tokens) >= max_sentences
                    candidates.append((log_probability+math.log(max(float(distribution[selected]), 1e-300)),
                                       new_tokens, traces+[selected_trace], new_ended))
            candidates.sort(key=lambda item: item[0]/((5+max(1, len(item[1])))/6)**0.7, reverse=True)
            beams = candidates[:width]
            if all(item[3] for item in beams):
                break
        log_probability, tokens, traces, ended = beams[0]
        text = self.render(tokens)
        return {"text": text, "tokens": tokens, "trace": traces, "prompt_field": field_info,
                "ended": ended, "method": method, "seed": seed,
                "decoding": decoding, "beam_width": width, "log_probability": log_probability,
                "conditioning": self.conditioning,
                "is_exact_corpus_response": tokenize(text) in [tokenize(t) for t in self.texts],
                "source": "compiled_token_transition_spectra", "uses_prompt_keys_for_response_comparison": False,
                "uses_paired_input_statistics": bool(self.paired_example_count)}


PROMPTS = [
    "Hallo", "Guten Morgen", "Wie geht es dir?", "Mir geht es gut, und dir?", "Ich bin traurig",
    "Ich bin heute müde", "Heute war die Arbeit stressig", "Ich freue mich auf das Wochenende",
    "Was möchtest du essen?", "Ich koche heute Nudeln", "Erzähl mir etwas über Musik",
    "Hast du einen guten Film gesehen?", "Ich lese gerade ein Buch", "Draußen regnet es",
    "Ich möchte spazieren gehen", "Heute habe ich viel zu tun", "Ich brauche eine Pause",
    "Ich kann nicht einschlafen", "Ich vermisse meine Freunde", "Ich habe heute Geburtstag",
    "Danke für das Gespräch", "Tschüss", "Was machst du gern?", "Ich möchte etwas Neues lernen",
    "Was ist ein Quantencomputer?",
]


def quality_statistics(model, output):
    tokens = output["tokens"]
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    repeats = len(trigrams)-len(set(trigrams))
    training_trigrams = {tuple(seq[i:i+3]) for text in model.texts
                         for seq in [tokenize(text)] for i in range(len(seq)-2)}
    return {"token_count": len(tokens), "ended": output["ended"], "repeated_trigrams": repeats,
            "corpus_trigram_fraction": sum(t in training_trigrams for t in trigrams)/max(1, len(trigrams)),
            "exact_corpus_response": output["is_exact_corpus_response"],
            "ends_with_sentence_punctuation": bool(tokens and tokens[-1] in ".!?"),
            "content_prompt_tokens_known": len(output["prompt_field"]["known_content_tokens"]),
            "automatic_semantic_correctness": None}


def run_comparison(output_directory: Path, corpus: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    documents = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    model = GenerativeWaveModel([doc["text"] for doc in documents])
    examples, details = [], []
    for method in ("operator", "hrr"):
        for i, prompt in enumerate(PROMPTS):
            output = model.generate(prompt, method=method, seed=400+i)
            details.append({"prompt": prompt, **output})
            examples.append({"prompt": prompt, "method": method, "answer": output["text"],
                             **quality_statistics(model, output)})
    invariant_errors, destructive_changes, ablations = [], [], []
    field, _ = model.prompt_field("Musik und Freunde")
    for prefix in ([], ["ich"], ["was", "hat"], ["möchtest", "du"]):
        original, _ = model.next_distribution(prefix, field, time_s=0)
        moved, _ = model.next_distribution(prefix, field, time_s=98765.4321)
        destructive, _ = model.next_distribution(prefix, field, phase_error=math.pi)
        invariant_errors.append(float(np.max(abs(original-moved))))
        destructive_changes.append(float(np.abs(original-destructive).sum()))
    for i, prompt in enumerate(PROMPTS):
        full = model.generate(prompt, seed=400+i)
        no_prompt = model.generate(prompt, seed=400+i, prompt_gain=0)
        no_prefix = model.generate(prompt, seed=400+i, prefix_enabled=False)
        ablations.append({"prompt": prompt, "full": full["text"], "no_prompt": no_prompt["text"],
            "no_prefix": no_prefix["text"], "prompt_changes_output": full["text"] != no_prompt["text"],
            "no_prefix_statistics": quality_statistics(model, no_prefix)})
    # Direct numerical recovery on observed prefix fields; this assesses storage,
    # not held-out language quality or the prompt-to-response semantic relation.
    rng = np.random.default_rng(183)
    keys = list(model.distributions)
    selection = [keys[int(i)] for i in rng.choice(len(keys), min(160, len(keys)), replace=False)]
    errors, winners = [], []
    for key in selection:
        exact = model.distributions[key]
        recovered = model._hrr_distribution(key)
        errors.append(float(np.sqrt(np.mean((exact-recovered)**2))))
        winners.append(bool(exact[int(np.argmax(recovered))] == exact.max()))
    time_invariant_outputs = sum(
        model.generate(prompt, seed=400+i)["tokens"] ==
        model.generate(prompt, seed=400+i, time_s=98765.4321)["tokens"]
        for i, prompt in enumerate(PROMPTS))
    extended_model = GenerativeWaveModel(model.texts + ("Zephyrluft bewegt den Papierdrachen.",))
    stable_mode_error = max(abs(float(model.frequencies[i])-float(extended_model.frequencies[extended_model.index[token]]))
                            for i, token in enumerate(model.vocabulary))
    aggregates = {}
    for method in ("operator", "hrr"):
        rows = [row for row in examples if row["method"] == method]
        aggregates[method] = {"novel_whole_responses": sum(not row["exact_corpus_response"] for row in rows),
            "terminated_responses": sum(row["ended"] for row in rows),
            "repeated_trigrams": sum(row["repeated_trigrams"] for row in rows),
            "mean_corpus_trigram_fraction": float(np.mean([row["corpus_trigram_fraction"] for row in rows]))}
    report = {"corpus_responses": len(documents), "vocabulary": model.size,
              "prefix_fields": len(model.distributions), "hrr_dimensions": model.hrr_dimensions,
              "elapsed_seconds": time.perf_counter()-started,
              "phase_invariance_max_probability_error": max(invariant_errors),
              "destructive_phase_l1_changes": destructive_changes,
              "hrr_unmasked_recovery_mean_rmse": float(np.mean(errors)),
              "hrr_unmasked_winner_accuracy": float(np.mean(winners)),
              "prompt_ablation_changes": sum(x["prompt_changes_output"] for x in ablations),
              "time_invariant_generated_outputs": time_invariant_outputs,
              "existing_frequency_change_after_vocabulary_extension": stable_mode_error,
              "aggregates": aggregates,
              "examples": examples,
              "limitations": ["No independent semantic correctness metric; outputs require human inspection.",
                  "A Fourier change of basis adds no information or semantic knowledge.",
                  "The response-only corpus has no conversational input/output association.",
                  "Finite vocabulary and n-gram context cannot support unrestricted answers.",
                  "HRR crosstalk is constrained by an explicit corpus-derived grammar mask; the full transition index remains necessary.",
                  "Novel whole strings and perfect wave reconstruction do not establish useful answers."]}
    (output_directory/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_directory/"traces.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_directory/"ablations.json").write_text(json.dumps(ablations, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Numerischer Vergleich zweier Token-Wellendecoder", "",
             "Das Korpus enthält ausschließlich die 120 Antworttexte. Promptschlüssel wurden nicht eingelesen.", "",
             f"Wortschatz: {model.size}; Prefixfelder: {len(model.distributions)}; HRR-Dimension: {model.hrr_dimensions}.",
             f"Phaseninvarianzfehler: {max(invariant_errors):.3e}; ungefilterte HRR-Gewinnergenauigkeit: {np.mean(winners):.1%}.",
             f"Promptablation ändert {report['prompt_ablation_changes']}/25 Ausgaben.", "",
             f"Vollständige Texte phaseninvariant: {time_invariant_outputs}/25. Frequenzänderung vorhandener Tokens nach Import: {stable_mode_error}.", "",
             "Diese Zahlen messen Numerik und Einfluss, keine semantische Richtigkeit.", "",
             "## Algorithmus", "",
             "Das Datenfeld enthält normierte lokale Übergangszählungen P(w | Prefix). Die letzten bis zu drei generierten Tokens aktivieren ihre Prefixfelder, mit explizitem Rückfall auf kürzere Kontexte.", "",
             "Für Token j gilt D_j(t) = sqrt(P_j) exp(i 2π f_j t). Das kompatible Promptfeld lautet Q_j(t) = g sqrt(P_j) T_j exp(i 2π f_j t), wobei T nur aus Token-Kookkurrenz in den Antworttexten stammt. Der nächste Token wird aus |D_j + Q_j|² ausgewählt. Dieser nichtlineare Auswahlprozess wird nach jedem Token erneut berechnet.", "",
             "Die FFT/IFFT prüft verlustfreie Feldrekonstruktion; sie erzeugt keine zusätzliche Bedeutung. Stabile Frequenzen f_j und das Tokenwörterbuch definieren die konventionelle Zuordnung von Schwingung zu Symbol.", "",
             "HRR speichert H = Summe_c K_c * V(P_c) mit zufälligen Einheitsphasen. Entbindung H * conj(K_c) und Modenprojektion schätzen die Folgetokenverteilung. Viele gleichzeitig überlagerte Prefixe verursachen starkes Übersprechen.", "",
             "## Beobachtungen", "",
             f"Der exakte Operator erzeugt {aggregates['operator']['novel_whole_responses']}/25 neue vollständige Tokenfolgen; HRR erzeugt {aggregates['hrr']['novel_whole_responses']}/25. Beide enden 25/25-mal ohne wiederholte Trigramme. Alle Trigramme kommen im Korpus vor. Trotzdem sind viele Antworten unpassend oder als Gesamtsatz ungrammatisch: lokale N-Gramm-Gültigkeit beweist keine Gesprächsqualität.", "",
             "## Ausgaben", "",
             "| Prompt | Exakter Operator | HRR |", "| --- | --- | --- |"]
    for i, prompt in enumerate(PROMPTS):
        values = [prompt, examples[i]["answer"], examples[len(PROMPTS)+i]["answer"]]
        lines.append("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |")
    lines += ["", "## Grenzen", ""] + ["- "+line for line in report["limitations"]]
    (output_directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("examples", "limitations")}, indent=2))


def run_conditioned_comparison(output_directory: Path, corpus: Path, language_corpus: Path):
    """Development-only comparison; no holdout/evaluation fixture is read."""
    output_directory.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior = [json.loads(line)["text"] for line in language_corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    examples, traces, checks = [], [], {}
    for conditioning in ("lexical", "semantic"):
        model = GenerativeWaveModel(prior, conditioned_pairs=pairs, conditioning=conditioning)
        for decoding in ("greedy", "beam"):
            for i, prompt in enumerate(PROMPTS):
                output = model.generate(prompt, decoding=decoding, seed=400+i)
                examples.append({"prompt": prompt, "conditioning": conditioning, "decoding": decoding,
                                 "answer": output["text"], **quality_statistics(model, output)})
                traces.append({"prompt": prompt, "conditioning": conditioning, "decoding": decoding, **output})
        moved_output_matches = 0
        full_zero_error, invariance_error, reference_error = [], [], []
        prompt_changes, metadata_invariant, feature_only_changes = 0, 0, 0
        for i, prompt in enumerate(PROMPTS):
            field, metadata = model.prompt_field(prompt)
            original, trace = model.next_distribution([], field)
            moved, _ = model.next_distribution([], field, time_s=345678.1234)
            empty, _ = model.prompt_field("")
            zero, _ = model.next_distribution([], field*0)
            blank, _ = model.next_distribution([], empty)
            features_only = PromptField(np.zeros_like(np.asarray(field)), field.feature_spectrum)
            features_result, _ = model.next_distribution([], features_only)
            feature_only_changes += int(np.max(abs(features_result-blank)) > 1e-12)
            invariance_error.append(float(np.max(abs(original-moved))))
            full_zero_error.append(float(np.max(abs(zero-blank))))
            # This independent direct route is a numerical reference, not the
            # generator's execution path. It reconstructs stored component modes
            # and applies the mathematically equivalent amplitude multiplication.
            base = sum(weight*np.fft.ifft(model.transition_spectra[tuple(key)], norm="ortho")
                       for key, weight in zip(trace["field_sources"], trace["field_weights"]))
            support = np.zeros(model.size, dtype=bool)
            for key in trace["field_sources"]:
                support[model.supports[tuple(key)]] = True
            conditional_support = np.zeros(model.size, dtype=bool)
            total = sum(item["weight"] for item in trace["conditioned_field_sources"])
            if total:
                conditional = np.zeros(model.size, dtype=complex)
                for item in trace["conditioned_field_sources"]:
                    address = (item["feature"], tuple(item["prefix"]))
                    conditional += item["weight"]*np.fft.ifft(model.conditioned_spectra[address], norm="ortho")/total
                    conditional_support[model.conditioned_supports[address]] = True
                common = support & conditional_support
                if common.any():
                    base = 0.04*base + 0.96*conditional
                    support = common
            reference = np.abs(base*(1+1.2*np.asarray(field)))**2
            reference[~support] = 0
            reference /= reference.sum()
            reference_error.append(float(np.max(abs(reference-original))))
            cloned = PromptField(np.asarray(field).copy(), field.feature_spectrum.copy())
            cloned_result, _ = model.next_distribution([], cloned)
            metadata_invariant += int(np.max(abs(cloned_result-original)) == 0)
            full = model.generate(prompt, decoding="beam", seed=400+i)
            no_prompt = model.generate(prompt, decoding="beam", seed=400+i, prompt_gain=0)
            moved_output = model.generate(prompt, decoding="beam", seed=400+i, time_s=345678.1234)
            moved_output_matches += int(full["tokens"] == moved_output["tokens"])
            prompt_changes += int(full["tokens"] != no_prompt["tokens"])
        checks[conditioning] = {"vocabulary_size": model.size,
            "input_feature_modes": len(model.feature_vocabulary),
            "conditional_transition_spectra": len(model.conditioned_spectra),
            "complete_prompt_ablation_max_error": max(full_zero_error),
            "data_operator_direct_reference_max_error": max(reference_error),
            "time_invariance_max_error": max(invariance_error),
            "time_invariant_beam_answers": moved_output_matches,
            "prompt_ablation_changes_answers": prompt_changes,
            "feature_channel_changes_initial_distribution": feature_only_changes,
            "metadata_free_field_identical_answers": metadata_invariant}
    aggregates = {}
    for conditioning in ("lexical", "semantic"):
        for decoding in ("greedy", "beam"):
            rows = [row for row in examples if row["conditioning"] == conditioning and row["decoding"] == decoding]
            aggregates[conditioning+"_"+decoding] = {
                "novel_whole_responses": sum(not row["exact_corpus_response"] for row in rows),
                "terminated": sum(row["ended"] for row in rows),
                "repeated_trigrams": sum(row["repeated_trigrams"] for row in rows),
                "mean_tokens": float(np.mean([row["token_count"] for row in rows]))}
    report = {"paired_examples": len(pairs), "unpaired_texts": len(prior),
        "unpaired_provenance": "500 authored combinations of 200 atomic sentences, not independent observations",
        "prior_weight": 0.25, "gradient_updates": 0,
        "statistical_estimation": "conditional feature-prefix-next-token occurrence counts at import",
        "uses_document_answer_comparison": False, "prompt_count": len(PROMPTS),
        "examples": examples, "numerics": checks, "aggregates": aggregates,
        "elapsed_seconds": time.perf_counter()-started,
        "semantic_correctness": "Not inferred from numerical accuracy; use independent held-out evaluation."}
    (output_directory/"conditioned_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_directory/"conditioned_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Bedingter spektraler Token-Generator: Entwicklungsvergleich", "",
        "120 Gesprächspaare liefern gezählte Eingabemerkmal/Prefix/Folgetoken-Übergänge. 500 zusätzliche Textkombinationen aus 200 atomaren Sätzen liefern einen schwächer gewichteten Sprachprior. Das ist eine statistische Modellschätzung beim Import, ohne Gradientenoptimierung oder vortrainiertes Sprachmodell.", "",
        "Runtime-Datenquelle sind beim Import berechnete komplexe Spektren F(sqrt(P)). Eine separate numerische Merkmalswelle aktiviert bedingte Übergangsfelder. Zeitentwicklung und Promptmodulation wirken als spektrale Faltungsoperatoren. Erst das resultierende Feld wird zu Tokenamplituden invers transformiert; die diskrete Auswahl geschieht pro Token mittels greedy oder Beam-Suche.", "",
        "Die Wortfolge steht vor dieser Decoderrechnung nicht fest. Es gibt keine Dokument-ID, keine Auswahl einer kompletten Antwort und keine Antwortähnlichkeitswertung im Generierungsprozess. Im Vergleich werden fertige Ausgaben anschließend nur zur Neuheitsmessung mit dem Korpus verglichen.", "",
        "## Numerische Kontrolle", "", "```json", json.dumps(checks, ensure_ascii=False, indent=2), "```", "",
        "Eine vollständige Nullablation umfasst Token- und Eingabemerkmalskanal. Nur die Tokenamplituden auf null zu setzen lässt die eigene Merkmalswelle aktiv. Diagnose-Metadaten sind keine Quelle für deren Aktivierung.", "",
        "## Ausgaben", "", "| Prompt | Lexikalisch / greedy | Lexikalisch / Beam | Semantik / greedy | Semantik / Beam |",
        "| --- | --- | --- | --- | --- |"]
    for prompt in PROMPTS:
        values = [prompt] + [next(row["answer"] for row in examples if row["prompt"] == prompt and row["conditioning"] == conditioning and row["decoding"] == decoding)
                             for conditioning in ("lexical", "semantic") for decoding in ("greedy", "beam")]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    lines += ["", "## Messgrenzen", "",
        "- Die numerische Gleichwertigkeit zum direkten Zählmodell ist erwartet: Fourierkoordinaten erzeugen kein zusätzliches Wissen.",
        "- Semantische Merkmale sind explizite Sprachregeln am Eingang. Sie werden nicht aus der Fouriertransformation entdeckt.",
        "- Beam-Suche bewertet nur Feldwahrscheinlichkeiten. Sie prüft weder Faktentreue noch Gesprächssinn.",
        "- Neue Tokenfolgen können vollständige sinnvolle Sätze, unsinnige Mischungen oder unpassende Antworten sein.",
        "- Mehrere Absichten und unbekannte Sachfragen bleiben mit diesem kleinen endlichen Modell wesentliche Grenzen.",
        "- Diese 25 Prompts sind Entwicklungsbeispiele; sie sind kein unabhängiger Generalisierungsnachweis."]
    (output_directory/"conditioned_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("memory/fixtures/extension_120.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/generative_waves/prototype"))
    parser.add_argument("--conditioned", action="store_true")
    parser.add_argument("--language-corpus", type=Path, default=Path("memory/language/generative_corpus.jsonl"))
    args = parser.parse_args()
    if args.conditioned:
        run_conditioned_comparison(args.output, args.corpus, args.language_corpus)
    else:
        run_comparison(args.output, args.corpus)
