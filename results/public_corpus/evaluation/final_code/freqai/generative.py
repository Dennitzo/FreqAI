"""Autoregressive token generation from interacting conditional Fourier fields.

Corpus counts compile the language operator; explicit input encoding and a
finite token alphabet supply linguistic prior knowledge. No complete reply
comparison or fragment retrieval is performed by this decoder.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Real
import re

import numpy as np
from scipy.sparse import csr_matrix

from .spectral_storage import CompactSpectrum, SparseDistributionReference, add_spectral_modes

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
                 conditioned_pairs=None, conditioning="semantic", prior_weight=0.25,
                 annotated_texts=None, storage="compact"):
        if type(order) is not int or not 1 <= order <= 5:
            raise ValueError("order must be an integer from 1 to 5")
        if conditioning not in {"semantic", "lexical"}:
            raise ValueError("conditioning must be semantic or lexical")
        if not np.isfinite(prior_weight) or not 0 < prior_weight <= 1:
            raise ValueError("prior_weight must be in (0,1]")
        if storage not in {"compact", "dense"}:
            raise ValueError("storage must be compact or dense")
        self.order = order
        self.storage = storage
        self.hrr_dimensions = hrr_dimensions
        self.conditioning = conditioning
        pairs = list(conditioned_pairs or [])
        pair_weights = []
        for pair in pairs:
            weight = pair.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, Real) or not np.isfinite(weight) or not 0 < weight <= 1:
                raise ValueError("pair weight must be finite and in (0,1]")
            pair_weights.append(float(weight))
            if type(pair.get("semantic_features", True)) is not bool:
                raise ValueError("semantic_features must be a boolean")
        annotations = list(annotated_texts or [])
        self.paired_example_count = len(pairs)
        self.annotated_example_count = len(annotations)
        paired_texts = [str(pair["text"]) for pair in pairs]
        corpus_texts = tuple(dict.fromkeys([str(text) for text in texts] + paired_texts +
                                         [str(item["text"]) for item in annotations]))
        sequences = [tokenize(text) for text in corpus_texts]
        token_sequences = dict(zip(corpus_texts, sequences))
        self.surface_forms = defaultdict(Counter)
        for text in corpus_texts:
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
        paired_text_weights = {}
        for text, weight in zip(paired_texts, pair_weights):
            paired_text_weights[text] = max(weight, paired_text_weights.get(text, 0.0))
        for text, seq in zip(corpus_texts, sequences):
            count_weight = paired_text_weights.get(text, prior_weight) if pairs else 1.0
            extended = [BOS] * order + seq + [EOS]
            for position in range(order, len(extended)):
                for depth in range(order + 1):
                    key = tuple(extended[position-depth:position]) if depth else ()
                    self.counts[key][extended[position]] += count_weight
        self.transition_spectra, self.supports, reference_rows = self._compile_fields(self.counts, references=True)
        # Real probabilities are a lazy diagnostic reference, never the runtime
        # source. Complex support-local FFTs need O(observed edges), not O(V*fields).
        self.distributions = SparseDistributionReference(self.size, reference_rows)
        if storage == "dense":
            self.distributions = dict(self.distributions.items())
        conditioned_counts = defaultdict(Counter)
        self.feature_frequency = Counter()
        feature_cache = {}
        entries = []
        for pair, weight in zip(pairs, pair_weights):
            prompt = str(pair["prompt"])
            include_semantics = pair.get("semantic_features", True)
            key = (prompt, include_semantics)
            if key not in feature_cache:
                feature_cache[key] = self._input_features(prompt, include_semantics=include_semantics)
            entries.append((feature_cache[key], str(pair["text"]), weight))
        entries.extend((dict(item["features"]), str(item["text"]), prior_weight) for item in annotations)
        # Coarse topics can also be derived from unpaired text. This does not
        # invent a conversational input/output relation for arbitrary sentences.
        annotated_set = {str(item["text"]) for item in annotations}
        for text in corpus_texts:
            if text not in paired_set and text not in annotated_set:
                topics = {key: value for key, value in self._input_features(text).items() if key.startswith("topic:")}
                if topics:
                    entries.append((topics, text, prior_weight))
        for features, text, entry_weight in entries:
            self.feature_frequency.update(features.keys())
            seq = [BOS]*order + token_sequences[text] + [EOS]
            transitions = []
            for position in range(order, len(seq)):
                for depth in range(order+1):
                    key = tuple(seq[position-depth:position]) if depth else ()
                    transitions.append((key, seq[position]))
            for feature in features:
                for key, token in transitions:
                    conditioned_counts[(feature, key)][token] += entry_weight
        self.conditioned_spectra, self.conditioned_supports, _ = self._compile_fields(conditioned_counts)
        del conditioned_counts, entries, feature_cache
        self.feature_vocabulary = sorted(self.feature_frequency)
        self.feature_index = {feature: i for i, feature in enumerate(self.feature_vocabulary)}
        self.feature_frequencies = np.array([stable_frequency("feature:"+feature) for feature in self.feature_vocabulary])
        # A response-only local co-occurrence prior, explicitly not an input->
        # response label lookup. Window size limits diffuse document-level noise.
        association = defaultdict(Counter)
        document_frequency = Counter()
        for seq in sequences:
            document_frequency.update(set(seq))
            for i, left in enumerate(seq):
                if left in STOP or not left.isalpha():
                    continue
                for j in range(max(0, i-6), min(len(seq), i+7)):
                    right = seq[j]
                    if left != right and right.isalpha():
                        association[self.index[left]][self.index[right]] += 1 / (1 + abs(i-j))
        indptr, indices, values = [0], [], []
        for i in range(self.size):
            row = association.get(i, {})
            norm = max(max(row.values(), default=0), 1e-12)
            for column in sorted(row):
                indices.append(column)
                values.append(row[column]/norm)
            indptr.append(len(indices))
        self.association = csr_matrix((values, indices, indptr), shape=(self.size, self.size), dtype=np.float64)
        if storage == "dense":
            self.association = self.association.toarray()
        self.idf = np.array([math.log1p(len(sequences)/(1+document_frequency[word])) for word in self.vocabulary])
        self._hrr_ready = False
        self._hrr_cache = {}

    def _compile_fields(self, count_rows, references=False):
        fields, supports, real_rows = {}, {}, {}
        for key, counts in count_rows.items():
            ordered = sorted((self.index[word], count) for word, count in counts.items())
            indices = np.fromiter((i for i, _ in ordered), dtype=np.intp, count=len(ordered))
            probabilities = np.fromiter((count for _, count in ordered), dtype=float, count=len(ordered))
            probabilities /= probabilities.sum()
            amplitudes = np.sqrt(probabilities)
            supports[key] = indices
            if self.storage == "compact":
                fields[key] = CompactSpectrum(self.size, indices, np.fft.fft(amplitudes, norm="ortho"))
            else:
                vector = np.zeros(self.size)
                vector[indices] = amplitudes
                fields[key] = np.fft.fft(vector, norm="ortho")
            if references:
                real_rows[key] = (indices, probabilities)
        return fields, supports, real_rows

    def storage_stats(self):
        """Allocated numerical payload, excluding Python object/table overhead."""
        banks = (self.transition_spectra, self.conditioned_spectra)
        field_count = sum(len(bank) for bank in banks)
        coefficient_bytes = sum(spectrum.nbytes for bank in banks for spectrum in bank.values())
        support_bytes = sum(indices.nbytes for bank in (self.supports, self.conditioned_supports) for indices in bank.values())
        if isinstance(self.association, csr_matrix):
            association_bytes = self.association.data.nbytes + self.association.indices.nbytes + self.association.indptr.nbytes
        else:
            association_bytes = self.association.nbytes
        reference_bytes = (self.distributions.nbytes if isinstance(self.distributions, SparseDistributionReference)
                           else sum(value.nbytes for value in self.distributions.values()))
        return {"storage": self.storage, "vocabulary_size": self.size, "field_count": field_count,
                "complex_coefficient_bytes": coefficient_bytes, "support_index_bytes": support_bytes,
                "reference_probability_bytes": reference_bytes, "association_bytes": association_bytes,
                "numerical_payload_bytes": coefficient_bytes+support_bytes+reference_bytes+association_bytes,
                "equivalent_dense_coefficient_bytes": field_count*self.size*np.dtype(complex).itemsize,
                "equivalent_dense_association_bytes": self.size*self.size*np.dtype(float).itemsize,
                "excludes": "Python objects, count tables, token strings, temporary compilation and generation arrays"}

    def _input_features(self, text, context=None, active_act=None, *, include_semantics=True):
        from .features import canonical_term
        features = {"word:"+canonical_term(word): 1.0 for word in tokenize(text) if word not in STOP and word.isalpha()}
        if self.conditioning == "semantic" and include_semantics:
            from .semantics import analyze
            acts = [active_act] if active_act is not None else analyze(text, context).acts
            for act in acts:
                if act.kind == "context_clarification":
                    features["intent:clarification"] = 8.0
                    continue
                if act.topic and act.topic not in {"wellbeing", "name", "identity", "capabilities", "conversation"}:
                    features["topic:" + act.topic] = 6.0
                if act.kind == "question":
                    continue
                if act.kind in {"name_question", "name_statement"}:
                    features["intent:name"] = 8.0
                    continue
                # Polarities and speaker roles remain distinct input addresses.
                value = act.value if act.kind in {"mood_statement", "acknowledgement"} else ""
                signature = ":".join((act.kind, act.target, value, act.topic))
                features["semantic:"+signature] = 8.0
        return features

    def render(self, tokens):
        return detokenize([token if token in STOP else self.display_tokens.get(token, token) for token in tokens])

    def prompt_field(self, prompt: str, context_text: str = "", context=None, active_act=None) -> tuple[np.ndarray, dict]:
        field = np.zeros(self.size)
        observed = []
        unknown = []
        for text, weight in ((context_text, 0.35), (prompt, 1.0)):
            for word in tokenize(text):
                if word in STOP or not word.isalpha():
                    continue
                if word in self.index:
                    i = self.index[word]
                    if isinstance(self.association, csr_matrix):
                        start, end = self.association.indptr[i:i+2]
                        field[self.association.indices[start:end]] += weight * self.idf[i] * self.association.data[start:end]
                    else:
                        field += weight * self.idf[i] * self.association[i]
                    field[i] += weight * self.idf[i] * 0.2
                    observed.append(word)
                else:
                    unknown.append(word)
        if field.max() > 0:
            field /= field.max()
        features = self._input_features(active_act.text if active_act is not None else prompt, context, active_act)
        feature_amplitudes = np.zeros(len(self.feature_vocabulary))
        for feature, weight in features.items():
            if feature in self.feature_index:
                # A categorical role/polarity does not become less authoritative
                # when more examples of that category are imported. Only lexical
                # rarity is IDF-weighted; structural-channel gains are explicit.
                scale = math.sqrt(max(1, self.feature_frequency[feature])) if feature.startswith("word:") else 1.0
                feature_amplitudes[self.feature_index[feature]] = weight/scale
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
        if not self.transition_spectra:
            raise ValueError("Empty data wave: no token transitions")
        if not np.isfinite([time_s, prompt_gain, phase_error]).all() or prompt_gain < 0:
            raise ValueError("Invalid field controls")
        if np.asarray(prompt_field).shape != (self.size,) or not np.isfinite(prompt_field).all():
            raise ValueError("Invalid prompt wave")
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
        # Decode support-local complex coefficients into one shared carrier.
        # Linearity gives F(sum(E_I F_I^-1(C_I))) = sum(global field spectra).
        # A dense reference follows the original frequency-domain accumulation.
        if self.storage == "compact":
            base_modes = np.zeros(self.size, dtype=np.complex128)
            for key, weight in zip(keys, weights):
                add_spectral_modes(base_modes, self.transition_spectra[key], weight)
            base_spectrum = None
        else:
            base_spectrum = sum(weight*self.transition_spectra[key] for key, weight in zip(keys, weights))
        support = np.zeros(self.size, dtype=bool)
        for key in keys:
            support[self.supports[key]] = True
        conditional_sources = []
        feature_spectrum = getattr(prompt_field, "feature_spectrum", np.array([], dtype=complex))
        if (np.asarray(feature_spectrum).ndim != 1 or
                len(feature_spectrum) not in (0, len(self.feature_vocabulary)) or
                not np.isfinite(feature_spectrum).all()):
            raise ValueError("feature spectrum does not match the model's feature dictionary")
        if len(feature_spectrum) and prompt_gain:
            feature_phases = np.exp(2j*np.pi*np.remainder(self.feature_frequencies*time_s, 1.0))
            moving_features = spectral_convolution(feature_spectrum, np.fft.fft(feature_phases, norm="ortho"))
            feature_modes = np.abs(np.fft.ifft(moving_features, norm="ortho"))
        else:
            feature_modes = np.array([])
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
                    if self.storage == "compact":
                        add_spectral_modes(condition_wave, spectrum, weight)
                    else:
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
                if self.storage == "compact":
                    base_modes = 0.04*base_modes + 0.96*condition_wave
                else:
                    base_spectrum = 0.04*base_spectrum + 0.96*condition_wave
                support = common_support
        if self.storage == "compact":
            base_spectrum = np.fft.fft(base_modes, norm="ortho")
        if method in {"operator", "direct"}:
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
            raise ValueError("method must be operator, direct or hrr")
        phases = np.exp(2j*np.pi*np.remainder(self.frequencies*time_s, 1.0))
        phase_spectrum = np.fft.fft(phases, norm="ortho")
        if method == "direct":
            # Independent pointwise reference, bypassing spectral time and
            # interaction convolutions. It should have identical probabilities.
            direct_amplitudes = np.fft.ifft(output_spectrum, norm="ortho") * phases
            direct_amplitudes *= 1 + prompt_gain*np.asarray(prompt_field)*np.exp(1j*phase_error)
            result_spectrum = np.fft.fft(direct_amplitudes, norm="ortho")
        else:
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
        # Algebraically equal fields must not pick different words because of
        # a few ulps at a probability tie. Quantization is part of the decoder.
        probabilities = np.round(probabilities, 12)
        probabilities /= probabilities.sum()
        trace = {"prefix": list(prefix_tokens[-self.order:]),
                 "field_sources": [list(key) for key in keys], "field_weights": weights,
                 "decoder": method, "support_size": int(support.sum()),
                 "conditioned_field_sources": conditional_sources,
                 "feature_field_norm": float(np.linalg.norm(feature_spectrum)),
                 "feature_decoder": "inverse_dft_mode_amplitude",
                 "runtime_source": "imported_complex_transition_spectra",
                 "fft_roundtrip_max_error": float(np.max(np.abs(np.fft.fft(amplitudes, norm="ortho")-result_spectrum))),
                 "phase_time_s": time_s, "prompt_gain": prompt_gain,
                 "grammar_mask": True, "support_source": "corpus_ngram_edges", "phase_error": phase_error}
        return probabilities, trace

    def generate(self, prompt: str, context_text: str = "", *, context_field=None, method="operator", time_s=0.0,
                 max_tokens=48, seed=17, prompt_gain=1.2, prefix_enabled=True, phase_error=0.0,
                 temperature=0.8, decoding="sample", beam_width=4, max_sentences=2, context=None,
                 split_acts=False, require_conditioning=False, input_field=None):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 128:
            raise ValueError("max_tokens must be from 1 to 128")
        if type(beam_width) is not int or not 1 <= beam_width <= 16:
            raise ValueError("beam_width must be from 1 to 16")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not self.transition_spectra:
            return {"text": "", "tokens": [], "trace": [], "prompt_field": {}, "ended": True,
                    "method": method, "seed": seed, "reason": "empty_data_wave"}
        if split_acts and self.conditioning == "semantic" and input_field is None:
            from .semantics import analyze
            interpretation = analyze(prompt, context)
            acts, seen = [], set()
            for act in interpretation.acts:
                signature = (act.kind, act.target, act.value if act.kind == "mood_statement" else "", act.topic)
                if signature not in seen:
                    acts.append(act)
                    seen.add(signature)
            if 1 < len(acts) <= 4:
                # Time-multiplexed input bands preserve multiple intentions.
                # A sentence boundary advances the finite-state controller to
                # the next band; every sentence still comes from token fields.
                outputs, tokens, traces = [], [], []
                for band_index, act in enumerate(acts):
                    band, info = self.prompt_field(prompt, context_text, context, active_act=act)
                    if len(tokens) >= max_tokens:
                        outputs.append({"tokens": [], "trace": [], "prompt_field": info, "ended": False})
                        continue
                    part = self.generate(prompt, context_text, context_field=context_field, method=method,
                        time_s=time_s, max_tokens=min(max_tokens-len(tokens), max(1, max_tokens//len(acts))), seed=seed+band_index,
                        prompt_gain=prompt_gain, prefix_enabled=prefix_enabled, phase_error=phase_error,
                        temperature=temperature, decoding=decoding, beam_width=beam_width, max_sentences=1,
                        context=context, require_conditioning=require_conditioning, input_field=(band, info))
                    for trace in part["trace"]:
                        traces.append({**trace, "step": len(traces), "input_band": band_index})
                    tokens.extend(part["tokens"])
                    outputs.append(part)
                return {"text": self.render(tokens), "tokens": tokens, "trace": traces,
                        "prompt_field": {"source": "ordered_input_feature_waves",
                                         "bands": [part["prompt_field"] for part in outputs]},
                        "ended": all(part["ended"] for part in outputs), "method": method, "seed": seed,
                        "decoding": decoding, "beam_width": beam_width, "conditioning": self.conditioning,
                        "input_bands": len(acts), "unanswered_bands": sum(not part["tokens"] for part in outputs),
                        "source": "compiled_token_transition_spectra", "uses_prompt_keys_for_response_comparison": False,
                        "uses_paired_transition_statistics": bool(self.paired_example_count)}
        field, field_info = input_field if input_field is not None else self.prompt_field(prompt, context_text, context)
        feature_spectrum = np.asarray(getattr(field, "feature_spectrum", []))
        if (feature_spectrum.ndim != 1 or len(feature_spectrum) not in (0, len(self.feature_vocabulary))
                or not np.isfinite(feature_spectrum).all()):
            raise ValueError("Invalid numerical feature spectrum")
        if require_conditioning and np.linalg.norm(feature_spectrum) < 1e-12:
            return {"text": "", "tokens": [], "trace": [], "prompt_field": field_info, "ended": True,
                    "method": method, "seed": seed, "reason": "no_supported_input_coupling"}
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
                distribution = np.round(distribution, 12)
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
            candidates.sort(key=lambda item: (-round(item[0]/((5+max(1, len(item[1])))/6)**0.7, 12), tuple(item[1])))
            beams = candidates[:width]
            if all(item[3] for item in beams):
                break
        log_probability, tokens, traces, ended = beams[0]
        text = self.render(tokens)
        return {"text": text, "tokens": tokens, "trace": traces, "prompt_field": field_info,
                "ended": ended, "method": method, "seed": seed,
                "decoding": decoding, "beam_width": width, "log_probability": log_probability,
                "conditioning": self.conditioning,
                "source": "compiled_token_transition_spectra", "uses_prompt_keys_for_response_comparison": False,
                "uses_paired_input_statistics": bool(self.paired_example_count)}

