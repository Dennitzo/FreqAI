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
    carrier_size = next_fast_len(self.size) if method == "operator" else self.size
    if self.storage == "compact":
        base_spectrum = np.fft.fft(base_modes, n=carrier_size, norm="ortho")
    elif carrier_size != self.size:
        base_spectrum = np.fft.fft(np.fft.ifft(base_spectrum, norm="ortho"),
                                   n=carrier_size, norm="ortho")
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
    phase_spectrum = np.fft.fft(phases, n=carrier_size, norm="ortho")
    if method == "direct":
        # Independent pointwise reference, bypassing spectral time and
        # interaction convolutions. It should have identical probabilities.
        direct_amplitudes = np.fft.ifft(output_spectrum, norm="ortho") * phases
        direct_amplitudes *= 1 + prompt_gain*np.asarray(prompt_field)*np.exp(1j*phase_error)
        result_spectrum = np.fft.fft(direct_amplitudes, norm="ortho")
    else:
        moving_spectrum = spectral_convolution(output_spectrum, phase_spectrum)
        prompt_spectrum = np.fft.fft(np.asarray(prompt_field), n=carrier_size, norm="ortho")
        interaction = prompt_gain*spectral_convolution(moving_spectrum, prompt_spectrum)*np.exp(1j*phase_error)
        result_spectrum = moving_spectrum + interaction
    # The actual token amplitudes are obtained from the resulting complex
    # field. There is no earlier planned token or complete answer string.
    amplitudes = np.fft.ifft(result_spectrum, norm="ortho")
    scores = np.abs(amplitudes[:self.size])**2
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
             "decoder": method, "support_size": int(support.sum()), "carrier_bins": carrier_size,
             "conditioned_field_sources": conditional_sources,
             "feature_field_norm": float(np.linalg.norm(feature_spectrum)),
             "feature_decoder": "inverse_dft_mode_amplitude",
             "runtime_source": "imported_complex_transition_spectra",
             "fft_roundtrip_max_error": float(np.max(np.abs(np.fft.fft(amplitudes, norm="ortho")-result_spectrum))),
             "phase_time_s": time_s, "prompt_gain": prompt_gain,
             "grammar_mask": True, "support_source": "corpus_ngram_edges", "phase_error": phase_error}
    return probabilities, trace
