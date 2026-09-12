# Isolated operator padding proposal

This is a performance proposal. Production code is unchanged by the prototype.
The existing conversation-quality evaluation must finish before integration.

For V known token modes, choose N = scipy.fft.next_fast_len(V). The runtime
carrier contains the same first V modes followed by N-V zero-amplitude modes.
Padding is applied to token-mode vectors BEFORE their FFT, never to spectral
coefficients. Stored complex data fields retain their original coordinate
meaning and shape, and every known token retains its existing frequency.

The modified operator calculates the same interference in this larger Fourier
basis. It decodes IFFT(result)[:V] to score words. The diagnostic roundtrip uses
the complete N-element result. The direct method remains the original V-mode
reference; HRR and the numerical feature channel are unchanged.

Minimal integration after the existing evaluation:

1. Expose `model.carrier_size = next_fast_len(model.size)` when the token
   dictionary is compiled. Use this only for the operator runtime transform.
2. Apply the changes in `operator_padding_method.diff`: basis, phase and prompt
   FFTs use the same N; scores use the first V decoded amplitudes; trace records
   N separately. Dense imported global fields are demodulated before padding.
3. The context display must use
   `FFT(context_amplitudes * token_phases, n=model.carrier_size, norm="ortho")`.
   Keep its known-mode count at V and separately report the carrier size N.
   Stored context amplitudes remain indexed by original token names, length V.
4. Extend the exact-context-display test to this same N-dimensional carrier.
   Test padding only zero modes, stable old frequencies after vocabulary growth,
   dense/compact references, phase interventions and data-wave nulling.
5. Compare all already-completed evaluation output token sequences before and
   after integration. A changed word is a failed performance-only regression;
   do not treat this as an opportunity to tune answer quality on the holdout.

No pruning, stored-response selection, new semantic rules, count-weight change,
or model training is included. The NumPy-to-SciPy FFT backend substitution is
not needed for this proposal.
