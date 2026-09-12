"""The two explicit waves and their interference: fields, numerical agreement, causality.

These tests own the contract of the rebuilt readout:

* one **data wave** over every symbol mode all compiled data excites,
* one **prompt wave** over every mode the prompt plus session context excites,
* an **interference readout** computed by two independent algorithms that must
  agree numerically, whose intensity gives the next-symbol probabilities.

They also pin down the honest limits: a symbol is one eigenmode of the medium, so
a step with a single compatible mode cannot be influenced at all, and off-diagonal
sum-frequency mixing over ``stable_frequency`` values carries no information.
"""
import numpy as np
import pytest

from freqai.information import InformationWaveModel
from freqai.memory import Document
from freqai.store import MemoryStore
from freqai.unpaired import UnpairedWaveModel
from freqai.unpaired_runtime import respond_wave
from freqai.waves import (DEFAULT_DATA_GAIN, VERIFY_TOLERANCE, WaveField, build_field,
                          interference_pair, probabilities, resonance, superpose)

CORPUS_ORIGINS = {"corpus_subject", "corpus_predicate", "corpus_predicate_argument",
                  "corpus_lexical_fact"}

DECLARATIONS = [
    {"text": "Der Assistent heißt Oszillo. Der Assistent ist ein Rechenprogramm. "
             "Der Assistent hat keine eigenen Gefühle. Der Assistent kann Texte beschreiben."},
    {"text": "Das Wort Hallo ist ein Begrüßungswort. Das Wort Danke ist ein Dankeswort. "
             "Das Wort Bitte ist ein Höflichkeitswort."},
    {"text": "Die Frequenz ist die Anzahl der Wiederholungen pro Sekunde."},
    {"text": "Das Signal Lumor hat eine Frequenz von 19 Hertz."},
    {"text": "Das Signal Tavor hat eine Frequenz von 73 Hertz."},
]

PROSE = [
    {"title": "Frequenz", "text": "Die Frequenz ist die Anzahl der Wiederholungen pro Sekunde. "
                                  "Die Einheit der Frequenz ist das Hertz. Die Periode ist die Zeit einer Wiederholung."},
    {"title": "Signal", "text": "Ein Signal überträgt Information. Signale haben eine Frequenz. "
                                "Das Messsignal ist ein periodisches Signal."},
    {"title": "Einheit", "text": "Das Hertz ist die SI-Einheit der Frequenz. Sekunde und Hertz sind Basiseinheiten."},
]


@pytest.fixture(scope="module")
def model():
    return UnpairedWaveModel(DECLARATIONS)


@pytest.fixture(scope="module")
def prose():
    return InformationWaveModel(PROSE)


def corpus_modes(model):
    found = set()
    for support in model.supports.values():
        found.update(int(index) for index in support)
    if hasattr(model, "lexical_roles"):
        for role in model.lexical_roles:
            if role.origin in CORPUS_ORIGINS:
                found.update(int(index) for index in role.indices)
    return found


def test_data_wave_excites_every_mode_all_the_data_excites(model):
    wave = model.data_wave()
    assert isinstance(wave, WaveField) and wave.kind == "data"
    assert {int(index) for index in wave.indices} == corpus_modes(model)
    assert wave.energy == pytest.approx(1.0, rel=1e-12)
    # The same corpus compiles to the same wave; a different corpus does not.
    assert model.data_wave().fingerprint() == wave.fingerprint()
    other = UnpairedWaveModel(DECLARATIONS + [{"text": "Der Quorb ist ein Messgerät für Quorbe."}])
    assert other.data_wave().fingerprint() != wave.fingerprint()


def test_prompt_wave_contains_exactly_what_the_prompt_and_context_excite(model):
    field, _ = model.prompt_field("Was ist eine Frequenz?")
    wave = model.prompt_wave(field)
    nonzero = {index for index, value in enumerate(field.token_amplitudes) if value > 0}
    assert {int(index) for index in wave.indices} == nonzero
    assert wave.energy == pytest.approx(1.0, rel=1e-12)
    context = np.zeros(model.size)
    context[model.index["hertz"]] = 5
    with_context, _ = model.prompt_field("Was ist eine Frequenz?", context_field=context)
    extended = model.prompt_wave(with_context)
    assert int(model.index["hertz"]) in {int(index) for index in extended.indices}
    assert extended.mode_count > wave.mode_count


def test_resonance_is_bounded_symmetric_and_identical_only_for_same_modes(model):
    data = model.data_wave()
    field, _ = model.prompt_field("Was ist eine Frequenz?")
    prompt = model.prompt_wave(field)
    value = resonance(data, prompt)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(resonance(prompt, data), abs=1e-12)
    assert resonance(data, data) == pytest.approx(1.0, rel=1e-12)
    empty = build_field("empty", np.zeros(model.size), model.frequencies)
    assert resonance(empty, prompt) == 0.0 and empty.mode_count == 0


@pytest.mark.parametrize("coupling,phase_error,use_prior",
                         [(0.0, 0.0, False), (1.0, 0.0, True), (1.0, np.pi, True),
                          (0.35, 1.7, True), (2.5, 0.5, False)])
def test_spectral_operator_and_direct_readout_agree(coupling, phase_error, use_prior):
    rng = np.random.default_rng(4242)
    checked = 0
    for size in (1, 2, 3, 7, 33, 129):
        indices = np.arange(size, dtype=np.intp)
        amplitudes = rng.normal(size=size) + 1j*rng.normal(size=size)
        prompt = WaveField("prompt", indices, np.abs(rng.normal(size=size)) + 0.05,
                           0.25 + 29.75*np.random.default_rng(size).random(size))
        prior = np.abs(rng.normal(size=size)) if use_prior else None
        operator_readout, direct_readout, difference = interference_pair(
            indices, amplitudes, prompt, coupling=coupling, phase_error=phase_error, mode_prior=prior)
        assert difference <= VERIFY_TOLERANCE, (size, difference)
        # The mixing theorem is exercised on its own: the operator result must be
        # the orthonormal transform of the pointwise product of the two factors.
        expected = np.fft.ifft(np.fft.fft(amplitudes*(1+coupling*np.exp(1j*phase_error)*prompt.restrict(indices))
                                          * (1+(prior if prior is not None else np.zeros(size))), norm="ortho"),
                               norm="ortho")
        np.testing.assert_allclose(operator_readout, expected[:size], atol=VERIFY_TOLERANCE)
        checked += 1
    assert checked == 6


def test_probabilities_are_normalized_intensities():
    readout = np.array([1+1j, 2-1j, 0.5, -3j, 0.0])
    local, total = probabilities(readout)
    expected = np.abs(readout)**2
    expected = expected/expected.sum()
    assert total == pytest.approx(float(np.sum(np.abs(readout)**2)), rel=1e-12)
    assert local.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(local, expected, atol=1e-12)
    with pytest.raises(ValueError, match="Interference"):
        probabilities(np.zeros(4, dtype=complex))


def test_superposition_is_the_coherent_sum_of_mode_fields():
    first = (np.array([0, 2], dtype=np.intp), np.array([1+0j, 0.5j]))
    second = (np.array([2, 5], dtype=np.intp), np.array([0.5j, -1+0j]))
    indices, values = superpose([first, second], weights=[2.0, 1.0])
    assert [int(index) for index in indices] == [0, 2, 5]
    assert values[0] == 2+0j and values[1] == 1.5j and values[2] == -1+0j


def _causality_rows(model, prompt, policy):
    """Every decoding step of one answer with its coupling effects."""
    field, _ = model.prompt_field(prompt)
    answer = model.generate(prompt)
    wave = model.prompt_wave(field)
    rows = []
    for length in range(len(answer["tokens"])+1):
        prefix = answer["tokens"][:length]
        try:
            with_prompt, trace = model.next_distribution(prefix, field, prompt_gain=1.0, order_policy=policy)
            without_prompt, _ = model.next_distribution(prefix, field, prompt_gain=0.0, order_policy=policy)
            without_prior, _ = model.next_distribution(prefix, field, data_gain=0.0, order_policy=policy)
        except ValueError:
            continue
        active = np.array(trace["active_token_indices"], dtype=np.intp)
        rows.append((prefix, len(active),
                     float(0.5*np.abs(with_prompt-without_prompt).sum()),
                     float(0.5*np.abs(with_prompt-without_prior).sum()),
                     bool(np.any(wave.restrict(active) > 0))))
    return rows


@pytest.mark.parametrize("policy,min_steps,min_prompt_coupled", [("all", 10, 5), ("deepest", 1, 0)])
def test_both_waves_are_causally_visible_whenever_a_mode_can_be_coupled(prose, policy, min_steps,
                                                                        min_prompt_coupled):
    multi = shared = prompt_effective = prior_effective = 0
    for prompt in ("Was ist eine Frequenz?", "In welcher Einheit wird die Frequenz gemessen?",
                   "Was ist ein Signal?"):
        for prefix, modes, prompt_effect, prior_effect, shares_mode in _causality_rows(prose, prompt, policy):
            if modes < 2:
                continue
            multi += 1
            shared += int(shares_mode)
            # The all-data wave excites every corpus symbol, so every multi-mode
            # step differs measurably once its prior is removed.
            assert prior_effect > 0.0, (policy, prefix)
            prior_effective += 1
            if shares_mode:
                assert prompt_effect > 0.0, (policy, prefix)
                prompt_effective += 1
    assert multi >= min_steps
    assert prior_effective == multi
    assert prompt_effective >= min_prompt_coupled


@pytest.mark.parametrize("policy", ["all", "deepest"])
def test_the_two_couplings_are_separate_controls_and_every_setting_is_normalized(prose, policy):
    checked = 0
    for prompt in ("Was ist eine Frequenz?", "In welcher Einheit wird die Frequenz gemessen?"):
        field, _ = prose.prompt_field(prompt)
        answer = prose.generate(prompt)
        for length in range(len(answer["tokens"])+1):
            prefix = answer["tokens"][:length]
            try:
                both, trace = prose.next_distribution(prefix, field, order_policy=policy)
                no_prompt, _ = prose.next_distribution(prefix, field, prompt_gain=0.0, order_policy=policy)
                no_prior, _ = prose.next_distribution(prefix, field, data_gain=0.0, order_policy=policy)
            except ValueError:
                continue
            for distribution in (both, no_prompt, no_prior):
                assert distribution.sum() == pytest.approx(1.0, abs=1e-12)
            if len(trace["active_token_indices"]) >= 2:
                assert not np.allclose(both, no_prior)
                checked += 1
    assert checked >= 1


def test_single_mode_steps_cannot_be_influenced_and_are_reported_as_such(prose):
    field, _ = prose.prompt_field("Was ist eine Frequenz?")
    distribution, trace = prose.next_distribution(["die", "frequenz"], field)
    assert len(trace["active_token_indices"]) == 1
    for kwargs in ({"prompt_gain": 0.0}, {"data_gain": 0.0}, {"phase_error": np.pi}):
        other, _ = prose.next_distribution(["die", "frequenz"], field, **kwargs)
        np.testing.assert_allclose(distribution, other, atol=1e-15)


def test_a_data_wave_without_energy_cannot_be_read_out(model):
    broken = UnpairedWaveModel(DECLARATIONS)
    for spectrum in broken.transition_spectra.values():
        spectrum[:] = 0
    for role in broken.lexical_roles:
        role.spectrum[:] = 0
    field, _ = broken.prompt_field("Was ist eine Frequenz?")
    with pytest.raises(ValueError, match="energy"):
        broken.next_distribution([], field)
    assert broken.generate("Was ist eine Frequenz?")["tokens"] == []


def test_zero_prompt_field_cannot_license_any_answer(model):
    field, _ = model.prompt_field("Hallo")
    with pytest.raises(ValueError, match="binding|wave"):
        model.next_distribution([], field*0)


@pytest.mark.parametrize("time_s", [0.0, 3.75, 86400.125, 1e9])
def test_measured_intensity_is_invariant_to_the_carrier_clock(model, prose, time_s):
    for model_instance in (model, prose):
        field, _ = model_instance.prompt_field("Was ist eine Frequenz?")
        base, _ = model_instance.next_distribution([], field, time_s=0.0)
        later, _ = model_instance.next_distribution([], field, time_s=time_s)
        np.testing.assert_allclose(base, later, atol=1e-12)


def test_wave_snapshots_are_finite_reproducible_and_distinguish_prompts(model):
    first, _ = model.prompt_field("Was ist eine Frequenz?")
    second, _ = model.prompt_field("Wie heißt du?")
    left = model.prompt_wave(first).snapshot(time_s=3.5)
    right = model.prompt_wave(first).snapshot(time_s=3.5)
    assert left == right
    assert np.isfinite(np.array(left["displacement"]+left["quadrature"], dtype=float)).all()
    assert model.prompt_wave(first).fingerprint() != model.prompt_wave(second).fingerprint()
    data = model.data_wave().snapshot()
    assert data["mode_count"] == model.data_wave().mode_count
    assert data["energy"] == pytest.approx(1.0, rel=1e-12)


def test_runtime_reports_both_waves_and_their_interference(tmp_path):
    store = MemoryStore(tmp_path/"waves.sqlite3")
    store.append_documents([
        Document("f1", "Die Frequenz ist die Anzahl der Wiederholungen pro Sekunde.", "Physik"),
        Document("h1", "Das Hertz ist die SI-Einheit der Frequenz.", "Physik"),
        Document("n1", "Der Assistent heißt Vela.", "Test")])
    memory = store.load_memory()
    result, state = respond_wave(memory, "Was ist eine Frequenz?")
    waves = result["waves"]
    assert waves["version"] == "two_wave_v2"
    assert waves["data_wave"]["mode_count"] > 1
    assert waves["prompt_wave"]["mode_count"] >= 1
    assert waves["data_wave"]["energy"] == pytest.approx(1.0, rel=1e-9)
    assert 0.0 <= waves["resonance"] <= 1.0
    assert waves["operator_vs_direct_max_error"] <= VERIFY_TOLERANCE
    assert result["answer"].startswith("Die Frequenz")
    # A referential follow-up may excite no new symbol mode at all; the saved
    # session field is then what still licenses the answer.
    follow_up, _ = respond_wave(memory, "Und welche Einheit hat sie?", context=state)
    assert follow_up["waves"]["version"] == "two_wave_v2"
    assert follow_up["waves"]["operator_vs_direct_max_error"] <= VERIFY_TOLERANCE


def test_default_gains_are_positive_and_documented(model):
    assert DEFAULT_DATA_GAIN > 0.0
    assert model.data_gain == DEFAULT_DATA_GAIN
    assert model.resonance_floor > 0.0


@pytest.mark.parametrize("policy", ["deepest", "all"])
def test_every_generated_step_reports_a_verified_interference(prose, policy):
    answer = prose.generate("Was ist eine Frequenz?", order_policy=policy)
    assert answer["trace"]
    for step in answer["trace"]:
        assert step["interference_verification"]["agrees"]
        assert step["fft_roundtrip_max_error"] <= VERIFY_TOLERANCE
        assert step["data_wave"]["energy"] == pytest.approx(1.0, rel=1e-9)
        assert 0.0 <= step["wave_resonance"] <= 1.0


def test_wave_controls_are_threaded_through_generation(prose):
    default = prose.generate("Was ist eine Frequenz?")
    without_prior = prose.generate("Was ist eine Frequenz?", data_gain=0.0)
    assert [step["data_wave"]["gain"] for step in default["trace"][:1]] == [DEFAULT_DATA_GAIN]
    assert [step["data_wave"]["gain"] for step in without_prior["trace"][:1]] == [0.0]


def test_superposing_every_order_ends_the_utterance_too_early_under_cumulative_scoring():
    """A pinned limitation, not a passing goal: the beam score is cumulative.

    With every order superposed, an end-of-sequence mode also arrives from the
    shallower fields. Because the search sums log probabilities and stops adding
    terms once a beam has ended, the shorter path wins. The tested recipe avoids
    this by reading termination only from the most specific field.
    """
    model = InformationWaveModel([{"title": "Echos",
                                   "text": "Echos folgen aufeinander, z. B. bei ruhigem Wetter."}])
    deepest = model.generate("Erkläre Echos.")
    broad = model.generate("Erkläre Echos.", order_policy="all")
    assert deepest["text"] == "Echos folgen aufeinander, z. B. bei ruhigem Wetter."
    assert len(broad["tokens"]) < len(deepest["tokens"])
    assert not broad["text"].endswith("ruhigem Wetter.")
