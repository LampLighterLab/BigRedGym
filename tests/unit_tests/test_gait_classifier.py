"""Exercise gait inference with measured contact events, without simulation."""

import math

import pytest

from gym.utils.gait_classifier import GaitClassifier, classify_relative_phases


IDEAL_PHASES = {
    "Trot": (math.pi, math.pi, 0.0),
    "Pace": (math.pi, 0.0, math.pi),
    "Bound": (0.0, math.pi, math.pi),
    "Pronk": (0.0, 0.0, 0.0),
}


def _classifier(**kwargs):
    # Synthetic 10 N/0 N signals only; these are not calibrated Go2 cutoffs.
    return GaitClassifier(contact_on=5.0, contact_off=2.0, **kwargs)


def _touchdowns(phases):
    references = tuple(float(t) for t in range(1, 6))
    return [references] + [
        tuple(t + phase / math.tau for t in references) for phase in phases
    ]


def _feed(classifier, touchdowns, *, stop=4.8, chatter=False):
    """Apply all simultaneous force changes in one timestamped sample."""
    events = {}
    for leg, times in enumerate(touchdowns):
        for time_s in times:
            changes = [(0.0, 10.0), (0.2, 0.0)]
            if chatter:
                changes += [(0.04, 3.0), (0.06, 6.0), (0.08, 3.0), (0.10, 6.0)]
            for delay, force in changes:
                timestamp = round(time_s + delay, 8)
                if 0.0 < timestamp <= stop:
                    events.setdefault(timestamp, []).append((leg, force))

    forces = [0.0] * 4
    results = [classifier.update(0.0, forces)]
    for timestamp in sorted(events):
        for leg, force in events[timestamp]:
            forces[leg] = force
        results.append(classifier.update(timestamp, forces))
    if not events or max(events) < stop:
        results.append(classifier.update(stop, forces))
    return results


@pytest.mark.parametrize("gait", IDEAL_PHASES)
def test_ideal_contacts_identify_each_gait(gait):
    results = _feed(_classifier(), _touchdowns(IDEAL_PHASES[gait]))
    result = results[-1]

    assert result.valid
    assert result.label == gait
    assert result.nearest_gait == gait
    assert result.distance == pytest.approx(0.0, abs=1e-10)
    assert result.distances[gait] == result.distance
    assert result.cycles_used == 2
    assert result.cycle_period_s == pytest.approx(1.0)
    assert result.time_s == pytest.approx(4.0)
    assert all(not result.valid for result in results if result.time_s < 3.0)


@pytest.mark.parametrize(
    ("phase", "label"),
    [(1.999, "Pronk"), (2.0, "Pronk"), (2.001, "Transition")],
)
def test_paper_threshold_uses_unsquared_euclidean_distance(phase, label):
    result = classify_relative_phases((phase, 0.0, 0.0))

    assert result.valid
    assert result.label == label
    assert result.nearest_gait == "Pronk"
    assert result.distance == pytest.approx(phase)


def test_phase_distance_wraps_at_two_pi():
    result = classify_relative_phases((math.tau - 0.02, 0.0, 0.0))

    assert result.label == "Pronk"
    assert result.distance == pytest.approx(0.02)


@pytest.mark.parametrize("gait", IDEAL_PHASES)
def test_touchdown_jitter_crosses_zero_and_half_cycle_boundaries(gait):
    references = tuple(float(t) for t in range(1, 6))
    partners = [
        tuple(
            t + phase / math.tau + (-0.03 if i % 2 == 0 else 0.03)
            for i, t in enumerate(references)
        )
        for phase in IDEAL_PHASES[gait]
    ]
    result = _feed(_classifier(), [references, *partners])[-1]

    assert result.valid
    assert result.label == gait
    assert result.distance < 0.1
    assert result.cycles_used == 2


def test_missing_leg_does_not_receive_a_gait_label():
    touchdowns = _touchdowns(IDEAL_PHASES["Trot"])
    touchdowns[2] = ()
    results = _feed(_classifier(), touchdowns)

    assert not any(result.valid for result in results)
    assert results[-1].relative_phases is None
    assert results[-1].nearest_gait is None
    assert results[-1].distance is None


def test_missing_partner_clears_a_previously_valid_gait():
    touchdowns = _touchdowns(IDEAL_PHASES["Trot"])
    touchdowns[2] = tuple(t for t in touchdowns[2] if t < 4.0)
    results = _feed(_classifier(), touchdowns, stop=5.2)

    assert any(result.valid for result in results)
    assert not results[-1].valid
    assert results[-1].label == "Insufficient stepping data"
    assert results[-1].relative_phases is None
    assert results[-1].nearest_gait is None
    assert results[-1].distance is None


def test_initial_loaded_feet_do_not_count_as_touchdowns():
    classifier = _classifier()
    for timestamp in (0.0, 0.5, 1.0, 1.5, 2.1):
        result = classifier.update(timestamp, [10.0] * 4)
        assert not result.valid
        assert result.relative_phases is None
        assert result.label != "Pronk"


def test_contact_hysteresis_rejects_chatter_during_stance():
    classifier = GaitClassifier(contact_on=5.0, contact_off=2.0)
    results = _feed(classifier, _touchdowns(IDEAL_PHASES["Pronk"]), chatter=True)

    assert results[-1].label == "Pronk"
    assert results[-1].valid
    assert all(
        result.cycle_period_s == pytest.approx(1.0)
        for result in results
        if result.valid
    )


def test_reset_discards_previous_gait_and_allows_clock_restart():
    classifier = _classifier()
    assert _feed(classifier, _touchdowns(IDEAL_PHASES["Trot"]))[-1].valid

    classifier.reset()
    result = classifier.update(0.0, [0.0] * 4)

    assert not result.valid
    assert result.relative_phases is None
    assert result.nearest_gait is None
    assert result.distance is None
    assert result.cycles_used == 0


def test_stopping_expires_the_previous_classification():
    classifier = _classifier(max_cycle_s=2.0)
    assert _feed(classifier, _touchdowns(IDEAL_PHASES["Trot"]), stop=4.8)[-1].valid

    result = classifier.update(6.6, [0.0] * 4)

    assert not result.valid
    assert result.label == "No stepping"
    assert result.relative_phases is None
    assert result.nearest_gait is None
    assert result.distance is None


@pytest.mark.parametrize("timestamp", [0.0, -1.0, math.nan, math.inf])
def test_invalid_or_nonincreasing_time_is_rejected(timestamp):
    classifier = _classifier()
    classifier.update(0.0, [0.0] * 4)

    with pytest.raises(ValueError):
        classifier.update(timestamp, [0.0] * 4)


@pytest.mark.parametrize(
    "forces",
    [
        [0.0] * 3,
        [0.0] * 5,
        [-1.0, 0.0, 0.0, 0.0],
        [math.nan, 0.0, 0.0, 0.0],
        [math.inf, 0.0, 0.0, 0.0],
    ],
)
def test_invalid_force_measurements_are_rejected(forces):
    with pytest.raises(ValueError):
        _classifier().update(0.0, forces)
