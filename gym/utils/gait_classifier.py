"""Classify footfalls over two RF cycles using the paper's distance cutoff of 2.

Relative phases are (LF, RH, LH) versus RF, in radians:
https://arxiv.org/html/2402.08662v2#S2.SS1.

Wrapped distances and circular means are implementation choices. Results update
each cycle; the paper samples every five seconds.
"""

from collections import deque
from dataclasses import dataclass, field, replace
import math
from typing import Sequence

import numpy as np


LEG_NAMES = ("RF", "LF", "RH", "LH")
GAIT_PHASES = {
    "Trot": (math.pi, math.pi, 0.0),
    "Pace": (math.pi, 0.0, math.pi),
    "Bound": (0.0, math.pi, math.pi),
    "Pronk": (0.0, 0.0, 0.0),
}
GAIT_DISTANCE_THRESHOLD = 2.0


@dataclass(frozen=True)
class GaitResult:
    """Gait label, phase errors in radians, and cycle metadata."""

    label: str
    relative_phases: tuple[float, float, float] | None = None
    nearest_gait: str | None = None
    distance: float | None = None
    distances: dict[str, float] = field(default_factory=dict)
    time_s: float | None = None
    cycle_period_s: float | None = None
    cycles_used: int = 0
    reason: str | None = None

    @property
    def valid(self) -> bool:
        """True when distances were computed."""
        return self.nearest_gait is not None


def classify_relative_phases(relative_phases: Sequence[float]) -> GaitResult:
    """Classify averaged (LF, RH, LH) phases relative to RF, in radians.

    Angles wrap at 2*pi. Distances above 2 are classified as Transition.
    """
    phases = np.asarray(relative_phases, dtype=np.float64)
    if phases.shape != (3,) or not np.isfinite(phases).all():
        raise ValueError("relative_phases must contain three finite angles")
    phases = phases % math.tau
    distances = {}
    for gait, ideal in GAIT_PHASES.items():
        difference = (phases - ideal + math.pi) % math.tau - math.pi
        distances[gait] = float(np.linalg.norm(difference))
    nearest = min(distances, key=distances.get)
    distance = distances[nearest]
    return GaitResult(
        label=nearest if distance <= GAIT_DISTANCE_THRESHOLD else "Transition",
        relative_phases=tuple(float(value) for value in phases),
        nearest_gait=nearest,
        distance=distance,
        distances=distances,
    )


class GaitClassifier:
    """Track one robot from RF, LF, RH, LH force magnitudes in newtons.

    Choose contact_on/off from measured traces. Cycle bounds reject rapid RF
    contact chatter and expire stale results. These are engineering settings.

    Sample at physics or control frequency. Three observed RF touchdowns provide
    two cycles; missing partner steps invalidate the window. Reset each episode.
    """

    def __init__(
        self,
        *,
        contact_on: float,
        contact_off: float,
        min_cycle_s: float = 0.1,
        max_cycle_s: float = 2.0,
    ):
        if not 0 <= contact_off < contact_on < math.inf:
            raise ValueError("contact cutoffs must satisfy 0 <= off < on")
        if not 0 < min_cycle_s < max_cycle_s < math.inf:
            raise ValueError("cycle bounds must satisfy 0 < min < max")
        self.contact_on = contact_on
        self.contact_off = contact_off
        self.min_cycle_s = min_cycle_s
        self.max_cycle_s = max_cycle_s
        self.reset()

    def reset(self):
        """Clear contact and cycle history."""
        self._contacts = None
        self._last_time = None
        self._start_time = None
        self._last_activity = None
        self._cycle_start = None
        self._touchdowns = [deque() for _ in range(3)]
        self._clear_window()
        self.result = GaitResult("Collecting cycles")

    def _clear_window(self):
        self._matched_touchdowns = None
        self._phase_offsets = None
        self._cycles = deque(maxlen=2)
        self._periods = deque(maxlen=2)

    def _status(self, label, time_s, reason=None):
        return GaitResult(
            label,
            time_s=time_s,
            cycle_period_s=float(np.mean(self._periods)) if self._periods else None,
            cycles_used=len(self._cycles),
            reason=reason,
        )

    def update(self, time_s: float, foot_forces: Sequence[float]) -> GaitResult:
        """Update from a sample with a strictly increasing timestamp.

        result.time_s records the last classification update. Initial contacts
        do not count as touchdowns; no stepping does not imply standing.
        """
        time_s = float(time_s)
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self._last_time is not None and time_s <= self._last_time:
            raise ValueError("time_s must increase strictly; call reset for a new run")
        forces = np.asarray(foot_forces, dtype=np.float64)
        if forces.shape != (4,) or not np.isfinite(forces).all() or (forces < 0).any():
            raise ValueError("foot_forces must be four finite, nonnegative magnitudes")
        self._last_time = time_s

        if self._contacts is None:
            self._contacts = forces >= self.contact_on
            self._start_time = self._last_activity = time_s
            self.result = self._status("Collecting cycles", time_s)
            return self.result

        contacts = np.where(
            self._contacts, forces > self.contact_off, forces >= self.contact_on
        )
        touchdowns = contacts & ~self._contacts
        self._contacts = contacts
        if touchdowns.any():
            self._last_activity = time_s

        # Record simultaneous partner touchdowns before closing the RF cycle.
        for leg, history in enumerate(self._touchdowns, start=1):
            if touchdowns[leg]:
                history.append(time_s)
            while history and history[0] < time_s - 3 * self.max_cycle_s:
                history.popleft()

        if (
            self._cycle_start is not None
            and time_s - self._cycle_start > self.max_cycle_s
        ):
            self._clear_window()
            self._cycle_start = None

        if touchdowns[0]:
            if self._cycle_start is None:
                self._cycle_start = time_s
                self.result = self._status("Collecting cycles", time_s)
            elif time_s - self._cycle_start >= self.min_cycle_s:
                self.result = self._finish_cycle(time_s)
                self._cycle_start = time_s
        elif (
            self._cycle_start is None and time_s - self._start_time >= self.max_cycle_s
        ):
            no_steps = time_s - self._last_activity >= self.max_cycle_s
            self.result = self._status(
                "No stepping" if no_steps else "Insufficient stepping data", time_s
            )
        return self.result

    def _finish_cycle(self, end_s):
        period = end_s - self._cycle_start
        matches = []
        for leg, history in enumerate(self._touchdowns):
            # Follow the previous phase, allowing touchdowns just before RF.
            offset = 0.0 if self._phase_offsets is None else self._phase_offsets[leg]
            target = self._cycle_start + offset * period
            candidates = [
                event
                for event in history
                if (
                    self._matched_touchdowns is None
                    or event > self._matched_touchdowns[leg]
                )
                and abs(event - target) <= period / 2 + 1e-9
            ]
            if not candidates:
                self._clear_window()
                return self._status(
                    "Insufficient stepping data",
                    end_s,
                    f"No matching {LEG_NAMES[leg + 1]} touchdown",
                )
            matches.append(min(candidates, key=lambda event: abs(event - target)))

        self._matched_touchdowns = np.asarray(matches)
        self._phase_offsets = (self._matched_touchdowns - self._cycle_start) / period
        self._cycles.append(self._phase_offsets * math.tau)
        self._periods.append(period)
        if len(self._cycles) < 2:
            return self._status("Collecting cycles", end_s)

        # Circular averaging keeps phases near 0 and 2*pi together.
        sine = np.sin(self._cycles).mean(axis=0)
        cosine = np.cos(self._cycles).mean(axis=0)
        if (np.hypot(sine, cosine) < 1e-6).any():
            return self._status(
                "Transition", end_s, "Two-cycle phase mean is undefined"
            )
        result = classify_relative_phases(np.arctan2(sine, cosine))
        return replace(
            result,
            time_s=end_s,
            cycle_period_s=float(np.mean(self._periods)),
            cycles_used=2,
        )
