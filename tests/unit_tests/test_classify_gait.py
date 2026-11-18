"""Check playback sampling and classifier history across episode boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gym.utils.gait_classifier import GaitResult
from scripts import classify_gait as driver


class FakeEnv:
    def __init__(self, on_step=None, viewer=None):
        names = ("RL_foot", "FL_foot", "trunk", "FR_foot", "RR_foot")
        self.robot_layout = SimpleNamespace(
            body_indices=lambda requested: [names.index(name) for name in requested]
        )
        self.device = "cpu"
        self.dt = 0.1
        self.common_step_counter = 0
        self.episode_length_buf = torch.tensor([10])
        self.terminated = torch.tensor([False])
        self.timed_out = torch.tensor([False])
        self.exit = False
        self.commands = torch.tensor([[1.0, 0.0, 0.0]])
        self.contact_forces = torch.zeros(1, len(names), 3)
        self.contact_forces[0, 3] = torch.tensor([3.0, 4.0, 0.0])
        self.contact_forces[0, [1, 4, 0], 2] = torch.tensor([12.0, 13.0, 14.0])
        self.grf = torch.full((1, 4), 0.25)
        self._backend = SimpleNamespace(
            _viewer=viewer, _viewer_overlay_fn=object(), close=lambda: None
        )
        self.on_step = on_step

    def step(self):
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.terminated.zero_()
        self.timed_out.zero_()
        if self.on_step:
            self.on_step(self)


class FakeRunner:
    actor_cfg = {"actions": ["target"], "disable_actions": False}

    def __init__(self, env):
        self.env = env
        self.action_calls = 0

    def get_inference_actions(self):
        return torch.zeros(1, 12)

    def set_actions(self, *_args):
        self.action_calls += 1

    def reset_envs(self):
        if bool(self.env.terminated[0] | self.env.timed_out[0]):
            self.env.episode_length_buf.zero_()


class RecordingClassifier:
    def __init__(self, **kwargs):
        self.cutoffs = kwargs
        self.updates = []
        self.resets = 0
        self.samples = 0

    def reset(self):
        self.resets += 1
        self.samples = 0

    def update(self, time_s, forces):
        self.updates.append((time_s, forces.copy(), self.resets))
        self.samples += 1
        if self.samples < 2:
            return GaitResult("Collecting cycles", time_s=time_s)
        return GaitResult(
            "Trot",
            relative_phases=(np.pi, np.pi, 0.0),
            nearest_gait="Trot",
            distance=0.0,
            time_s=time_s,
        )


@pytest.fixture
def classifier(monkeypatch):
    instance = RecordingClassifier(contact_on=5.0, contact_off=2.0)
    monkeypatch.setattr(driver, "GaitClassifier", lambda **kwargs: instance)
    return instance


def _run(env, duration=0.4):
    args = SimpleNamespace(contact_on=5.0, contact_off=2.0, duration=duration)
    runner = FakeRunner(env)
    return driver.classify(env, runner, args), runner


def test_sampling_uses_named_feet_and_force_magnitudes(classifier):
    result, runner = _run(FakeEnv(), duration=0.3)

    assert result.valid
    assert runner.action_calls == 3
    assert [row[0] for row in classifier.updates] == pytest.approx([0, 0.1, 0.2, 0.3])
    for _, forces, _ in classifier.updates:
        np.testing.assert_allclose(forces, [5.0, 12.0, 13.0, 14.0])


def test_counter_rewind_clears_manual_reset_history(classifier):
    def reset_at_second_step(env):
        if env.common_step_counter == 2:
            env.episode_length_buf.zero_()

    _run(FakeEnv(on_step=reset_at_second_step))

    assert classifier.resets == 1
    assert [row[2] for row in classifier.updates] == [0, 0, 1, 1, 1]


@pytest.mark.parametrize("flag", ["terminated", "timed_out"])
def test_episode_end_skips_failed_contacts_and_clears_history(classifier, flag, capsys):
    def end_second_step(env):
        if env.common_step_counter == 2:
            getattr(env, flag).fill_(True)

    _run(FakeEnv(on_step=end_second_step))

    assert classifier.resets == 1
    assert [row[0] for row in classifier.updates] == pytest.approx([0, 0.1, 0.3, 0.4])
    assert [row[2] for row in classifier.updates] == [0, 0, 1, 1]
    assert "Gait: Resetting" in capsys.readouterr().out


def test_closed_viewer_stops_without_replacing_overlay(classifier):
    texts = []
    viewer = SimpleNamespace(is_running=lambda: False, set_texts=texts.append)
    env = FakeEnv(viewer=viewer)
    overlay = env._backend._viewer_overlay_fn

    _, runner = _run(env)

    assert runner.action_calls == 0
    assert env._backend._viewer_overlay_fn is overlay
    assert texts


def test_main_preserves_commands_when_keyboard_setup_changes_them(monkeypatch):
    env = FakeEnv()
    expected = env.commands.clone()
    observed = []
    monkeypatch.setattr(driver, "setup", lambda args: (env, FakeRunner(env)))
    monkeypatch.setattr(driver, "setup_viewer", lambda env, args: env.commands.zero_())
    monkeypatch.setattr(
        driver,
        "classify",
        lambda env, runner, args: observed.append(env.commands.clone()),
    )

    driver.main(["--task", "go2_osc", "--headless"])

    torch.testing.assert_close(observed[0], expected)


def test_cli_keeps_playback_flags():
    args = driver.get_args(
        [
            "--task",
            "go2_osc",
            "--headless",
            "--load_run",
            "saved",
            "--checkpoint",
            "2000",
        ]
    )

    assert args.num_envs == 1
    assert args.duration == 30
    assert (args.contact_on, args.contact_off) == (5, 2)
    assert args.headless
    assert args.load_run == "saved"
    assert args.checkpoint == 2000


@pytest.mark.parametrize(
    "options",
    [
        ["--duration", "nan"],
        ["--duration", "0"],
        ["--num_envs", "0"],
        ["--contact_on", "2", "--contact_off", "5"],
        ["--contact_on", "inf"],
    ],
)
def test_cli_rejects_invalid_sampling_options(options):
    with pytest.raises(SystemExit):
        driver.get_args(["--task", "go2_osc", *options])
