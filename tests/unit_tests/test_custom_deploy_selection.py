"""Custom adapter selection and settings, without SDK or robot connections."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _selected_process(code, implementation=None):
    env = dict(os.environ)
    env.pop("GO2_CUSTOM_IMPLEMENTATION", None)
    if implementation is not None:
        env["GO2_CUSTOM_IMPLEMENTATION"] = implementation
    return subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_default_adapter_preserves_go2trot_settings():
    result = _selected_process("""
        from go2_deploy.deploy_config import DeployConfig
        from go2_deploy.utility.import_custom_impl import custom_impl

        cfg = DeployConfig()
        assert custom_impl.__name__ == "go2trot_base"
        assert cfg.task_name == "go2trot"
        assert cfg.kp == 30.0 and cfg.kd == 2.0
        assert cfg.exp_moving_avg is True
        assert cfg.phase_frequency == 2.0
        assert cfg.load_run is None and cfg.checkpoint == -1
        assert cfg.obs_dim() == 48
        other = DeployConfig()
        cfg.default_dof_pos[0] = 123.0
        cfg.command_limits["lin_vel_x"] = 123.0
        assert other.default_dof_pos[0] == 0.0
        assert other.command_limits["lin_vel_x"] == 2.0
    """)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["pd_emergent", "pd_emergent.py"])
def test_selected_adapter_provides_effective_controller_settings(name):
    result = _selected_process("""
        import torch
        from go2_deploy.deploy_config import DeployConfig
        from go2_deploy.utility.import_custom_impl import custom_impl

        cfg = DeployConfig()
        assert custom_impl.__name__ == "pd_emergent"
        assert cfg.task_name == "go2_osc"
        assert cfg.ctrl_freq == 100
        assert cfg.kp == 20.0 and cfg.kd == 0.5
        assert cfg.exp_moving_avg is False
        assert cfg.obs_dim() == 53
        assert cfg.obs_vector[-2:] == ["oscillator_obs", "dof_pos_target"]
        assert cfg.load_run == (
            "Nov16_06-26-08_ORC111_forward4096_reference_smoothness"
        )
        assert cfg.checkpoint == 2000
        torch.testing.assert_close(
            cfg.default_dof_pos, torch.tensor([0.0, 0.66, -1.36] * 4)
        )
    """, name)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["../pd_emergent.py", "a/b.py", "a\\b.py"])
def test_adapter_selection_rejects_paths(name):
    result = _selected_process(
        "from go2_deploy.utility.import_custom_impl import custom_impl", name
    )
    assert result.returncode != 0
    assert "must be a filename" in result.stderr


@pytest.mark.parametrize("explicit_run", [None, "explicit_run"])
def test_pc_loader_uses_selected_run_and_checkpoint(monkeypatch, explicit_run):
    from go2_deploy import rl_controller

    runner_cfg = SimpleNamespace(load_run="from_saved_cfg")
    env_cfg = SimpleNamespace(env=SimpleNamespace())
    train_cfg = SimpleNamespace(
        runner=runner_cfg, logging=SimpleNamespace(enable_local_saving=True)
    )
    calls = {}
    actor = object()
    runner = SimpleNamespace(
        alg=SimpleNamespace(actor=actor), switch_to_eval=lambda: None
    )

    def get_cfgs(**kwargs):
        calls.update(kwargs)
        return env_cfg, train_cfg

    registry = rl_controller.task_registry
    monkeypatch.setattr(registry, "get_cfgs", get_cfgs)
    monkeypatch.setattr(registry, "convert_frequencies_to_params", lambda *args: None)
    monkeypatch.setattr(registry, "set_log_dir_name", lambda *args: None)
    monkeypatch.setattr(registry, "make_env", lambda **kwargs: object())
    monkeypatch.setattr(registry, "make_alg_runner", lambda *args: runner)
    monkeypatch.setattr(rl_controller, "set_seed", lambda *args: None)
    cfg = SimpleNamespace(
        task_name="go2_osc", load_run="pinned_run", checkpoint=2000
    )

    assert rl_controller.setup_actor(explicit_run, cfg) is actor
    expected_run = explicit_run or "pinned_run"
    assert calls == {
        "name": "go2_osc", "original_cfg": True, "load_run": expected_run
    }
    assert runner_cfg.load_run == expected_run
    assert runner_cfg.checkpoint == 2000
