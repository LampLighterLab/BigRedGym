import torch

from gym.envs.go2.go2osc import Go2Osc
from gym.envs.go2.go2osc_config import Go2OscCfg
from gym.envs.go2.go2_config import Go2RunnerCfg
from gym.utils.task_registry import task_registry, select_backend


def test_oscillator_integration():

    cfg = Go2OscCfg()
    cfg.env.num_envs = 2
    cfg.seed = 1
    cfg.init_state.reset_mode = "reset_to_basic"

    task_registry.convert_frequencies_to_params(cfg, Go2RunnerCfg())

    backend = select_backend(cfg, "cpu")

    env = Go2Osc(cfg, "cpu", True, backend)

    try:
        print("Initial phases:", env.phase)

        for _ in range(50):
            env.step()

        print("Final phases:", env.phase)
        print("Measured forces:", env.measured_grf)
        print("Desired forces:", env.target_grf)
        print("Frequencies:", env.phase_frequency)

        assert env.phase.shape == (2, 4)
        assert torch.isfinite(env.phase).all()
        assert torch.isfinite(env.measured_grf).all()

    finally:
        backend.close()