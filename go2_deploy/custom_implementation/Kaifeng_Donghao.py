# Custom deployment for the go2sym task (Kaifeng Jia kj366, Donghao Hong dh787).
# Template: go2_deploy/custom_implementation/go2trot_base.py (fixdeploy branch).
# Every observation piece is a standard one, so no custom buffers or pieces.
# Settings this file cannot express (gains, smoothing, default posture,
# target clipping, task name) are listed in Kaifeng_Donghao_notes.md.
from gym.envs.go2.go2sym_config import Go2SymCfg


class CustomDeployConfig:
    # Same order as Go2RunnerCfg.actor.obs (go2sym actor), 45 values in total.
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
    ]

    custom_obs_sizes = {}

    # go2sym scaling (Go2Cfg.scaling). Differs from go2trot in dof_pos_target:
    # [0.5236, 1.2654, 0.4712] per leg (hip, thigh, calf) instead of [0.2, 0.3, 0.3].
    class DeployScaling(Go2SymCfg.scaling):
        pass


def get_custom_buffers_dict():
    return {}


def update_custom_buffers(main_controller, t):
    pass


def get_custom_obs_pieces():
    return {}
