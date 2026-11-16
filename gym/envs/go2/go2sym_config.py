"""Go2 flat-ground locomotion for the left-right symmetry comparison.

Same environment, rewards, observations and network as the ``go2`` task,
except that dof_pos_target is clipped (see Go2SymCfg.control); the three
methods share this task and differ only in the algorithm. Pick the method on
the command line:

    uv run --frozen scripts/train.py --task=go2sym --device=cuda:0 --headless \\
        --symmetry_mode={none,augment,mirror_loss} [--mirror_loss_coef=1.0] \\
        --seed=0 --run_name=<mode>_s0
"""

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2SymCfg(Go2Cfg):
    class control(Go2Cfg.control):
        # Clip dof_pos_target to the interval that already saturates the PD
        # torque from anywhere in the joint range (Go2.dof_pos_target_bounds).
        # Without it, unbounded policy outputs feed back through the
        # dof_pos_target observation and diverged two 3000-iteration
        # mirror-loss runs (value loss -> inf, then NaN).
        clip_dof_pos_target = True
        # Per-episode action latency of 0-2 control steps (0-20 ms): the
        # 9 policies trained without it lost survival and tracking with
        # 1-2 steps of delay in evaluate_sim2real.py.
        action_latency_steps_range = [0, 2]


class Go2SymRunnerCfg(Go2RunnerCfg):
    class algorithm(Go2RunnerCfg.algorithm):
        symmetry_mode = "none"  # "none" | "augment" | "mirror_loss"
        mirror_loss_coef = 1.0  # used only by "mirror_loss"

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2sym"
        algorithm_class_name = "PPO2Sym"
