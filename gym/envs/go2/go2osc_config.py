import math

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2OscCfg(Go2Cfg):

    class env(Go2Cfg.env):
        num_envs = 64
        episode_length_s = 8

    class init_state(Go2Cfg.init_state):
        reset_mode = "reset_to_basic"

    class control(Go2Cfg.control):
        ctrl_frequency = 50
        desired_sim_frequency = 200

    class osc:
        omega = 2.0
        sigma = 1.0
        xi = 0.0
        amplitude = math.pi / 4

    class commands(Go2Cfg.commands):
        class ranges(Go2Cfg.commands.ranges):
            lin_vel_x = [0.3, 0.7]
            lin_vel_y = 0.0
            yaw_vel = 0.0

    class scaling(Go2Cfg.scaling):
        dof_pos_target = 4 * [0.2, 0.3, 0.3]

        

class Go2OscRunnerCfg(Go2RunnerCfg):

    class actor(Go2RunnerCfg.actor):
        obs = Go2RunnerCfg.actor.obs + [
            "phase_obs",
            "phase_frequency",
            "measured_grf",
            "target_grf",
        ]

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + [
            "phase_obs",
            "phase_frequency",
            "measured_grf",
            "target_grf",
        ]

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                force_tracking = 4.0

    class algorithm(Go2RunnerCfg.algorithm):
        rollout_size = 16384
        batch_size = 2048
        max_gradient_steps = 8

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2osc"