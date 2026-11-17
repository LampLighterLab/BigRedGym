from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2StandCfg(Go2Cfg):
    class env(Go2Cfg.env):
        episode_length_s = 6  # long enough that it must hold the pose

    class commands(Go2Cfg.commands):
        class ranges(Go2Cfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = 0.0
            yaw_vel = 0.0

    class asset(Go2Cfg.asset):
        # Only the feet may touch the ground (substring match on body names).
        terminate_after_contacts_on = ["base", "Head", "hip", "thigh", "calf"]

    class reward_settings(Go2Cfg.reward_settings):
        # About 80 deg nose-up: near upright, with margin before tipping back.
        gravity_target = [-1.0, 0.0, -0.18]
        gravity_sigma = 1.0
        max_pitch_deg = 110.0  # terminate once tipped this far back
        hind_feet_names = ["RL_foot", "RR_foot"]
        com_line_sigma = 0.1  # [m]
        front_feet_names = ["FL_foot", "FR_foot"]
        front_feet_height_target = 0.15  # [m]
        front_feet_height_sigma = 0.1  # [m]
        # Four legs reach at most ~0.40 m; balanced upright (90 deg) ~0.60 m.
        stand_height_target = 0.55  # [m]
        stand_height_sigma = 0.12  # [m]
        ang_acc_scale = 100.0  # [rad/s^2]


class Go2StandRunnerCfg(Go2RunnerCfg):
    class critic(Go2RunnerCfg.critic):
        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                gravity_target = 3.0
                com_over_hind_feet = 3.0
                front_feet_height = 3.0
                stand_height = 3.0
                base_ang_acc = 0.05
                orientation = 0.0  # level-body reward fights rearing
                min_base_height = 0.0  # replaced by stand_height

            class termination_weight:
                termination = 100.0  # scaled by dt: -1 per fall

    class algorithm(Go2RunnerCfg.algorithm):
        discount_horizon = 2.0  # [s] gamma ~0.995, so falls later still count

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2stand"
