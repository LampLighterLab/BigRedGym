from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg

GO2_LEGS = ("FL", "FR", "RL", "RR")


class Go2HeightCfg(Go2TrotCfg):
    class commands(Go2TrotCfg.commands):
        class ranges(Go2TrotCfg.commands.ranges):
            base_height = [0.15, 0.38]

    class asset(Go2TrotCfg.asset):
        class robot_layout(Go2TrotCfg.asset.robot_layout):
            dof_groups = {
                "thighs": [f"{leg}_thigh_joint" for leg in GO2_LEGS],
                "calves": [f"{leg}_calf_joint" for leg in GO2_LEGS],
            }

    class reward_settings(Go2TrotCfg.reward_settings):
        base_height_tracking_scale = 0.1
        thigh_length = 0.213
        calf_length = 0.213
        foot_radius = 0.022

    class scaling(Go2TrotCfg.scaling):
        base_height_command = Go2TrotCfg.scaling.base_height


class Go2HeightRunnerCfg(Go2TrotRunnerCfg):
    class actor(Go2TrotRunnerCfg.actor):
        obs = Go2TrotRunnerCfg.actor.obs + ["base_height_command"]

    class critic(Go2TrotRunnerCfg.critic):
        obs = Go2TrotRunnerCfg.critic.obs + ["base_height_command"]

        class reward(Go2TrotRunnerCfg.critic.reward):
            class weights(Go2TrotRunnerCfg.critic.reward.weights):
                tracking_base_height = 4.0
                min_base_height = 0.0

    class runner(Go2TrotRunnerCfg.runner):
        experiment_name = "go2height"
