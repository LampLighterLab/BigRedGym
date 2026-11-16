from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2OscCfg(Go2Cfg):

    class env(Go2Cfg.env):
        num_envs = 64
        episode_length_s = 16
        termination_height = 0.20  # m; terminate if the base drops below this

    class init_state(Go2Cfg.init_state):
        reset_mode = "reset_to_range"

    class control(Go2Cfg.control):
        ctrl_frequency = 50
        desired_sim_frequency = 200

    class asset(Go2Cfg.asset):
        penalize_contacts_on = ["thigh", "calf"]

    class osc:
        frequency = 2.0  # Hz; starting frequency during playback
        frequency_range = [0.0, 5.0]  # Hz; sampled during training
        frequency_resampling_time = 2.0  # seconds between training changes
        zero_frequency_fraction = 0.1  # Include exact 0 Hz commands
        randomize_frequency = True
        frequency_step = 0.25  # Hz per Up/Down key press during playback
        # Cycle offsets stagger the legs by 90 degrees in foot order.
        phase_offsets = {
            "FL_foot": 0.0,
            "FR_foot": 0.25,
            "RL_foot": 0.5,
            "RR_foot": 0.75,
        }
        contact_threshold = 0.01  # fraction of body weight
        knee_clearance = 0.10  # m; penalize knee joints below this height

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
        ]

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + [
            "phase_obs",
            "phase_frequency",
            "measured_grf",
        ]

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                gait_contact = 4.0
                collision = 0.5
                knee_clearance = 2.0
                tracking_lin_vel = 0.0
                tracking_ang_vel = 0.0
                min_base_height = 0.0

    class algorithm(Go2RunnerCfg.algorithm):
        rollout_size = 16384
        batch_size = 2048
        max_gradient_steps = 8

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2osc"
