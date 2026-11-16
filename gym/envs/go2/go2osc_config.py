"""ORC(111) configuration adapted to Go2 and BigRedGym's PPO2 runner"""

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2OscCfg(Go2Cfg):
    class env(Go2Cfg.env):
        num_envs = 4096
        episode_length_s = 4.0

    class control(Go2Cfg.control):
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 20.0}
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 0.5}
        ctrl_frequency = 100
        desired_sim_frequency = 500

    class osc:
        # Eq. (5): moving omega is 2 + (|vx| - 0.5), capped at 4 Hz.
        stop_threshold = 0.5
        omega_stop = 1.0
        omega_step = 2.0
        omega_slope = 1.0
        omega_max = 4.0
        coupling_stop = 4.0
        coupling_step = 1.0
        coupling_slope = 0.0
        coupling_max = 1.0
        offset = 1.0

        # Reference standard deviations; set all three to zero for diagnostics.
        omega_var = 0.25
        coupling_var = 0.25
        process_noise_std = 0.25

    class commands(Go2Cfg.commands):
        resampling_time = 3.0
        var = 1.0  # Forward-command standard deviation in m/s.

        class ranges(Go2Cfg.commands.ranges):
            lin_vel_x = [-3.0, -1.0, 0.0, 1.0, 3.0]  # Discrete centers.
            lin_vel_y = 1.0
            yaw_vel = 3.0

    class push_robots(Go2Cfg.push_robots):
        toggle = True
        interval_s = 10.0
        max_push_vel_xy = 0.2

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            contact_friction_range = [0.4, 1.0]
            link_mass_scale_range = None

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {}  # Keep the nominal PD gains fixed.

    class scaling(Go2Cfg.scaling):
        base_ang_vel = [0.3, 0.3, 0.1]
        # Canonical joint order: FL, FR, RL, RR; hip, thigh, calf per leg.
        dof_pos = 4 * [0.2, 0.3, 0.3]
        dof_pos_obs = dof_pos
        dof_pos_target = dof_pos


class Go2OscRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        frequency = 100
        hidden_dims = [256, 256, 128]
        activation = "elu"
        normalize_obs = False
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "oscillator_obs",
            "dof_pos_target",
        ]

        class noise:
            # The runner applies these directly; phase features stay exact.
            base_ang_vel = 0.01
            dof_pos_obs = 0.01
            dof_vel = 0.005

    class critic(Go2RunnerCfg.critic):
        hidden_dims = [256, 256, 128]
        activation = "elu"
        normalize_obs = False
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "oscillator_obs",
            "oscillators_vel",
            "dof_pos_target",
        ]

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                ang_vel_xy = 0.0
                torques = 12 * 5.0e-7  # Local torque reward uses a joint mean.
                # Keep Go2 smoothing: its formulas differ from ORCAgym's.
                action_rate = 0.1
                action_rate2 = 0.01
                swing_grf = 5.0
                stance_grf = 5.0
                cursorial = 0.25

            class termination_weight(Go2RunnerCfg.critic.reward.termination_weight):
                termination = 0.15

    class algorithm(Go2RunnerCfg.algorithm):
        # Reference collection size; four passes of eight minibatches per update.
        rollout_size = 4096 * 32
        batch_size = rollout_size // 8
        max_gradient_steps = 4 * 8
        clip_param = 0.2
        entropy_coef = 0.02
        learning_rate = 1.0e-5
        schedule = "adaptive"
        desired_kl = 0.01
        lr_range = [1.0e-5, 1.0e-2]
        lr_ratio = 1.5
        use_clipped_value_loss = True
        max_grad_norm = 1.0
        discount_horizon = 1.0
        GAE_bootstrap_horizon = 2.0

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2_osc"
        run_name = "ORC_111"
        max_iterations = 2000
        algorithm_class_name = "PPO2"
