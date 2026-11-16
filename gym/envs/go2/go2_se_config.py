from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2SECfg(Go2Cfg):
    class env(Go2Cfg.env):
        episode_length_s = 10
        # * joint state history sampled at t - 0.02, t - 0.04, t - 0.06 s
        history_dt = 0.02
        history_length = 3

    class init_state(Go2Cfg.init_state):
        # * Ji et al. Table I initial-state noise (orientation as roll/pitch)
        dof_pos_range = {
            "hip": [-0.2, 0.2],
            "thigh": [0.46, 0.86],
            "calf": [-1.56, -1.16],
        }
        dof_vel_range = {
            "hip": [-2.5, 2.5],
            "thigh": [-2.5, 2.5],
            "calf": [-2.5, 2.5],
        }
        root_pos_range = [
            [0.0, 0.0],  # x
            [0.0, 0.0],  # y
            [0.40, 0.40],  # z
            [-0.2, 0.2],  # roll
            [-0.2, 0.2],  # pitch
            [-3.14, 3.14],  # yaw
        ]
        root_vel_range = [
            [-1.0, 1.0],  # x
            [-0.5, 0.5],  # y
            [-0.5, 0.5],  # z
            [-0.7, 0.7],  # roll
            [-0.7, 0.7],  # pitch
            [-0.7, 0.7],  # yaw
        ]
        # * probability a timed-out env continues from its final state
        continue_prob = 0.25

    class commands(Go2Cfg.commands):
        resampling_time = 5.0
        zero_command_prob = 0.1

        class ranges(Go2Cfg.commands.ranges):
            # * full range; the training curriculum narrows lin_vel_x
            lin_vel_x = [-1.75, 3.5]  # [m/s]
            lin_vel_y = 1.0  # [m/s]
            yaw_vel = 2.0  # [rad/s]

        class curriculum:
            # * eq. (1): v_max = v_start + v_gain / (1 + exp(-rate (it - mid)))
            # * paper: midpoint 1000, rate 0.002 over 2500 iterations of
            # * ~320k samples. Ours are 2**15 samples, so we compress to 1500
            # * iterations keeping the same shape (rate * midpoint = 2)
            toggle = True
            v_start = 1.0
            v_gain = 2.5
            rate = 0.0033
            midpoint = 600
            backward_ratio = 0.5

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            contact_friction_range = [0.4, 1.0]  # paper
            link_mass_scale_range = [0.9, 1.1]

        class episode(Go2Cfg.domain_randomization.episode):
            # * paper: Kp +-2 of 17, Kd +-0.1 of 0.4 (~+-12%, +-25%)
            scale_ranges = {
                "p_gains": [0.88, 1.12],
                "d_gains": [0.75, 1.25],
            }

    class asset(Go2Cfg.asset):
        foot_radius = 0.022  # [m] collision sphere radius in go2.urdf
        # * paper: episode ends on any body contact except knees and feet
        terminate_after_contacts_on = ["base", "Head", "hip", "thigh"]

    class reward_settings(Go2Cfg.reward_settings):
        contact_force_threshold = 1.0  # [N] foot counts as in contact above this
        foot_clearance_target = 0.09  # [m] paper's desired swing foot height
        # * paper uses 1.0; with our ~100x smaller sample budget the policy
        # * settled on standing still, so we use the Go2 default sharpness
        lin_vel_tracking_sigma = 0.25

    class scaling(Go2Cfg.scaling):
        dof_pos_error_history = 3 * Go2Cfg.scaling.dof_pos_target
        dof_vel_history = 3 * Go2Cfg.scaling.dof_vel


class Go2SEPaperCfg(Go2SECfg):
    """Env settings for the paper-reward task (go2_se_paper)."""

    class reward_settings(Go2SECfg.reward_settings):
        lin_vel_tracking_sigma = 1.0  # paper: exp(-||cmd - v||^2)

    class scaling(Go2SECfg.scaling):
        # * paper: q_des = q_nominal + 0.1 * a. The Go2 default (0.47-1.27
        # * rad per unit) made exploration noise ~6x costlier under the
        # * paper's action-smoothness penalties, and the policy froze.
        dof_pos_target = 12 * [0.1]


# Proprioceptive observations, all available on hardware.
PROPRIO_OBS = [
    "base_ang_vel",
    "projected_gravity",
    "dof_pos_obs",
    "dof_vel",
    "dof_pos_history",  # last three joint targets (action history)
    "dof_pos_error_history",
    "dof_vel_history",
    "foot_pos_base",
]
SE_TARGETS = ["base_lin_vel", "foot_height", "foot_contact"]


class Go2SERunnerCfg(Go2RunnerCfg):
    runner_class_name = "EstimatorRunner"

    class actor(Go2RunnerCfg.actor):
        hidden_dims = [512, 256, 64]
        # * the estimator's outputs are appended to these by the runner
        obs = PROPRIO_OBS + ["commands"]
        normalize_obs = False
        smooth_exploration = False
        add_noise = True

        class noise(Go2RunnerCfg.actor.noise):
            # * gaussian std in physical units (divided by obs scale);
            # * the paper's larger uniform noise is left for Step 2 (DR)
            projected_gravity = 0.03
            dof_pos_history = 0.01
            dof_pos_error_history = 0.01
            dof_vel_history = 0.005
            foot_pos_base = 0.01  # [m]

    class critic(Go2RunnerCfg.critic):
        hidden_dims = [512, 256, 64]
        # * the critic is privileged: it sees the true estimator targets
        obs = (
            PROPRIO_OBS
            + ["commands", "base_height"]
            + SE_TARGETS
            + ["feet_air_time", "feet_stance_time"]
        )
        normalize_obs = False
        # * rewards: Go2 defaults (additive), inherited from Go2RunnerCfg.
        # * The paper's Table II rewards are in Go2SEPaperRunnerCfg; with our
        # * sample budget they converged to standing still.

    class state_estimator:
        obs = PROPRIO_OBS
        targets = SE_TARGETS
        # * trained with binary cross-entropy, sigmoid applied at prediction
        bce_targets = ["foot_contact"]
        hidden_dims = [256, 128]
        activation = "elu"
        learning_rate = 1.0e-3
        batch_size = 2**15
        max_gradient_steps = 10
        # * True: actor gets ground-truth targets instead of estimates
        # * (privileged baseline for comparing against the estimator)
        oracle = False

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2_se"
        max_iterations = 1500


class Go2SEOracleRunnerCfg(Go2SERunnerCfg):
    class state_estimator(Go2SERunnerCfg.state_estimator):
        oracle = True

    class runner(Go2SERunnerCfg.runner):
        experiment_name = "go2_se_oracle"


class Go2SEPaperRunnerCfg(Go2SERunnerCfg):
    """Ji et al. Table II rewards, r_tot = r_pos * exp(0.2 * r_neg).
    With ~2**15 samples/iteration this converged to standing still twice
    (paper uses ~320k samples/iteration); kept for Step 2."""

    class actor(Go2SERunnerCfg.actor):
        # * with the paper's 0.1 action scale this is ~0.1 rad of noise
        init_noise_std = 1.0

    class critic(Go2SERunnerCfg.critic):
        class reward:
            # * Ji et al. Table II; r_tot = r_pos * exp(neg_scale * r_neg)
            neg_scale = 0.2

            class weights:  # r_pos terms
                lin_vel_exp = 3.0
                ang_vel_exp = 3.0
                air_time = 0.3

            class negative_weights:  # r_neg terms (paper coefficients)
                foot_slip = -0.08
                foot_clearance = -15.0
                orientation_angle = -3.0
                joint_torque = -6e-4
                joint_pos = -0.75
                joint_speed = -6e-4
                joint_acc = -0.02
                action_smoothness1 = -2.5
                action_smoothness2 = -1.2
                base_motion = -1.5

            class termination_weight:
                # * one-off -10 (relative to per-step rewards ~6)
                termination = 10.0

    class runner(Go2SERunnerCfg.runner):
        experiment_name = "go2_se_paper"
