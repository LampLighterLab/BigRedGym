from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg


class Go2PayloadCfg(Go2TrotCfg):
    class commands(Go2TrotCfg.commands):
        class ranges(Go2TrotCfg.commands.ranges):
            # Fixed height: the actor doesn't observe it, matching deploy.
            base_height = [0.30, 0.30]

    class domain_randomization(Go2TrotCfg.domain_randomization):
        class payload(Go2TrotCfg.domain_randomization.payload):
            mass_range = [0.0, 5.0]
            # Base collision box top is z=0.057; box center sits half its
            # height above that. Re-measure against the hardware mount.
            mount_pos = [0.0, 0.0, 0.097]
            com_offset_range = [[-0.05, 0.05], [-0.02, 0.02], [-0.01, 0.02]]
            size = [0.2, 0.12, 0.08]

    class scaling(Go2TrotCfg.scaling):
        # [mass kg, COM offset x, y, z m], matching domain_randomization.payload.
        payload_obs = [5.0, 0.05, 0.02, 0.02]


class Go2PayloadRunnerCfg(Go2TrotRunnerCfg):
    class actor(Go2TrotRunnerCfg.actor):
        # Matches the deploy obs_vector: no height target, no payload sensor.
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
        ]

    class critic(Go2TrotRunnerCfg.critic):
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
            "payload_obs",
        ]

    class runner(Go2TrotRunnerCfg.runner):
        experiment_name = "go2payload"
