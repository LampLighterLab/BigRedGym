import torch

from gym.utils.sampling import masked_update, torch_rand_float

from gym.envs.go2.go2trot import Go2Trot


class Go2Height(Go2Trot):
    """Go2Trot that also tracks a commanded base height."""

    def _init_buffers(self):
        super()._init_buffers()
        self.base_height_command = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self._thigh_actuators = self._actuator_indices("thighs")
        self._calf_actuators = self._actuator_indices("calves")
        default_pos = self.default_dof_pos[0].index_select(0, self.actuated_dof_indices)
        thigh = default_pos[self._thigh_actuators]
        calf = default_pos[self._calf_actuators]
        l1 = self.cfg.reward_settings.thigh_length
        l2 = self.cfg.reward_settings.calf_length
        self._nominal_foot_x = -l1 * torch.sin(thigh) - l2 * torch.sin(thigh + calf)
        self._validate_height_range()

    def _actuator_indices(self, group_name):
        return torch.tensor(
            [
                self.actuated_dof_names.index(name)
                for name in self.robot_layout.dof_groups[group_name]
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _validate_height_range(self):
        low, high = self.command_ranges["base_height"]
        if not 0.0 < low < high:
            raise ValueError(f"invalid base_height command range {[low, high]}")
        heights = torch.tensor([[low], [high]], device=self.device)
        posture = self._leg_posture(heights)
        limits = self._actuated_dof_pos_limits
        if not torch.all((posture >= limits[:, 0]) & (posture <= limits[:, 1])):
            raise ValueError(
                f"base_height command range {[low, high]} needs leg postures "
                "outside the reachable workspace or joint limits"
            )

    def _leg_posture(self, base_height):
        settings = self.cfg.reward_settings
        l1, l2 = settings.thigh_length, settings.calf_length
        depth = base_height - settings.foot_radius
        foot_x = self._nominal_foot_x
        reach_sq = foot_x.square() + depth.square()
        calf = -torch.acos((reach_sq - l1**2 - l2**2) / (2 * l1 * l2))
        thigh = torch.atan2(-foot_x, depth) - torch.atan2(
            l2 * torch.sin(calf), l1 + l2 * torch.cos(calf)
        )
        posture = (
            self.default_dof_pos.index_select(1, self.actuated_dof_indices)
            .expand(base_height.shape[0], -1)
            .clone()
        )
        posture[:, self._thigh_actuators] = thigh
        posture[:, self._calf_actuators] = calf
        return posture

    def _resample_commands(self, command_mask):
        super()._resample_commands(command_mask)
        low, high = self.command_ranges["base_height"]
        height = torch_rand_float(
            low, high, shape=self.base_height_command.shape, device=self.device
        )
        masked_update(self.base_height_command, height, command_mask)

    def _reward_tracking_base_height(self):
        error = (self.base_height - self.base_height_command).flatten()
        error = error / self.cfg.reward_settings.base_height_tracking_scale
        return self._sqrdexp(error) - 0.2 * error.abs()

    def _reward_reference_traj(self):
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        reference = self._leg_posture(self.base_height_command) + self.gait_reference
        scale = self.scales["dof_pos"].index_select(0, self.actuated_dof_indices)
        error = (reference - pos) / scale
        return (self._sqrdexp(error) - 0.2 * error.abs()).mean(dim=1)
