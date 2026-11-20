import torch

from gym.envs.go2.go2_se import Go2SE


class Go2SEJump(Go2SE):
    """Go2SE with a jump command (project Step 3: landing on slippery ground).

    A jump is a fixed-length window that starts when triggered: at random
    during training, by the Up arrow in play.py. The policy observes

        jump_command = [active, time remaining in the window / duration]

    During the window, three jump rewards are paid (take-off velocity, all
    feet off the ground, height gained while airborne) and the penalties that
    resist an explosive motion are switched off. Landing is judged by the
    normal walking rewards, which is where velocity estimation matters.
    """

    # * penalties that fight a jump; off while the jump window is open
    _GATED = (
        "air_time",
        "foot_clearance",
        "joint_torque",
        "joint_pos",
        "joint_speed",
        "joint_acc",
        "action_smoothness1",
        "action_smoothness2",
        "base_motion",
    )

    def _init_buffers(self):
        super()._init_buffers()
        # * time since the last jump started; large = no jump yet
        self.jump_timer = torch.full((self.num_envs,), 100.0, device=self.device)
        self.jump_start_height = torch.zeros(self.num_envs, device=self.device)
        self.jump_active = torch.zeros(self.num_envs, device=self.device)
        self.jump_command = torch.zeros(self.num_envs, 2, device=self.device)
        # * play.py's teleop turns this off and triggers jumps with Up
        self.random_jumps = True

    def trigger_jump(self, mask=None):
        """Start a jump in the envs in mask (all envs if None) not already jumping."""
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        mask = mask & (self.jump_timer >= self.cfg.jump.duration)
        self.jump_timer.masked_fill_(mask, 0.0)
        self.jump_start_height[:] = torch.where(
            mask, self.root_states[:, 2], self.jump_start_height
        )
        self._update_jump_command()

    def _update_jump_command(self):
        T = self.cfg.jump.duration
        active = self.jump_timer < T
        self.jump_active[:] = active.float()
        self.jump_command[:, 0] = self.jump_active
        self.jump_command[:, 1] = torch.where(
            active, 1.0 - self.jump_timer / T, torch.zeros_like(self.jump_timer)
        )

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.jump_timer += self.dt
        if self.random_jumps:
            j = self.cfg.jump
            ready = self.jump_timer >= j.duration + j.cooldown
            start = ready & (
                torch.rand(self.num_envs, device=self.device) < self.dt / j.mean_interval
            )
            self.trigger_jump(start)
        self._update_jump_command()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self.jump_timer.masked_fill_(reset_mask, 100.0)
        self._update_jump_command()

    # * ------------------------------------------------------------------
    # * jump rewards (positive, paid only inside the jump window)
    # * ------------------------------------------------------------------

    def _airborne(self):
        return (self.foot_contact.sum(dim=1) == 0).float()

    def _reward_jump_takeoff(self):
        # * upward (world-frame) base velocity in the first half of the window
        first_half = (self.jump_timer < 0.5 * self.cfg.jump.duration).float()
        vz = self.root_states[:, 9].clip(0.0, self.cfg.jump.max_takeoff_vel)
        return self.jump_active * first_half * vz

    def _reward_jump_flight(self):
        return self.jump_active * self._airborne()

    def _reward_jump_height(self):
        gain = self.root_states[:, 2] - self.jump_start_height
        gain = gain.clip(0.0, self.cfg.jump.max_height)
        return self.jump_active * self._airborne() * gain


def _gate(name):
    def reward(self):
        return getattr(Go2SE, f"_reward_{name}")(self) * (1.0 - self.jump_active)

    reward.__name__ = f"_reward_{name}"
    return reward


for _name in Go2SEJump._GATED:
    setattr(Go2SEJump, f"_reward_{_name}", _gate(_name))
