"""PPO2 with left-right symmetry (Mittal et al., ICRA 2024).

symmetry_mode:
- "none":        plain PPO2; the symmetry error is still measured for logging.
- "augment":     every minibatch is doubled with its mirror image
                 (M_o s, M_a a). Advantages, value targets and old
                 log-probabilities are copied from the original sample, as in
                 PPO+aug of Mittal et al. / rsl_rl.
- "mirror_loss": adds mirror_loss_coef * MSE(mu(M_o s), M_a mu(s).detach())
                 to the actor loss; the PPO surrogate uses original data only.

The runner must call ``set_mirrors`` before ``update``.
"""

import torch
import torch.nn as nn

from learning.utils import create_uniform_generator
from .ppo2 import PPO2

SYMMETRY_MODES = ("none", "augment", "mirror_loss")


class PPO2Sym(PPO2):
    def __init__(
        self,
        actor,
        critic,
        symmetry_mode="none",
        mirror_loss_coef=0.0,
        **kwargs,
    ):
        super().__init__(actor, critic, **kwargs)
        if symmetry_mode not in SYMMETRY_MODES:
            raise ValueError(f"symmetry_mode must be one of {SYMMETRY_MODES}")
        if symmetry_mode == "mirror_loss" and mirror_loss_coef <= 0.0:
            raise ValueError("mirror_loss mode needs mirror_loss_coef > 0")
        self.symmetry_mode = symmetry_mode
        self.mirror_loss_coef = mirror_loss_coef
        self.mirror_actor_obs = None
        self.mirror_critic_obs = None
        self.mirror_actions = None
        self.mean_symmetry_error = 0.0
        self.mean_mirror_loss = 0.0
        self.skipped_updates = 0  # non-finite losses skipped, cumulative

    def _finite_or_skip(self, loss):
        """True if loss is finite; otherwise count it and skip the step."""
        if torch.isfinite(loss):
            return True
        self.skipped_updates += 1
        return False

    def set_mirrors(self, actor_obs, critic_obs, actions):
        self.mirror_actor_obs = actor_obs
        self.mirror_critic_obs = critic_obs
        self.mirror_actions = actions

    def _require_mirrors(self):
        if self.mirror_actor_obs is None:
            raise RuntimeError("PPO2Sym.set_mirrors was not called")

    def update(self, data):
        self._require_mirrors()
        super().update(data)

    # * Critic ---------------------------------------------------------------
    def update_critic(self, data):
        if self.symmetry_mode != "augment":
            return super().update_critic(data)

        self.mean_value_loss = 0
        counter = 0
        generator = create_uniform_generator(
            data, self.batch_size, max_gradient_steps=self.max_gradient_steps
        )
        for batch in generator:
            obs = torch.cat(
                (batch["critic_obs"], self.mirror_critic_obs(batch["critic_obs"]))
            )
            old_values = torch.cat((batch["values"], batch["values"]))
            returns = torch.cat((batch["returns"], batch["returns"]))
            values = self.critic.evaluate(obs)
            if self.use_clipped_value_loss:
                value_clipped = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (values - returns).pow(2), (value_clipped - returns).pow(2)
                ).mean()
            else:
                value_loss = (values - returns).pow(2).mean()

            if not self._finite_or_skip(value_loss):
                continue
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.mean_value_loss += value_loss.item()
            counter += 1
        self.mean_value_loss /= max(counter, 1)

    # * Actor ----------------------------------------------------------------
    def update_actor(self, data):
        self.mean_surrogate_loss = 0
        self.mean_symmetry_error = 0
        self.mean_mirror_loss = 0
        counter = 0

        self.actor.update_distribution(data["actor_obs"])
        data["old_sigma_batch"] = self.actor.action_std.detach()
        data["old_mu_batch"] = self.actor.action_mean.detach()
        data["old_actions_log_prob_batch"] = self.actor.get_actions_log_prob(
            data["actions"]
        ).detach()

        generator = create_uniform_generator(
            data, self.batch_size, max_gradient_steps=self.max_gradient_steps
        )
        for batch in generator:
            n = batch["actor_obs"].shape[0]
            obs = batch["actor_obs"]
            mirrored_obs = self.mirror_actor_obs(obs)
            actions = batch["actions"]
            advantages = torch.squeeze(batch["advantages"])
            old_log_prob = torch.squeeze(batch["old_actions_log_prob_batch"])

            if self.symmetry_mode == "augment":
                obs = torch.cat((obs, mirrored_obs))
                actions = torch.cat((actions, self.mirror_actions(actions)))
                advantages = torch.cat((advantages, advantages))
                old_log_prob = torch.cat((old_log_prob, old_log_prob))

            self.actor.update_distribution(obs)
            log_prob = self.actor.get_actions_log_prob(actions)
            mu_all = self.actor.action_mean
            mu = mu_all[:n]
            sigma = self.actor.action_std[:n]
            entropy = self.actor.entropy

            # * Symmetry error ||mu(M_o s) - M_a mu(s)||^2, logged in every mode
            if self.symmetry_mode == "augment":
                mu_mirror = mu_all[n:]
            else:
                with torch.set_grad_enabled(self.symmetry_mode == "mirror_loss"):
                    mu_mirror = self.actor.act_inference(mirrored_obs)
            target = self.mirror_actions(mu).detach()
            sym_err = (mu_mirror - target).pow(2).sum(dim=-1).mean()

            # * KL-adaptive learning rate on the original samples only
            if self.desired_kl is not None and self.schedule == "adaptive":
                self._adapt_learning_rate(
                    mu, sigma, batch["old_mu_batch"], batch["old_sigma_batch"]
                )

            # * Surrogate loss
            ratio = torch.exp(log_prob - old_log_prob)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            loss = surrogate_loss - self.entropy_coef * entropy.mean()

            if self.symmetry_mode == "mirror_loss":
                mirror_loss = (mu_mirror - target).pow(2).mean()
                loss = loss + self.mirror_loss_coef * mirror_loss

            if not self._finite_or_skip(loss):
                continue
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.mean_surrogate_loss += surrogate_loss.item()
            self.mean_symmetry_error += sym_err.item()
            if self.symmetry_mode == "mirror_loss":
                self.mean_mirror_loss += mirror_loss.item()
            counter += 1

        counter = max(counter, 1)
        self.mean_surrogate_loss /= counter
        self.mean_symmetry_error /= counter
        self.mean_mirror_loss /= counter

    def _adapt_learning_rate(self, mu, sigma, old_mu, old_sigma):
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1.0e-5)
                + (torch.square(old_sigma) + torch.square(old_mu - mu))
                / (2.0 * torch.square(sigma))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            lr_min, lr_max = self.lr_range
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(lr_min, self.learning_rate / self.lr_ratio)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(lr_max, self.learning_rate * self.lr_ratio)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
