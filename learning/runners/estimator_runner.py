import os
import torch
from tensordict import TensorDict

from learning.algorithms import StateEstimator, get_algorithm_class
from learning.modules import Actor, Critic, SmoothActor, StateEstimatorNN

# * share the logger and storage instances with the parent runner's methods
from .on_policy_runner import OnPolicyRunner, logger, storage


class EstimatorRunner(OnPolicyRunner):
    """PPO with a concurrently trained, explicit state estimator (Ji et al. 2022).

    Every policy step the estimator predicts x_hat (e.g. base velocity, foot
    height, contact probability) from proprioception; the actor sees its own
    observations concatenated with x_hat, and the critic sees the true states.
    After each PPO update the estimator is regressed onto the true states
    collected during the same rollout.

    With state_estimator.oracle = True the actor gets the true states instead,
    which gives the privileged baseline to compare the estimator against.
    """

    def parse_train_cfg(self, train_cfg):
        super().parse_train_cfg(train_cfg)
        self.se_cfg = train_cfg["state_estimator"]
        self.oracle = self.se_cfg["oracle"]
        # * the actor's obs start with the estimator's inputs so that both
        # * networks see the *same* noisy sensor reading each step
        n_se = len(self.se_cfg["obs"])
        assert self.actor_cfg["obs"][:n_se] == self.se_cfg["obs"], (
            "actor.obs must start with state_estimator.obs"
        )
        self.actor_extra_obs = self.actor_cfg["obs"][n_se:]

    def _set_up_alg(self):
        num_se_obs = self.get_obs_size(self.se_cfg["obs"])
        num_se_targets = self.get_obs_size(self.se_cfg["targets"])
        num_actor_obs = self.get_obs_size(self.actor_cfg["obs"]) + num_se_targets
        num_actions = self.get_action_size(self.actor_cfg["actions"])
        num_critic_obs = self.get_obs_size(self.critic_cfg["obs"])

        if self.actor_cfg["smooth_exploration"]:
            actor = SmoothActor(num_actor_obs, num_actions, **self.actor_cfg)
        else:
            actor = Actor(num_actor_obs, num_actions, **self.actor_cfg)
        critic = Critic(num_critic_obs, **self.critic_cfg)
        alg_class = get_algorithm_class(self.cfg["algorithm_class_name"])
        self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)

        bce_mask = torch.cat(
            [
                torch.full(
                    (self.get_obs_size([name]),),
                    name in self.se_cfg["bce_targets"],
                    dtype=torch.bool,
                )
                for name in self.se_cfg["targets"]
            ]
        )
        estimator_nn = StateEstimatorNN(
            num_se_obs,
            num_se_targets,
            hidden_dims=self.se_cfg["hidden_dims"],
            activation=self.se_cfg["activation"],
        )
        self.se = StateEstimator(
            estimator_nn,
            learning_rate=self.se_cfg["learning_rate"],
            batch_size=self.se_cfg["batch_size"],
            max_gradient_steps=self.se_cfg["max_gradient_steps"],
            bce_mask=bce_mask,
            device=self.device,
        )

    def get_actor_inputs(self, noisy=True):
        """Return (actor_obs, se_obs, se_targets) for the current env state."""
        noise = self.actor_cfg["noise"]
        if noisy and self.actor_cfg["add_noise"]:
            se_obs = self.get_noisy_obs(self.se_cfg["obs"], noise)
            extra = self.get_noisy_obs(self.actor_extra_obs, noise)
        else:
            se_obs = self.get_obs(self.se_cfg["obs"])
            extra = self.get_obs(self.actor_extra_obs)
        se_targets = self.get_obs(self.se_cfg["targets"])
        if self.oracle:
            estimate = se_targets
        else:
            # * no policy gradient flows into the estimator
            estimate = self.se.predict(se_obs).detach()
        actor_obs = torch.cat((se_obs, extra, estimate), dim=-1)
        return actor_obs, se_obs, se_targets

    def learn(self, states_to_log_dict=None):
        n_policy_steps = int((1 / self.env.dt) / self.actor_cfg["frequency"])
        assert n_policy_steps > 0, "actor frequency should be less than ctrl_freq"
        self.set_up_logger(dt=self.env.dt * n_policy_steps)

        rewards_dict = self.initialize_rewards_dict(n_policy_steps)

        self.alg.switch_to_train()
        self.se.state_estimator.train()
        actor_obs, se_obs, se_targets = self.get_actor_inputs(noisy=False)
        critic_obs = self.get_obs(self.critic_cfg["obs"])
        tot_iter = self.it + self.num_learning_iterations
        self.save()

        # * start up storage
        transition = TensorDict({}, batch_size=self.env.num_envs, device=self.device)
        transition.update(
            {
                "actor_obs": actor_obs,
                "next_actor_obs": actor_obs,
                "actions": self.alg.act(actor_obs),
                "critic_obs": critic_obs,
                "next_critic_obs": critic_obs,
                "se_obs": se_obs,
                "se_targets": se_targets,
                "rewards": self.get_rewards({"termination": 0.0})["termination"],
                "timed_out": self.get_timed_out(),
                "terminated": self.get_terminated(),
                "dones": self.get_timed_out() | self.get_terminated(),
            }
        )
        storage.initialize(
            transition,
            self.env.num_envs,
            self.env.num_envs * self.num_steps_per_env,
            device=self.device,
        )

        logger.tic("runtime")
        for self.it in range(self.it + 1, tot_iter + 1):
            logger.tic("iteration")
            logger.tic("collection")
            if hasattr(self.env, "update_command_curriculum"):
                self.env.update_command_curriculum(self.it)

            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs)
                    self.set_actions(
                        self.actor_cfg["actions"],
                        actions,
                        self.actor_cfg["disable_actions"],
                    )

                    transition.update(
                        {
                            "actor_obs": actor_obs,
                            "actions": actions,
                            "critic_obs": critic_obs,
                            "se_obs": se_obs,
                            "se_targets": se_targets,
                        }
                    )
                    for step in range(n_policy_steps):
                        self.env.step()
                        self.update_rewards_dict(rewards_dict, step)

                    self.reset_envs()

                    total_rewards = self.compute_total_rewards(rewards_dict)

                    actor_obs, se_obs, se_targets = self.get_actor_inputs()
                    critic_obs = self.get_obs(self.critic_cfg["obs"])

                    transition.update(
                        {
                            "next_actor_obs": actor_obs,
                            "next_critic_obs": critic_obs,
                            "rewards": total_rewards,
                            "timed_out": self.env.timed_out,
                            "dones": self.env.timed_out | self.env.terminated,
                        }
                    )
                    storage.add_transitions(transition)

                    logger.log_rewards(rewards_dict.sum(dim=0))
                    logger.log_rewards({"total_rewards": total_rewards})
                    logger.finish_step(self.env.timed_out | self.env.terminated)
            logger.toc("collection")

            logger.tic("learning")
            self.alg.update(storage.data)
            self.se.update_from_data(storage.data)
            self.log_estimation_error(storage.data)
            storage.clear()
            logger.toc("learning")
            logger.log_all_categories()

            logger.finish_iteration()
            logger.toc("iteration")
            logger.toc("runtime")
            logger.print_to_terminal()

            if self.it % self.save_interval == 0:
                self.save()
        self.save()

    # * ------------------------------------------------------------------
    # * rewards: r_tot = r_pos * exp(neg_scale * r_neg) + termination
    # * (Ji et al. eq. 2). Without critic.reward.negative_weights this is
    # * the usual additive sum.
    # * ------------------------------------------------------------------

    def _negative_weights(self):
        return self.critic_cfg["reward"].get("negative_weights", {})

    def initialize_rewards_dict(self, n_steps):
        rewards_dict = super().initialize_rewards_dict(n_steps)
        for key in self._negative_weights():
            rewards_dict.update(
                {key: torch.zeros(n_steps, self.env.num_envs, device=self.device)}
            )
        return rewards_dict

    def update_rewards_dict(self, rewards_dict, step):
        super().update_rewards_dict(rewards_dict, step)
        rewards_dict[step].update(
            self.get_rewards(
                self._negative_weights(),
                modifier=self.env.dt,
                mask=~self.env.terminated,
            ),
            inplace=True,
        )

    def compute_total_rewards(self, rewards_dict):
        reward_cfg = self.critic_cfg["reward"]
        zeros = torch.zeros(rewards_dict.batch_size, device=self.device)

        def total(keys):
            return sum((rewards_dict[k] for k in keys), zeros)

        r_pos = total(reward_cfg["weights"])  # already multiplied by dt
        r_neg = total(self._negative_weights()) / self.env.dt  # per-step rate
        r_term = total(reward_cfg["termination_weight"])
        scale = reward_cfg.get("neg_scale", 0.2)
        if self._negative_weights():
            combined = r_pos * torch.exp(scale * r_neg)
        else:
            combined = r_pos
        return (combined + r_term).sum(dim=0)

    @torch.no_grad
    def log_estimation_error(self, data):
        """Mean absolute error of each estimated state over the rollout."""
        estimate = self.se.predict(data["se_obs"].flatten(0, 1))
        error = (estimate - data["se_targets"].flatten(0, 1)).abs().mean(dim=0)
        idx = 0
        for name in self.se_cfg["targets"]:
            size = self.get_obs_size([name])
            setattr(self.se, f"{name}_error", error[idx : idx + size].mean().item())
            idx += size

    def burn_in_normalization(self, n_iterations=100):
        raise NotImplementedError(
            "EstimatorRunner does not support observation normalization; "
            "set actor.normalize_obs and critic.normalize_obs to False."
        )

    def set_up_logger(self, dt=None):
        super().set_up_logger(dt)
        if self._negative_weights():
            logger.register_rewards(list(self._negative_weights().keys()))
        logger.register_category(
            "state_estimator",
            self.se,
            ["mean_mse_loss", "mean_bce_loss"]
            + [f"{name}_error" for name in self.se_cfg["targets"]],
        )
        for name in self.se_cfg["targets"]:
            setattr(self.se, f"{name}_error", 0.0)

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "model_{}.pt".format(self.it))
        temporary = path + ".tmp"
        torch.save(
            {
                "actor_state_dict": self.alg.actor.state_dict(),
                "critic_state_dict": self.alg.critic.state_dict(),
                "estimator_state_dict": self.se.state_estimator.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
                "estimator_optimizer_state_dict": self.se.optimizer.state_dict(),
                "iter": self.it,
            },
            temporary,
        )
        os.replace(temporary, path)

    def load(self, path, load_optimizer=True):
        super().load(path, load_optimizer)
        loaded_dict = torch.load(path, weights_only=True, map_location=self.device)
        self.se.state_estimator.load_state_dict(loaded_dict["estimator_state_dict"])
        if load_optimizer:
            self.se.optimizer.load_state_dict(
                loaded_dict["estimator_optimizer_state_dict"]
            )

    def switch_to_eval(self):
        super().switch_to_eval()
        self.se.state_estimator.eval()

    def get_inference_actions(self):
        actor_obs, _, _ = self.get_actor_inputs(noisy=False)
        return self.alg.actor.act_inference(actor_obs)

    def get_estimate(self):
        """Current estimate of the target states (for plotting/evaluation)."""
        return self.se.predict(self.get_obs(self.se_cfg["obs"]))

    def export(self, path):
        super().export(path)
        self.se.export(path)
