import torch
import torch.nn as nn
import torch.optim as optim

from learning.modules import StateEstimatorNN
from learning.storage import SERolloutStorage
from learning.utils import create_uniform_generator


class StateEstimator:
    """This class provides a learned state estimator.
    This is trained with supervised learning, using only on-policy data
    collected in a rollout storage.
    predict() function provides state estimation for RL given the observation
    update() function optimizes for the nn params
    process_env_step() function stores values in a rollout storage
    """

    state_estimator: StateEstimatorNN

    def __init__(
        self,
        state_estimator,  # nn module
        learning_rate=1e-3,
        num_mini_batches=1,
        num_learning_epochs=1,
        batch_size=2**15,
        max_gradient_steps=10,
        bce_mask=None,
        device="cpu",
        **kwargs,
    ):
        # general parameters
        self.device = device
        self.learning_rate = learning_rate
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.batch_size = batch_size
        self.max_gradient_steps = max_gradient_steps
        # * boolean mask over outputs trained with BCE (e.g. contact
        # * probability); all other outputs are trained with MSE
        self.bce_mask = None if bce_mask is None else bce_mask.to(device)
        self.mean_loss = 0.0
        self.mean_mse_loss = 0.0
        self.mean_bce_loss = 0.0

        # SE storage
        self.transition = SERolloutStorage.Transition()
        self.storage = None

        # SE network and optimizer
        self.state_estimator = state_estimator
        self.state_estimator.to(self.device)
        self.optimizer = optim.Adam(self.state_estimator.parameters(), lr=learning_rate)
        self.SE_loss_fn = nn.MSELoss()

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, se_shape):
        self.storage = SERolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            se_shape,
            device=self.device,
        )

    def predict(self, obs):
        estimate = self.state_estimator.evaluate(obs)
        if self.bce_mask is not None:
            estimate = torch.where(self.bce_mask, torch.sigmoid(estimate), estimate)
        return estimate

    def loss(self, obs, targets):
        """MSE on regression outputs plus BCE (on logits) on probabilities"""
        raw = self.state_estimator.evaluate(obs)
        if self.bce_mask is None:
            return self.SE_loss_fn(raw, targets), raw.new_zeros(())
        mse = self.SE_loss_fn(raw[:, ~self.bce_mask], targets[:, ~self.bce_mask])
        bce = nn.functional.binary_cross_entropy_with_logits(
            raw[:, self.bce_mask], targets[:, self.bce_mask]
        )
        return mse, bce

    def update_from_data(self, data):
        """Supervised update on a rollout TensorDict with keys
        "se_obs" and "se_targets" (as collected by EstimatorRunner)."""
        self.mean_mse_loss = 0.0
        self.mean_bce_loss = 0.0
        counter = 0
        generator = create_uniform_generator(
            data, self.batch_size, max_gradient_steps=self.max_gradient_steps
        )
        for batch in generator:
            mse, bce = self.loss(batch["se_obs"], batch["se_targets"])
            loss = mse + bce
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.mean_mse_loss += mse.item()
            self.mean_bce_loss += bce.item()
            counter += 1
        self.mean_mse_loss /= counter
        self.mean_bce_loss /= counter
        self.mean_loss = self.mean_mse_loss + self.mean_bce_loss
        return self.mean_loss

    def process_env_step(self, obs, SE_targets):
        # Record the transition
        self.transition.SE_targets = SE_targets
        self.transition.observations = obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def update(self):
        """Update the SE neural network weights via supervised learning"""
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        mean_loss = 0
        for obs_batch, SE_target_batch in generator:
            SE_estimate_batch = self.state_estimator.evaluate(obs_batch)

            SE_loss = self.SE_loss_fn(SE_estimate_batch, SE_target_batch)
            self.optimizer.zero_grad()
            SE_loss.backward()
            self.optimizer.step()
            mean_loss += SE_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_loss /= num_updates
        self.storage.clear()

        return mean_loss

    def export(self, path):
        self.state_estimator.export(path)
