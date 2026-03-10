"""
ggdm.py

Implementation of the GGDM class.
"""

from copy import deepcopy
from functools import partial
from typing import Union

import torch
from condevo.diffusion import DM
from torch import Tensor, no_grad, optim, vmap
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class GGDM(DM):
    """Gradient guidance diffusion model base-class."""

    def __init__(self, progress_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_disable = not progress_bar

    @no_grad()
    def sample(
        self,
        shape: tuple,
        num: int = None,
        x_source: Tensor = None,
        conditions=None,
        t_start=None,
        **kwargs,
    ):
        """
        Sample `num` points from the diffusion model with given `shape` and `conditions`. If `num` is not specified, sample points from `x_source` tensor.

        Args:
            shape (tuple): Shape of the sampled points.
            num (int, optional): Number of points to be sampled. Defaults to None.
            x_source (torch.tensor, optional): Source tensor to sample points. Defaults to None.
            conditions (tuple, optional): Conditions for the diffusion model. Defaults to None.
            t_start (int, optional): Starting time step for the diffusion process. Defaults to None.

        Returns:
            torch.tensor: Sampled points from the diffusion model.
        """

        if (num is None) and (x_source is None):
            raise ValueError("Either `num` or `x_source` should be specified")

        if (num is not None) and (x_source is not None):
            raise ValueError(
                "Only one of `num` and `x_source` should be specified"
            )

        if conditions is None:
            conditions = tuple()

        if num is not None:
            x_source = self.draw_random(num, *shape)

        self.eval()
        self.nn.eval()

        do_save = kwargs.get("save_diffusion", False) or getattr(
            self, "save_diffusion_flag", False
        )

        sample_vectorized = vmap(
            partial(self.sample_point, save_diffusion=do_save),
            randomness="different",
        )
        x_sampled = sample_vectorized(x_source, *conditions, t_start=t_start)

        if do_save:
            self.latest_diffusion = x_sampled.clone()

            # Return only the LAST step (Batch, Dim) so standard ES logic continues working
            x_sampled = x_sampled[:, -1, :]
            return x_sampled

        # check for valid parameter range
        exceeding_x = self.exceeds_diff_range(x_sampled) > 0
        exceeding_count = 0
        while self.diff_range not in [None, 0, 0.0] and exceeding_x.any():
            # new sample points
            exceeding_x_source = self.draw_random(
                int(sum(exceeding_x)), *shape
            )

            if exceeding_count > 2:  # try 10 times to sample valid points
                # clamp to diff_range if too many iterations
                exceeding_x_source = self.diff_clamp(exceeding_x_source)
                x_sampled[exceeding_x] = exceeding_x_source
                break

            else:
                exceeding_conditions = [
                    condition[exceeding_x] for condition in conditions
                ]
                x_resampled = sample_vectorized(
                    exceeding_x_source, *exceeding_conditions
                )

                # check for valid parameter range and integrate into samples
                exceeding_resampled = self.exceeds_diff_range(x_resampled) > 0
                valid_resampled = torch.where(exceeding_x)[0][
                    ~exceeding_resampled
                ]
                x_sampled[valid_resampled] = x_resampled[~exceeding_resampled]

            exceeding_x = self.exceeds_diff_range(x_sampled) > 0
            exceeding_count += 1

        return x_sampled

    ### Currently only included to change tqdm behaviour
    def fit(
        self,
        x,
        *conditions,
        weights=None,
        optimizer: Union[str, type] = optim.Adam,
        max_epoch=100,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        scheduler="cosine",
    ):
        """Train the diffusion model to the given data.

        The diffusion model is first set to training mode, then the optimizer
        is initialized with the given parameters. The loss function is set to
        MSELoss. If weights are not specified, they are set to ones.
        After training, the model is set to evaluation mode.

        :param x: torch.tensor, Input data for the diffusion model.
        :param conditions: tuple, Conditions for the diffusion model (will be
        concatenated with x in last `dim`).
        :param weights: torch.tensor, Weights for data point in the loss
        function (to weight high-fitness appropriately). Defaults to None.
        :param optimizer: str or type, Optimizer for the diffusion model.
        Defaults to optim.Adam.
        :param max_epoch: int, Maximum number of epochs for training. Defaults
        to 100.
        :param lr: float, Learning rate for the optimizer. Defaults to 1e-3.
        :param weight_decay: float, Weight decay for the optimizer. Defaults to
        1e-5.
        :param batch_size: int, Batch size for the training data. Defaults to
        32.
        :param scheduler: str or type, Scheduler for the optimizer. Defaults to
        None; other choices are "cosine", "linear", "reduce_on_plateau", or a
        torch scheduler instance.

        :return: list, Loss history of the training process.
        """

        self.train()
        device = self.device

        # --- initialize optimizer ---
        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)

        optimizer = optimizer(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        # --- learning rate scheduler ---
        if scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max_epoch, eta_min=1e-6
            )
        elif scheduler == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=max_epoch,
            )
        elif scheduler == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.9,
                patience=10,
                threshold=1e-4,
                min_lr=1e-6,
            )

        # --- loss function ---
        loss_function = MSELoss(reduction="none")
        grad_clip_value = (
            self.clip_gradients
            if not isinstance(self.clip_gradients, bool)
            else 1.0
        )

        # -- Scale data and clean according to defined condevo.preprocessign.Scaler ---
        x, weights, conditions = self.scaler.clean(
            x=x, weights=weights, conditions=conditions
        )

        if self.diff_range_filter:
            # filter out potential exceeding data
            exceeding = self.exceeds_diff_range(x) > 0
            if exceeding.any():
                x = x[~exceeding]
                weights = weights[~exceeding]
                conditions = tuple(c[~exceeding] for c in conditions)

        if x.shape[0] == 0:
            raise ValueError(
                "All samples removed after NaN cleaning and/or diff_range filtering."
            )

        # transform data into z-space
        x, weights, conditions = self.scaler.fit_transform(
            x=x, weights=weights, conditions=conditions
        )

        dataset = TensorDataset(x, *conditions, weights)
        training_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # --- log dataset ---
        self.logger.next()
        # self.logger.log_dataset(x, weights, *conditions)

        # --- training loop ---
        loss_history, best_model, best_loss = [], None, torch.inf
        bar = tqdm(
            range(max_epoch),
            desc="Training\t",  ### "Training Diffusion Model",
            unit="epoch",
            disable=self.tqdm_disable,  ### Added disable option
        )
        bar.set_postfix(loss=0.0)
        for epoch in bar:
            epoch_loss = 0
            num_updates = 0
            for x_batch, *c_batch, w_batch in training_dataloader:
                x_batch = x_batch.to(device)
                c_batch = [c.to(device) for c in c_batch]
                w_batch = w_batch.to(device)
                optimizer.zero_grad()

                v, v_pred = self.eval_val_pred(x_batch, *c_batch)
                loss = loss_function(v, v_pred) * w_batch
                reg_loss = self.regularize(x_batch, w_batch, *c_batch)
                loss = (loss + reg_loss).mean()

                if self.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), grad_clip_value
                    )

                epoch_loss = epoch_loss + loss.item()

                loss.backward()
                optimizer.step()
                num_updates += 1

            self.logger.log_scalar(f"Loss/Train", epoch_loss, epoch)
            epoch_loss = epoch_loss / (
                num_updates or 1
            )  # avoid division by zero
            loss_history.append(epoch_loss)
            bar.set_postfix(loss=epoch_loss)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                self.logger.log_scalar(f"LR", current_lr, epoch)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = deepcopy(self.nn.state_dict())

        # load best model
        if best_model is not None:
            self.nn.load_state_dict(best_model)

        self.logger.log_scalar(
            f"Loss/Generation", best_loss, self.logger.generation
        )
        self.eval()
        self.nn.eval()
        return loss_history
