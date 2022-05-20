#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os
from typing import List

import numpy as np

from sad.generator import ImplicitFeedbackGenerator
from sad.model import SADModel

from .base import TrainerBase, TrainerFactory


@TrainerFactory.register
class SGDTrainer(TrainerBase):
    def __init__(
        self,
        config: dict,
        model: SADModel,
        generator: ImplicitFeedbackGenerator,
        task: "TrainingTask",
    ):
        super().__init__(config, model, generator, task)
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

    @property
    def w_l2(self):
        """:obj:`float`: Read directly from ``self.spec``. The weight of ``L2``
        penalty on parameters of ``XI`` and ``H`` in a SAD model."""
        w_l2 = self.spec.get("w_l2", 0)
        return w_l2

    @w_l2.setter
    def w_l2(self, w_l2: float):
        self.spec["w_l2"] = w_l2

    @property
    def w_l1(self):
        """:obj:`float`: Read directly from ``self.spec``. The weight of ``L1``
        penalty on parameter ``T`` in a SAD model."""
        w_l1 = self.spec.get("w_l1", 0)
        return w_l1

    @w_l1.setter
    def w_l1(self, w_l1: float):
        self.spec["w_l1"] = w_l1

    @property
    def u_idxs(self) -> List[int]:
        """Read directly from ``self.spec``. A list of users represented by user
        indices, on whom log likelihood will be evaluated. Configurable to a subset of
        users for efficiency consideration."""
        u_idxs = self.spec.get("u_idxs")
        if isinstance(u_idxs, int):
            u_idxs = range(u_idxs)
        else:
            u_idxs = [i for i in range(self.model.n)] if not u_idxs else u_idxs
        return u_idxs

    @property
    def i_idxs(self) -> List[int]:
        """Read directly from ``self.spec``. A list of items, represented by item
        indices. The pairwise comparison over those items from users in ``self.u_idxs``
        will be used to evaluate the model during training. Configurable to a subset of
        items for efficiency consideration."""
        i_idxs = self.spec.get("i_idxs")
        if isinstance(i_idxs, int):
            i_idxs = range(i_idxs)
        else:
            i_idxs = [i for i in range(self.model.m)] if not i_idxs else i_idxs
        return i_idxs

    def save(self, working_dir: str = None):
        """Save trainer configuration."""
        if not working_dir:
            working_dir = self.working_dir
        model_s3_key_path = self.model.s3_key_path
        os.makedirs(os.path.join(working_dir, model_s3_key_path), exist_ok=True)
        json.dump(
            self.config,
            open(
                os.path.join(working_dir, model_s3_key_path, "trainer_config.json"), "w"
            ),
        )

    def train(self):
        generator = self.generator
        self.logger.info("Generator begins to prepare data ...")
        generator.prepare()
        self.logger.info("Data preparation done ...")
        model = self.model
        model.reset_parameters()

        n_iters = self.n_iters
        eval_at_every_step = self.eval_at_every_step

        self.on_loop_begin()

        for iter_idx in range(n_iters):
            self.on_iter_begin(iter_idx)

            for step_idx, (u_idx, i_idx, j_idx, obs_uij) in enumerate(generator):
                self.on_step_begin(iter_idx, step_idx)

                g = model.get_gradient_wrt_xuij(u_idx, i_idx, j_idx, obs_uij)
                model.gradient_update(
                    u_idx, i_idx, j_idx, g, self.w_l2, self.w_l1, self.lr
                )

                # calculate log likelihood for a set of users at step end
                ll = 0
                ll0 = 0
                t_sparsity = 0
                mse = -1
                if eval_at_every_step > 0 and ((step_idx) % eval_at_every_step == 0):
                    t_sparsity = model.get_t_sparsity()
                    for u_idx in self.u_idxs:
                        n_items_to_evaluate = len(self.i_idxs)
                        for ii_idx in range(n_items_to_evaluate):
                            i_idx = self.i_idxs[ii_idx]
                            for jj_idx in range(ii_idx + 1, n_items_to_evaluate):
                                j_idx = self.i_idxs[jj_idx]
                                obs_uij = generator.get_obs_uij(u_idx, i_idx, j_idx)
                                ll += model.log_likelihood(u_idx, i_idx, j_idx, obs_uij)

                    if hasattr(generator, "X0"):  # if generator contains true parameter
                        model.calculate_preference_tensor()
                        mse = (
                            np.sum((generator.X0 - model.X) ** 2)
                            / model.n
                            / model.m
                            / model.m
                        )

                    if hasattr(generator, "ll0"):
                        ll0 = generator.ll0

                self.on_step_end(
                    iter_idx, step_idx, ll=ll, t_sparsity=t_sparsity, mse=mse, ll0=ll0
                )

            # calculate log likelihood for a set of users at iter end
            ll = 0
            t_sparsity = model.get_t_sparsity()
            n_items_to_evaluate = len(self.i_idxs)
            for u_idx in self.u_idxs:
                for ii_idx in range(n_items_to_evaluate):
                    i_idx = self.i_idxs[ii_idx]
                    for jj_idx in range(ii_idx + 1, n_items_to_evaluate):
                        j_idx = self.i_idxs[jj_idx]
                        obs_uij = generator.get_obs_uij(u_idx, i_idx, j_idx)
                        ll += model.log_likelihood(u_idx, i_idx, j_idx, obs_uij)
            mse = -1
            if hasattr(generator, "X0"):  # if generator contains true parameter
                model.calculate_preference_tensor()
                mse = (
                    np.sum((generator.X0 - model.X) ** 2) / model.n / model.m / model.m
                )

            ll0 = 0
            if hasattr(generator, "ll0"):
                ll0 = generator.ll0

            self.on_iter_end(iter_idx, ll=ll, t_sparsity=t_sparsity, mse=mse, ll0=ll0)

        self.on_loop_end(ll=ll, t_sparsity=t_sparsity, mse=mse, ll0=ll0)

    def load(self, folder: str):
        pass
