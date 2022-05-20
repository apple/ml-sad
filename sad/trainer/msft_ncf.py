#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os
from typing import List

from sad.generator import ImplicitFeedbackGenerator
from sad.model import MSFTRecNCFModel

from .base import TrainerBase, TrainerFactory


@TrainerFactory.register
class MSFTRecNCFTrainer(TrainerBase):
    def __init__(
        self,
        config: dict,
        model: MSFTRecNCFModel,
        generator: ImplicitFeedbackGenerator,
        task: "TrainingTask",
    ):
        super().__init__(config, model, generator, task)
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

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

        dataset = generator.msft_ncf_dataset
        model.initialize_msft_ncf_model(self)

        self.on_loop_begin()
        model.msft_ncf_model.fit(dataset)
        self.on_loop_end()

    def load(self, folder: str):
        pass
