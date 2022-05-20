#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os

from sad.generator import ImplicitFeedbackGenerator
from sad.model import SVDModel

from .base import TrainerBase, TrainerFactory


@TrainerFactory.register
class SVDTrainer(TrainerBase):
    def __init__(
        self,
        config: dict,
        model: SVDModel,
        generator: ImplicitFeedbackGenerator,
        task: "TrainingTask",
    ):
        super().__init__(config, model, generator, task)
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

    @property
    def reg(self) -> float:
        """Regularization parameter. Read directly from ``"reg"`` field
        in ``self.spec``."""
        return self.spec.get("reg", 0.01)

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

        model.initialize_svd_model(self)
        dataset = generator.surprise_dataset

        self.on_loop_begin()
        model.svd_model.fit(dataset)
        self.on_loop_end()

    def load(self, folder: str):
        pass
