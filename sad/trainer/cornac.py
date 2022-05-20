#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os

from sad.generator import ImplicitFeedbackGenerator
from sad.model import CornacModel

from .base import TrainerBase, TrainerFactory


@TrainerFactory.register
class CornacTrainer(TrainerBase):
    def __init__(
        self,
        config: dict,
        model: CornacModel,
        generator: ImplicitFeedbackGenerator,
        task: "TrainingTask",
    ):
        super().__init__(config, model, generator, task)
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

    @property
    def lambda_reg(self):
        """:obj:`float`: Read directly from ``self.spec``. The ``lambda`` regularization
        parameter that will be used during training. Specific to
        ``sad.model.CoracModel``."""
        lambda_reg = self.spec.get("lambda", 0)
        return lambda_reg

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
        model.initialize_cornac_model(self)

        dataset = generator.cornac_dataset

        self.on_loop_begin()
        model.cornac_model.fit(dataset)
        self.on_loop_end()

    def load(self, folder: str):
        pass
