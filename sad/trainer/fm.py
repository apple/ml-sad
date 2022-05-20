#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os

from sad.generator import ImplicitFeedbackGenerator
from sad.model import FMModel

from .base import TrainerBase, TrainerFactory


@TrainerFactory.register
class FMTrainer(TrainerBase):
    def __init__(
        self,
        config: dict,
        model: FMModel,
        generator: ImplicitFeedbackGenerator,
        task: "TrainingTask",
    ):
        super().__init__(config, model, generator, task)
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

    @property
    def loss_name(self) -> str:
        """Read directly from ``"loss"`` field in ``self.spec``. Currently can take
        ``"bpr"|"warp"`` two values. Default is ``"bpr"``. Specific to
        ``sad.model.FMModel``."""
        return self.spec.get("loss", "bpr")

    @property
    def n_negative_samples(self) -> int:
        """Read directly from ``"n_negative_samples"`` field in ``self.spec``. It means
        the number of samples that will be drawn for ``"warp"`` loss."""
        return self.spec.get("n_negative_samples", 1)

    @property
    def w_l2(self) -> float:
        """Weight of L2 regularization to parameters. Read directly from ``"w_l2"`` field
        in ``self.spec``."""
        return self.spec.get("w_l2", 0.01)

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

        data_df = generator.data_df
        data_df = data_df[data_df["rating"] > 0]
        data_df = data_df.rename(columns={"userID": "user_id", "itemID": "item_id"})
        model.initialize_fm_model(self)

        self.on_loop_begin()
        model.fm_model.fit(
            data_df[["user_id", "item_id"]], epochs=self.n_epochs, verbose=True
        )
        self.on_loop_end()

    def load(self, folder: str):
        pass
