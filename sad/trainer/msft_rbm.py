#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os

import numpy as np
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.datasets.sparse import AffinityMatrix

from sad.generator import ImplicitFeedbackGenerator
from sad.model import MSFTRecRBMModel

from .base import TrainerBase, TrainerFactory


@TrainerFactory.register
class MSFTRecRBMTrainer(TrainerBase):
    def __init__(
        self,
        config: dict,
        model: MSFTRecRBMModel,
        generator: ImplicitFeedbackGenerator,
        task: "TrainingTask",
    ):
        super().__init__(config, model, generator, task)
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

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
        header = {
            "col_user": "userID",
            "col_item": "itemID",
            "col_rating": "rating",
        }
        am_all = AffinityMatrix(df=data_df, **header)
        data, _, _ = am_all.gen_affinity_matrix()
        data_trn, data_val = numpy_stratified_split(data, ratio=0.85)

        model.initialize_msft_rbm_model(self)

        self.on_loop_begin()
        model.msft_rbm_model.fit(data_trn, xtst=data_val)
        self.on_loop_end()

        scores, _ = model.msft_rbm_model.recommend_k_items(
            data, top_k=model.m, remove_seen=False
        )
        np.savez(os.path.join(self.task.artifact_dir, "scores.npz"), scores=scores)

    def load(self, folder: str):
        pass
