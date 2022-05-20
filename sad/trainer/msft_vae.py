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
from scipy.special import logit

from sad.generator import ImplicitFeedbackGenerator
from sad.model import MSFTRecVAEModel

from .base import TrainerBase, TrainerFactory

EPS = 1e-10


@TrainerFactory.register
class MSFTRecVAETrainer(TrainerBase):
    def __init__(
        self,
        config: dict,
        model: MSFTRecVAEModel,
        generator: ImplicitFeedbackGenerator,
        task: "TrainingTask",
    ):
        super().__init__(config, model, generator, task)
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

    @property
    def beta(self) -> float:
        """The beta parameter in beta-VAE model. Will read directly from ``"beta"`` field
        from ``self.spec``."""
        return self.spec.get("beta", 1.0)

    @property
    def evaluation_flag(self) -> bool:
        """An attribute that is specific to ``MSFTRecVAETrainer``. When set to ``True``,
        enable to calculate relative preference scores for each item pair with
        ``i``-th item being more preferrable than ``j``-th item."""
        return self.spec.get("evaluation", False)

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

    def evaluation(self, scores: np.ndarray):
        """Actual method to run the evaluation. During evaluation, item relative scores
        will be calculate for each item pair with i is more preferrable than j. Score
        mean, std and log likelihood of the model will be calculated.


        Args:
            scores (:obj:`np.ndarray`): Pre-calculated user-item preference.

        """
        if not self.evaluation_flag:
            return

        generator = self.generator
        model = self.model

        scores[scores < EPS] = EPS
        scores[scores > 1 - EPS] = 1 - EPS
        scores = logit(scores)

        ll = 0
        n_users = 10
        n_items = 100
        for u_idx in range(n_users):
            self.logger.info(f"Calculating ll using {u_idx+1}/{n_users} user ...")
            u_id = generator.user_idx_to_id[u_idx]
            for i_idx in range(n_items):
                i_id = generator.item_idx_to_id[i_idx]
                for j_idx in range(n_items):
                    j_id = generator.item_idx_to_id[j_idx]
                    obs = generator.get_obs_uij(u_idx, i_idx, j_idx)

                    if obs == 0:  # missing data
                        continue

                    o = 1 if obs == 1 else 0
                    score1 = scores[u_idx, i_idx]
                    score2 = scores[u_idx, j_idx]
                    xuij = score1 - score2
                    ll += (o - 1) * xuij - np.log(1 + np.exp(-1 * xuij))

        preference_scores = []
        for u_idx in range(len(generator.user_idx_to_id)):
            if (u_idx) % 50 == 0:
                self.logger.info(f"Seeping {u_idx+1}/{model.n} user ...")

            pairwise_relationships = generator.user_idx_to_preference[u_idx]
            for i_id, j_id in pairwise_relationships.keys():
                i_idx = generator.item_id_to_idx[i_id]
                j_idx = generator.item_id_to_idx[j_id]

                score1 = scores[u_idx, i_idx]
                score2 = scores[u_idx, j_idx]

                preference_scores.append(score1 - score2)

        preference_scores = np.array(preference_scores)
        metrics = {
            "model_id": [self.task.model_id],
            "ll": [ll],
            "preference_score_mean": np.mean(preference_scores),
            "preference_score_std": np.std(preference_scores),
        }

        model_s3_key_path = self.model.s3_key_path
        abs_model_path = os.path.join(self.task.output_dir, model_s3_key_path)
        os.makedirs(abs_model_path, exist_ok=True)
        json.dump(open(os.path.join(abs_model_path, "metrics.json"), "w"), metrics)

        self.logger.info("Evaluation Done!")

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

        model.initialize_msft_vae_model(self)

        self.on_loop_begin()
        model.msft_vae_model.fit(
            x_train=data_trn,
            x_valid=data_val,
            x_val_tr=data_val,
            x_val_te=data_val,
            mapper=am_all,
        )
        self.on_loop_end()

        scores = model.msft_vae_model.recommend_k_items(
            data, k=model.m, remove_seen=False
        )
        np.savez(os.path.join(self.task.artifact_dir, "scores.npz"), scores=scores)

        model_s3_key_path = self.model.s3_key_path
        abs_model_path = os.path.join(self.task.output_dir, model_s3_key_path)
        os.makedirs(abs_model_path, exist_ok=True)
        np.savez(os.path.join(abs_model_path, "scores.npz"), scores=scores)

        if self.evaluation_flag:
            self.evaluation(scores)

    def load(self, folder: str):
        pass
