#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
import pickle
from typing import Any

import numpy as np
from rankfm.rankfm import RankFM

from sad.utils.misc import my_logit

from .base import ModelBase, ModelFactory


@ModelFactory.register
class FMModel(ModelBase):
    def __init__(self, config: dict, task: "TrainingTask"):
        super().__init__(config, task)
        self.fm_model = None

    @property
    def fm_model(self) -> RankFM:
        """The Factorization Machine (FM) model instance object. We are using the
        implementation of FM from ``RankFM`` package. This model will be initialized via
        ``sad.trainer.FMTrainer`` when calling method ``self.initialize_fm_model()`` of
        this class. This is because some paraemters that are required to initialize a
        ``RankFM`` model are owned by trainer. Therefore those parameters need to be
        passed from the trainer."""
        return self._fm_model

    @fm_model.setter
    def fm_model(self, fm_model: RankFM):
        self._fm_model = fm_model

    @property
    def n(self) -> int:
        """The number of users."""
        return self.spec.get("n")

    @property
    def m(self) -> int:
        """The number of items."""
        return self.spec.get("m")

    @property
    def k(self) -> int:
        """The number of latent dimensions."""
        return self.spec.get("k")

    def initialize_fm_model(self, trainer: "FMTrainer"):
        """Initialize a FM model object implemented in package ``RankFM``. Some training
        parameters in a ``trainer`` object will be needed, therefore a
        ``sad.trainer.FMTrainer`` object is supplied as an argument. The trainer
        is supposed to call this method and supply itself as an argument. After calling,
        ``self.fm_model`` property will contain the actual model object.

        Args:
            trainer (:obj:`sad.trainer.FMTrainer`): A trainer that will call this
                method to initialize a FM model.

        """
        model = RankFM(
            factors=self.k,
            loss=trainer.loss_name,
            max_samples=trainer.n_negative_samples,
            alpha=trainer.w_l2,
            beta=trainer.w_l2,
            learning_rate=trainer.lr,
        )

        self.fm_model = model

    def get_xuij(self, u_id: str, i_id: str, j_id: str, **kwargs) -> float:
        """Calculate preference score between two items for a particular user. The
        preference strength of an item for a user of this model class is the logit of
        model's prediction probability. The difference between preference strengths of
        the two items from the provided user is how the preference score is calculated.
        For this class, user and item ids (not indices) are needed as arguments.

        Args:
            u_id (:obj:`str`): User ID.
            i_id (:obj:`str`): Item ID.
            j_id (:obj:`str`): Item ID.

        Returns:
            :obj:`float`: Preference score between item ``i_id`` and ``j_id`` for
            user ``u_id``.

        """
        data = np.array([[u_id, i_id], [u_id, j_id]])
        prediction = self.fm_model.predict(data)
        return my_logit(prediction[0]) - my_logit(prediction[1])

    def log_likelihood(
        self, u_id: str, i_id: str, j_id: str, obs_uij: int, **kwargs
    ) -> float:
        """Calculate log likelihood.

        Args:
            u_id (:obj:`str`): A user ID.
            i_id (:obj:`str`): An item ID. The ID of left item in preference tensor.
            j_id (:obj:`str`): An item ID. The ID of right item in preference tensor.
            obs_uij (:obj:`int`): The observation of ``(u_id, i_id, j_id)`` from dataset.
                Take ``1|-1|0`` three different values. ``"1"`` suggests item ``i_id`` is
                more preferable than item ``j_id`` for user ``u_id``. ``"-1"``
                suggests the opposite. ``"0"`` means the preference information is not
                available (missing data).
        Returns:
            (:obj:`float`): Return the contribution to the log likelihood from
            observation of ``(u_id, i_id, j_id)``. Return ``0`` when the observation
            is missing.
        """
        if obs_uij == 0:  # missing data
            return 0

        o = 1 if obs_uij == 1 else 0
        xuij = self.get_xuij(u_id=u_id, i_id=i_id, j_id=j_id)
        l = (o - 1) * xuij - np.log(1 + np.exp(-1 * xuij))
        return l

    def save(self, working_dir: str = None):
        """Save trained FM model to a folder (``self.s3_key_path``) rooted at
        ``working_dir``. The trained FM model (``self.fm_model``) will be saved as a
        pickle file named ``model.pickle`` under the folder.

        Model configuration (``self.config``) will be saved too.


        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model and its configuration will be saved.

        """
        if not working_dir:
            working_dir = self.working_dir
        working_dir = os.path.join(working_dir, self.s3_key_path)
        os.makedirs(working_dir, exist_ok=True)
        pickle.dump(
            self.fm_model, open(os.path.join(working_dir, "model.pickle"), "wb")
        )
        json.dump(
            self.config,
            open(os.path.join(working_dir, "model_config.json"), "w"),
        )

    def save_checkpoint(self, working_dir: str, checkpoint_id: int = 1):
        """Haven't implemented this functionality yet."""
        pass

    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError

    def load(self, working_dir: str = None, filename: str = None):
        """Load model from a folder.

        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model and configuration are stored.
            filename (:obj:`str`): Filename containing model parameters. The full path
                of the file will be
                ``os.path.join(working_dir, self.s3_key_path, filename)``.

        """
        if not working_dir:
            working_dir = self.working_dir
        working_dir = os.path.join(working_dir, self.s3_key_path)
        pickle_filename = os.path.join(working_dir, "model.pickle")
        model_obj = pickle.load(open(pickle_filename, "rb"))
        self.fm_model = model_obj

    def load_checkpoint(self, working_dir: str, checkpoint_id: int = 1):
        """Havn't implemented this functionality yet."""
        pass

    def load_best(self, working_dir: str, criterion: str = "ll"):
        """Havn't implemented this functionality yet."""
        pass

    def reset_parameters(self):
        """Doing nothing."""
        pass

    def parameters_for_monitor(self) -> dict:
        """Return nothing."""
        return {}
