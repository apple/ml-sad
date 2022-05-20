#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import surprise
from recommenders.models.surprise.surprise_utils import predict

from sad.utils.misc import my_logit

from .base import ModelBase, ModelFactory


@ModelFactory.register
class SVDModel(ModelBase):
    def __init__(self, config: dict, task: "TrainingTask"):
        super().__init__(config, task)
        self.svd_model = None
        self.prediction_cache = {}

    @property
    def svd_model(self) -> surprise.SVD:
        """Singular Value Decomposition (SVD) model instance object. We are using the
        implementation of SVD from ``surprise`` package. This model will be initialized
        via ``sad.trainer.SVDTrainer`` when calling method
        ``self.initialize_svd_model()`` of this class."""
        return self._svd_model

    @svd_model.setter
    def svd_model(self, svd_model: surprise.SVD):
        self._svd_model = svd_model

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

    @property
    def prediction_cache(self) -> Dict[Tuple[str, str], float]:
        """A dictionary contains the prediction cache. The key is a user id and item id
        pair, and value is model's prediction."""
        return self._prediction_cache

    @prediction_cache.setter
    def prediction_cache(self, prediction_cache: Dict[Tuple[str, str], float]):
        self._prediction_cache = prediction_cache

    def initialize_svd_model(self, trainer: "SVDTrainer"):
        """Initialize a SVD model object implemented in package ``surprise``. Some
        training parameters in a ``trainer`` object will be needed, therefore a
        ``sad.trainer.SVDTrainer`` object is supplied as an argument. The trainer
        is supposed to call this method and supply itself as the argument. After calling,
        ``self.svd_model`` property will contain the actual model object.

        Args:
            trainer (:obj:`sad.trainer.SVDTrainer`): A trainer that will call this
                method to initialize a SVD model.

        """
        model = surprise.SVD(
            n_factors=self.k,
            n_epochs=trainer.n_epochs,
            lr_all=trainer.lr,
            reg_all=trainer.reg,
            verbose=True,
        )

        self.svd_model = model

    def get_xuij(self, u_id: str, i_id: str, j_id: str, **kwargs) -> float:
        """Calculate preference score between two items for a particular user. The
        preference strength of an item for a user of this model class is the logit of
        model's prediction probability. The difference between preference strengths of
        the two items from the provided user is how the preference score is calculated.
        For this class, user and item ids (instead of indices) are needed as arguments.

        Args:
            u_id (:obj:`str`): User ID.
            i_id (:obj:`str`): Item ID.
            j_id (:obj:`str`): Item ID.

        Returns:
            :obj:`float`: Preference score between item ``i_id`` and ``j_id`` for
            user ``u_id``.

        """
        if (u_id, i_id) in self.prediction_cache:
            prediction_i = self.prediction_cache[(u_id, i_id)]
        else:
            data_df = pd.DataFrame({"user_id": [u_id], "item_id": [i_id]})
            prediction = predict(
                self.svd_model, data_df, usercol="user_id", itemcol="item_id"
            )
            prediction_i = prediction["prediction"][0]
            self.prediction_cache[(u_id, i_id)] = my_logit(prediction_i)

        if (u_id, j_id) in self.prediction_cache:
            prediction_j = self.prediction_cache[(u_id, j_id)]
        else:
            data_df = pd.DataFrame({"user_id": [u_id], "item_id": [j_id]})
            prediction = predict(
                self.svd_model, data_df, usercol="user_id", itemcol="item_id"
            )
            prediction_j = prediction["prediction"][0]
            self.prediction_cache[(u_id, j_id)] = my_logit(prediction_j)

        return prediction_i - prediction_j

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
            observation of ``(u_id, i_id, j_id)``. Return ``0`` when the observation is
            missing.
        """
        if obs_uij == 0:  # missing data
            return 0

        o = 1 if obs_uij == 1 else 0
        xuij = self.get_xuij(u_id=u_id, i_id=i_id, j_id=j_id)
        l = (o - 1) * xuij - np.log(1 + np.exp(-1 * xuij))
        return l

    def save(self, working_dir: str = None, filename: str = "model-params.npz"):
        """Save trained SVD model to a folder (``self.s3_key_path``) rooted at
        ``working_dir``. The model object ``self.svd_model`` will be saved as a pickle
        file named ``model.pickle`` in the folder.

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
            self.svd_model, open(os.path.join(working_dir, "model.pickle"), "wb")
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
        self.svd_model = model_obj

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
