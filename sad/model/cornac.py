#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
import pickle
from typing import Any

import cornac.models as CModels
import numpy as np

from sad.utils.misc import my_logit

from .base import ModelBase, ModelFactory

ADDITIONAL_FIELD_NAMES = ["train_set"]


@ModelFactory.register
class CornacModel(ModelBase):
    def __init__(self, config: dict, task: "TrainingTask"):
        super().__init__(config, task)
        self.cornac_model = None

    @property
    def cornac_model(self) -> CModels.Recommender:
        """A model instance object from Cornac package. This model will be initialized
        via ``sad.trainer.CornacTrainer`` when calling method
        ``self.initialize_cornac_model()`` of this class. This is because some parameters
        needed to initialize a Cornac model are actually related to trainer
        specifications. Therefore those parameters need to be passed from trainer."""
        return self._cornac_model

    @cornac_model.setter
    def cornac_model(self, cornac_model: CModels.Recommender):
        self._cornac_model = cornac_model

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

    def initialize_cornac_model(self, trainer: "CornacTrainer"):
        """Initialize a model object implemented in Cornac package. Some training
        parameters in a ``trainer`` object will be needed, therefore a
        ``sad.trainer.CornacTrainer`` object is supplied as an argument. The trainer is
        supposed to call this method and supply itself as an argument. After calling,
        ``self.cornac_model`` property will contain the actual model object.
        ``"cornac_model_name"`` field in ``self.spec`` contains the class name that will
        be used to initialize a Cornac model instance.

        Args:
            trainer (:obj:`sad.trainer.CornacTrainer`): A trainer that will call this
                method to initialize a Cornac model.

        Raises:
            AttributeError: When supplied ``"cornac_model_name"`` is not an existing
                Cornac model class in ``models`` module from Cornac package.

        """
        cornac_model_name = self.spec.get("cornac_model_name", "BPR")
        if not hasattr(CModels, cornac_model_name):
            raise AttributeError(
                f"Cornac model package does not have {cornac_model_name} implemented."
            )
        cornac_model_class = getattr(CModels, cornac_model_name)
        if cornac_model_name == "BiVAECF":
            # "BiVAECF" needs additional setup
            cornac_model = cornac_model_class(
                k=self.k,
                encoder_structure=[128, 64, 32],
                act_fn="relu",
                beta_kl=0.01,
                n_epochs=trainer.n_epochs,
                learning_rate=trainer.lr,
                batch_size=trainer.generator.batch_size,
                likelihood="bern",
                verbose=True,
            )
        else:
            cornac_model = cornac_model_class(
                k=self.k,
                max_iter=trainer.n_iters,
                learning_rate=trainer.lr,
                lambda_reg=trainer.lambda_reg,
                verbose=True,
            )
        self.cornac_model = cornac_model

    def get_xuij(self, u_idx: int, i_idx: int, j_idx: int, **kwargs) -> float:
        """Calculate preference score between two items for a particular user. The
        preference strength of an item for a user of this model class is the logit of
        model's prediction probability. The difference between preference strengths of
        the two items from the provided user is how the preference score is calculated.
        For this class, user and item indices are needed.

        Args:
            u_idx (:obj:`int`): User index, from ``0`` to ``self.n-1``.
            i_idx (:obj:`int`): Item index, from ``0`` to ``self.m-1``.
            j_idx (:obj:`int`): Item index, from ``0`` to ``self.m-1``.

        Returns:
            :obj:`float`: Preference score between ``i_idx``-th item and ``j_idx``-th
            item for ``u_idx``-th user.

        """
        # fmt: off
        return my_logit(self.cornac_model.score(u_idx, i_idx)) - \
                my_logit(self.cornac_model.score(u_idx, j_idx))
        # fmt: on

    def log_likelihood(
        self, u_idx: int, i_idx: int, j_idx: int, obs_uij: int, **kwargs
    ) -> float:
        """Calculate log likelihood.

        Args:
            u_idx (:obj:`int`): Index of user in user set. 0-based.
            i_idx (:obj:`int`): Index of i-th item. It is the idx of left item in
                preference tensor.
            j_idx (:obj:`int`): Index of j-th item. It is the idx of right item in
                preference tensor.
            obs_uij (:obj:`int`): The observation at ``(u_idx, i_idx, j_idx)``. Take
                ``1|-1|0`` three different values. ``"1"`` suggests ``i_idx``-th item is
                more preferable than ``j_idx``-th item for ``u_idx``-th user. ``"-1"``
                suggests the opposite. ``"0"`` means the preference information is not
                available (missing data).
        Returns:
            (:obj:`float`): Return the contribution to the log likelihood from
            observation at ``(u_idx, i_idx, j_idx)``. Return ``0`` when the observation
            is missing.
        """
        if obs_uij == 0:  # missing data
            return 0

        o = 1 if obs_uij == 1 else 0
        xuij = self.get_xuij(u_idx=u_idx, i_idx=i_idx, j_idx=j_idx)
        l = (o - 1) * xuij - np.log(1 + np.exp(-1 * xuij))
        return l

    def save(self, working_dir: str = None):
        """Save trained Cornac model to a folder (``self.s3_key_path``) rooted at
        ``working_dir``. The actual save operation will be delegated to
        ``self.cornac_model.save()``. In the meanwhile, some additional fields defined
        by ``ADDITIONAL_FIELD_NAMES`` macro in this module will be serialized to pickle
        files in the same folder.

        Model configuration (``self.config``) will be saved too.


        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model and some additional information will be saved.

        """
        if not working_dir:
            working_dir = self.working_dir
        working_dir = os.path.join(working_dir, self.s3_key_path)
        os.makedirs(working_dir, exist_ok=True)
        self.cornac_model.save(working_dir)
        for field_name in ADDITIONAL_FIELD_NAMES:
            if hasattr(self.cornac_model, field_name):
                pickle.dump(
                    getattr(self.cornac_model, field_name),
                    open(os.path.join(working_dir, f"{field_name}.pickle"), "wb"),
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
                where model and some additional information are stored.
            filename (:obj:`str`): Filename containing model parameters. The full path
                of the file will be
                ``os.path.join(working_dir, self.s3_key_path, filename)``.

        """

        if not working_dir:
            working_dir = self.working_dir
        cornac_model_name = self.spec.get("cornac_model_name", "BPR")
        working_dir = os.path.join(working_dir, self.s3_key_path)
        self.cornac_model = CModels.Recommender.load(
            os.path.join(working_dir, cornac_model_name)
        )
        for field_name in ADDITIONAL_FIELD_NAMES:
            pickle_filename = os.path.join(working_dir, f"{field_name}.pickle")
            if os.path.exists(pickle_filename):
                field_obj = pickle.load(open(pickle_filename, "rb"))
                setattr(self.cornac_model, field_name, field_obj)

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
