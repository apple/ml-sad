#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
from typing import Any

import numpy as np
from recommenders.models.rbm.rbm import RBM

from sad.utils.misc import my_logit

from .base import ModelBase, ModelFactory


@ModelFactory.register
class MSFTRecRBMModel(ModelBase):
    def __init__(self, config: dict, task: "TrainingTask"):
        super().__init__(config, task)
        self.msft_rbm_model = None
        self.w = np.zeros((self.m, self.k))
        self.bv = np.zeros((1, self.m))
        self.bh = np.zeros((1, self.k))

    @property
    def msft_rbm_model(self) -> RBM:
        """The Restricted Boltzmann Machine (RBM) model instance object. We are using
        the implementation of RBM from ``recommenders`` package developed and maintained
        by Mircrosoft. This model will be initialized via
        ``sad.trainer.MSFTRecRBMTrainer`` when calling method
        ``self.initialize_msft_rbm_model()`` of this class. This is because some
        parameters that are required to initialize a RBM model are actually specified in
        its trainer."""
        return self._msft_rbm_model

    @msft_rbm_model.setter
    def msft_rbm_model(self, msft_rbm_model: RBM):
        self._msft_rbm_model = msft_rbm_model

    @property
    def hidden_units(self) -> int:
        """The the number of hidden units in the RBM model. Its value will read
        directly from ``"k"`` field in ``self.spec``."""
        return self.k

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
    def w(self) -> np.ndarray:
        """The weight in RBM model. The size is in ``m x k``. It's value will be
        initialized to zero. When loading a pre-trained ``MSFTRecRBMModel``, its value
        will be loaded too."""
        return self._w

    @w.setter
    def w(self, w: np.ndarray):
        self._w = w

    @property
    def bv(self) -> np.ndarray:
        """The bias for visible unit. The size is ``1 x m``. It's value will be
        initialized to zero. When loading a pre-trained ``MSFTRecRBMModel``, its value
        will be loaded too."""
        return self._bv

    @bv.setter
    def bv(self, bv: np.ndarray):
        self._bv = bv

    @property
    def bh(self) -> np.ndarray:
        """The bias for hidden unit. The size is ``1 x k``. It's value will be
        initialized to zero. When loading a pre-trained ``MSFTRecRBMModel``, its value
        will be loaded too."""
        return self._bh

    @bh.setter
    def bh(self, bh: np.ndarray):
        self._bh = bh

    def initialize_msft_rbm_model(self, trainer: "MSFTRecRBMTrainer"):
        """Initialize a ``RBM`` model object implemented in Python package
        ``recommenders`` . Some training parameters in a ``trainer`` object will be
        needed, therefore a ``sad.trainer.MSFTRecRBMTrainer`` object is supplied as an
        argument. The trainer is supposed to call this method and supply itself as an
        argument. After calling, ``self.msft_rbm_model`` property will contain the actual
        model object.

        Args:
            trainer (:obj:`sad.trainer.MSFTRecRBMTrainer`): A trainer that will call this
                method to initialize a RBM model object.

        """

        model = RBM(
            hidden_units=self.hidden_units,
            learning_rate=trainer.lr,
            minibatch_size=trainer.generator.batch_size,
            training_epoch=trainer.n_epochs,
            with_metrics=True,
            seed=np.random.randint(100000),
        )
        self.msft_rbm_model = model

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
        # fmt: off
        return my_logit(self.msft_rbm_model.predict(u_id, i_id)) - \
                my_logit(self.msft_rbm_model.predict(u_id, j_id))
        # fmt: on

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

    def save(self, working_dir: str = None):
        """Save trained RBM model to a folder (``self.s3_key_path``) rooted at
        ``working_dir``. The three parameters in the RBM are first converted to numpy
        arrays, and then saved to file ``weights.npz`` in the folder of
        ``os.path.join(self.s3_key_path, working_dir)``.

        Model configuration (``self.config``) will be saved too.


        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model and its configuration will be saved.

        """
        if not working_dir:
            working_dir = self.working_dir
        working_dir = os.path.join(working_dir, self.s3_key_path)
        os.makedirs(working_dir, exist_ok=True)
        w = self.msft_rbm_model.w.eval(self.msft_rbm_model.sess)
        bv = self.msft_rbm_model.bv.eval(self.msft_rbm_model.sess)
        bh = self.msft_rbm_model.bh.eval(self.msft_rbm_model.sess)
        np.savez(os.path.join(working_dir, "weights.npz"), w=w, bv=bv, bh=bh)
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
        """Load model from a folder. Need tests to confirm working properly.

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
        weights = np.load(os.path.join(working_dir, "weights.npz"))
        self.w = weights["w"]
        self.bv = weights["bv"]
        self.bh = weights["bh"]
        model = RBM(hidden_units=self.k)

        self.msft_rbm_model = model

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
