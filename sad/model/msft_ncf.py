#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
from typing import Any, List

import numpy as np
from recommenders.models.ncf.ncf_singlenode import NCF

from sad.utils.misc import my_logit

from .base import ModelBase, ModelFactory


@ModelFactory.register
class MSFTRecNCFModel(ModelBase):
    def __init__(self, config: dict, task: "TrainingTask"):
        super().__init__(config, task)
        self.msft_ncf_model = None

    @property
    def msft_ncf_model(self) -> NCF:
        """The Neural Collaborative Filtering (NCF) model instance object. We are using
        the implementation of NCF from ``recommenders`` package developed and maintained
        by Mircrosoft. This model will be initialized via
        ``sad.trainer.MSFTRecNCFTrainer`` when calling
        method ``self.initialize_msft_ncf_model()`` of this class. This is because some
        parameters required to initialize a NCF model are actually specified in trainer.
        Therefore those paraemters need to be passed from trainer to this model."""
        return self._msft_ncf_model

    @msft_ncf_model.setter
    def msft_ncf_model(self, msft_ncf_model: NCF):
        self._msft_ncf_model = msft_ncf_model

    @property
    def layer_sizes(self) -> List[int]:
        """The layer sizes of the MLP part of the NCF model. Its value will be read
        directly from ``"layer_sizes"`` field in ``self.spec``. Default to ``[128]``,
        a one layer perceptron with 128 nodes."""
        layer_sizes = self.spec.get("layer_sizes") or [128]
        return layer_sizes

    @property
    def model_type(self) -> str:
        """The type of NCF model that is supported by ``"recommenders"`` package.
        Currently could take ``"MLP|GMF|NeuMF"``. Read directly from ``"model_type"``
        field in ``self.spec``. Default to ``"NeuMF"``."""
        model_type = self.spec.get("model_type", "NeuMF")
        model_type = model_type.lower()
        assert model_type in {
            "mlp",
            "gmf",
            "neumf",
        }, f"Provided model type {model_type} is not supported."
        return model_type

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
        """The number of latent dimentions."""
        return self.spec.get("k")

    def initialize_msft_ncf_model(self, trainer: "MSFTRecNCFTrainer"):
        """Initialize a ``NCF`` model object implemented in Python package
        ``recommenders`` . Some training parameters in a ``trainer`` object will be
        needed, therefore a ``sad.trainer.MSFTRecNCFTrainer`` object is supplied as an
        argument. The trainer is supposed to call this method and supply itself as an
        argument. After calling, ``self.msft_ncf_model`` property will contain the actual
        model object.

        Args:
            trainer (:obj:`sad.trainer.MSFTRecNCFTrainer`): A trainer that will call this
                method to initialize a NCF model object.

        """

        model = NCF(
            n_users=self.n,
            n_items=self.m,
            n_factors=self.k,
            model_type=self.model_type,
            layer_sizes=self.layer_sizes,
            n_epochs=trainer.n_epochs,
            batch_size=trainer.generator.batch_size,
            learning_rate=trainer.lr,
            verbose=1,
        )
        self.msft_ncf_model = model

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
        return my_logit(self.msft_ncf_model.predict(u_id, i_id)) - \
                my_logit(self.msft_ncf_model.predict(u_id, j_id))
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
        """Save trained NCF model to a folder (``self.s3_key_path``) rooted at
        ``working_dir``. The actual saving operation will be delegated to
        ``self.msft_ncf_model.save()``. In the meanwhile, some additional information
        about the model will be saved to ``additional_info.json``. Those additional
        information will be used when loading a trained NCF model.

        Model configuration (``self.config``) will be saved too.


        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model and its configuration will be saved.

        """
        if not working_dir:
            working_dir = self.working_dir
        working_dir = os.path.join(working_dir, self.s3_key_path)
        os.makedirs(working_dir, exist_ok=True)
        self.msft_ncf_model.save(working_dir)

        additional_info = {
            "user2id": self.msft_ncf_model.user2id,
            "item2id": self.msft_ncf_model.item2id,
            "id2user": self.msft_ncf_model.id2user,
            "id2item": self.msft_ncf_model.id2item,
        }
        json.dump(
            additional_info,
            open(os.path.join(working_dir, "additional_info.json"), "w"),
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
        model = NCF(
            n_users=self.n,
            n_items=self.m,
            n_factors=self.k,
            model_type=self.model_type,
            layer_sizes=self.layer_sizes,
        )
        dir_name = f"{self.model_type}_dir"
        arg_dict = {dir_name: working_dir}
        model.load(**arg_dict)

        additional_info = json.load(
            open(os.path.join(working_dir, "additional_info.json"))
        )
        model.user2id = additional_info.get("user2id")
        model.item2id = additional_info.get("item2id")
        model.id2user = additional_info.get("id2user")
        model.id2item = additional_info.get("id2item")

        self.msft_ncf_model = model

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
