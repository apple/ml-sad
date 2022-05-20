#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
from typing import Any

import numpy as np
from recommenders.models.vae.standard_vae import StandardVAE

from .base import ModelBase, ModelFactory


@ModelFactory.register
class MSFTRecVAEModel(ModelBase):
    def __init__(self, config: dict, task: "TrainingTask"):
        super().__init__(config, task)
        self.msft_vae_model = None

    @property
    def msft_vae_model(self) -> StandardVAE:
        """Variational Auto Encoder (VAE) model instance object. We are using the
        implementation of VAE from ``recommenders`` package developed and maintained by
        MSFT. This model will be initialized via ``sad.trainer.VAETrainer`` when calling
        method ``self.initialize_msft_vae_model()`` of this class. This is because some
        parameters that are required to initialize a VAE model are actually specified in
        its trainer."""
        return self._msft_vae_model

    @msft_vae_model.setter
    def msft_vae_model(self, msft_vae_model: StandardVAE):
        self._msft_vae_model = msft_vae_model

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

    def initialize_msft_vae_model(self, trainer: "MSFTRecVAETrainer"):
        """Initialize a VAE model object implemented in package ``recommenders``. Some
        training parameters in a ``trainer`` object will be needed, therefore a
        ``sad.trainer.MSFTRecVAETrainer`` object is supplied as an argument. The trainer
        is supposed to call this method and supply itself as the argument. After calling,
        ``self.msft_vae_model`` property will contain the actual model object.

        Args:
            trainer (:obj:`sad.trainer.MSFTRecVAETrainer`): A trainer that will call this
                method to initialize a VAE model.

        """
        working_dir = os.path.join(self.working_dir, self.s3_key_path)
        os.makedirs(working_dir, exist_ok=True)
        weight_file = os.path.join(working_dir, "vae_weights.hdf5")

        model = StandardVAE(
            n_users=self.n,  # Number of unique users in the training set
            original_dim=self.m,  # Number of unique items in the training set
            intermediate_dim=512,  # Se intermediate dimention to 512
            latent_dim=self.k,
            n_epochs=trainer.n_epochs,
            batch_size=trainer.generator.batch_size,
            k=self.m,
            save_path=weight_file,
            verbose=0,
            seed=np.random.randint(100000),
            drop_encoder=0.5,
            drop_decoder=0.5,
            annealing=False,
            beta=trainer.beta,
        )

        self.msft_vae_model = model

    def get_xuij(self, u_id: str, i_id: str, j_id: str, **kwargs) -> float:
        """Haven't implemented yet."""
        return 0

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
        """Save trained VAE model to a folder (``self.s3_key_path``) rooted at
        ``working_dir``. The actual saving operation will be delegated to
        ``self.msft_vae_model.model.save()``.

        Model configuration (``self.config``) will be saved too.

        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model and its configuration will be saved.

        """
        self.msft_vae_model.model.save(self.msft_vae_model.save_path)
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

        model = StandardVAE(
            n_users=self.n,  # Number of unique users in the training set
            original_dim=self.m,  # Number of unique items in the training set
            intermediate_dim=512,  # Se intermediate dimention to 512
            latent_dim=self.k,
            k=self.m,
            save_path=os.path.join(working_dir, "vae_weights.hdf5"),
            drop_encoder=0.5,
            drop_decoder=0.5,
            annealing=False,
            beta=1.0,
        )

        model.model.load_weights(os.path.join(working_dir, "vae_weights.hdf5"))
        self.msft_vae_model = model

    def load_checkpoint(self, working_dir: str, checkpoint_id: int = 1):
        """Haven't implemented this functionality yet."""
        pass

    def load_best(self, working_dir: str, criterion: str = "ll"):
        """Haven't implemented this functionality yet."""
        pass

    def reset_parameters(self):
        """Doing nothing."""
        pass

    def parameters_for_monitor(self) -> dict:
        """Return nothing."""
        return {}
