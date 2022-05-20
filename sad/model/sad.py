#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
from typing import Any

import numpy as np
import scipy

from .base import ModelBase, ModelFactory


@ModelFactory.register
class SADModel(ModelBase):
    def __init__(self, config: dict, task: "TrainingTask"):
        super().__init__(config, task)
        self.initialize_params()

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
    def inner_flag(self) -> bool:
        """Whether right matrix will be non-negative."""
        return self.spec.get("inner", False)

    @property
    def T_ceiling(self) -> float:
        """The largest value of T that is allowed."""
        return 5 if self.inner_flag else 10

    def initialize_params(self):
        """Initialize user matrix ``self.XI``, left item matrix ``self.H`` and right item
        matrix ``self.T`` by drawing entries from a standard normal distribution. When
        right item matrix is assumed to be non-negative (``self.inner_flag`` is ``True``),
        ``self.T`` will be storing the logrithm of true tau matrix."""
        self.XI = np.random.normal(size=(self.k, self.n))
        self.H = np.random.normal(size=(self.k, self.m))
        # When self.inner_flag is true, the actual T is the exponential of it
        self.T = np.random.normal(size=(self.k, self.m))

    def calculate_preference_tensor(self):
        """Calculate preference tensor ``self.X`` using user and item matrices."""
        T = self.T if not self.inner_flag else np.exp(self.T)
        X1 = np.einsum("ki,kj,kl->ijl", self.XI, self.H, T)
        X2 = np.einsum("ki,kj,kl->ijl", self.XI, T, self.H)
        self.X = X1 - X2

    def calculate_probability_tensor(self):
        """Calculate probability tensor by applying logistic function to preference
        tensor ``self.X``."""
        self.Pr = 1.0 / (1 + np.exp(-self.X))

    def draw_observation_tensor(self) -> np.ndarray:
        """Draw a complete observation tensor from the generative model of SAD.

        Returns:
            :obj:`np.ndarray`: Three-way tensor with dimension ``n x m x m`` representing
            personalized preferences between item pairs.

        """
        Obs = np.zeros((self.n, self.m, self.m))
        for i in range(self.n):
            for j1 in range(self.m):
                for j2 in range(j1 + 1, self.m):
                    r = np.random.binomial(1, self.Pr[i, j1, j2])
                    if r == 0:
                        r = -1
                    Obs[i, j1, j2] = r
                    Obs[i, j2, j1] = -1 * r
        return Obs

    def get_xuij(
        self,
        u_idx: int,
        i_idx: int,
        j_idx: int,
        XI: np.ndarray = None,
        H: np.ndarray = None,
        T: np.ndarray = None,
        **kwargs,
    ) -> float:
        """Calculate preference score between two items for a particular user.
        Parameter values in current model will be used to calculate the preference score
        if no additional parameters are provided as arguments.

        Args:
            u_idx (:obj:`int`): User index, from ``0`` to ``self.n-1``.
            i_idx (:obj:`int`): Item index, from ``0`` to ``self.m-1``.
            j_idx (:obj:`int`): Item index, from ``0`` to ``self.m-1``.
            XI (:obj:`np.ndarray`): An optional user matrix. When provided, user vector
                will be taken from provided ``XI`` instead of ``self.XI``.
            H (:obj:`np.ndarray`): An optional left item matrix. When provided, left
                item vector will be taken from provided ``H`` instead of ``self.H``.
            T (:obj:`np.ndarray`): An optional right item matrix. When provided, right
                item vector will be taken from provided ``T`` instead of ``self.T``.
                Subject to exponentiation when ``self.inner_flag`` is ``True``.

        Returns:
            :obj:`float`: Preference score between ``i_idx``-th item and ``j_idx``-th
            item for ``u_idx``-th user.

        """
        if XI is None:
            XI = self.XI
        if H is None:
            H = self.H
        if T is None:
            T = self.T
        # fmt: off
        tau_j = T[:, j_idx] if not self.inner_flag else np.exp(T[:, j_idx])
        tau_i = T[:, i_idx] if not self.inner_flag else np.exp(T[:, i_idx])
        return np.sum(XI[:, u_idx] * H[:, i_idx] * tau_j) \
                - np.sum(XI[:, u_idx] * H[:, j_idx] * tau_i)
        # fmt: on

    def get_gradient_wrt_xuij(
        self, u_idx: int, i_idx: int, j_idx: int, obs_uij: int
    ) -> float:
        """
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
            (:obj:`float`): Return ``d(p)/d(x_uij)``, the gradient of log likehood with
            respect to ``x_uij``, the ``(u_idx, i_idx, j_idx)`` element in preference
            tensor.
        """
        if obs_uij == 0:  # missing data
            return 0

        o = 1 if obs_uij == 1 else 0
        xuij = self.get_xuij(u_idx=u_idx, i_idx=i_idx, j_idx=j_idx)
        g = o - scipy.special.expit(xuij)
        return g

    def gradient_update(
        self,
        u_idx: int,
        i_idx: int,
        j_idx: int,
        g: float,
        w_l2: float,
        w_l1: float,
        lr: float,
    ):
        """

        Args:
            u_idx (:obj:`int`): Index of user in user set. 0-based.
            i_idx (:obj:`int`): Index of i-th item. It is the idx of left item in
                preference tensor.
            j_idx (:obj:`int`): Index of j-th item. It is the idx of right item in
                preference tensor.
            g (:obj:`float`): The gradient of log likelihood wrt ``x_uij``.
            w_l2 (:obj:`float`): The weight of l2 regularization.
            w_l1 (:obj:`float`): The weight of l1 regularization.
            lr (:obj:`float`): Learning rate.
        """
        if g == 0:  # gradient is zero, exit
            return

        H_i = np.copy(self.H[:, i_idx])
        H_j = np.copy(self.H[:, j_idx])

        T_i = (
            np.copy(self.T[:, i_idx])
            if not self.inner_flag
            else np.exp(np.copy(self.T[:, i_idx]))
        )
        T_j = (
            np.copy(self.T[:, j_idx])
            if not self.inner_flag
            else np.exp(np.copy(self.T[:, j_idx]))
        )

        XI_u = np.copy(self.XI[:, u_idx])

        gXI_u = H_i * T_j - H_j * T_i
        self.XI[:, u_idx] += lr * (g * gXI_u - w_l2 * 2 * XI_u)
        #### XI[:, u_idx] += lr * w * gXI_u

        gH_i = XI_u * T_j
        self.H[:, i_idx] += lr * (g * gH_i - w_l2 * 2 * H_i)
        #### H[:, i_idx] += lr * w * gH_i

        gH_j = -1 * XI_u * T_i
        self.H[:, j_idx] += lr * (g * gH_j - w_l2 * 2 * H_j)
        #### H[:, j_idx] += lr * w * gH_j

        gT_i = -1 * XI_u * H_j * T_i
        self.T[:, i_idx] += lr * g * gT_i
        self.T[:, i_idx] -= lr * w_l2 * 2 * T_i  # add l2
        if w_l1 != 0:
            gT_i_reg = np.zeros(self.k)
            gT_i_reg[T_i > 1] = 1 if not self.inner_flag else T_i[T_i > 1]
            gT_i_reg[T_i < 1] = -1 if not self.inner_flag else -1 * T_i[T_i < 1]
            self.T[:, i_idx] -= lr * w_l1 * gT_i_reg

        gT_j = XI_u * H_i * T_j
        self.T[:, j_idx] += lr * g * gT_j
        self.T[:, i_idx] -= lr * w_l2 * 2 * T_i  # add l2
        if w_l1 != 0:
            gT_j_reg = np.zeros(self.k)
            gT_j_reg[T_j > 1] = 1 if not self.inner_flag else T_j[T_j > 1]
            gT_j_reg[T_j < 1] = -1 if not self.inner_flag else -1 * T_j[T_j < 1]
            self.T[:, j_idx] -= lr * w_l1 * gT_j_reg

    def log_likelihood(
        self,
        u_idx: int,
        i_idx: int,
        j_idx: int,
        obs_uij: int,
        XI: np.ndarray = None,
        H: np.ndarray = None,
        T: np.ndarray = None,
        **kwargs,
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
            XI (:obj:`np.ndarray`): An optional user matrix. When provided, user vector
                will be taken from provided ``XI`` instead of ``self.XI``.
            H (:obj:`np.ndarray`): An optional left item matrix. When provided, left item
                vector will be taken from provided ``H`` instead of ``self.H``.
            T (:obj:`np.ndarray`): An optional right item matrix. When provided, right
                item vector will be taken from provided ``T`` instead of ``self.T``.
                Subject to exponentiation when ``self.inner_flag`` is set to ``True``.

        Returns:
            (:obj:`float`): Return the contribution to the log likelihood from
            observation at ``(u_idx, i_idx, j_idx)``. Return ``0`` when the observation
            is missing.
        """
        if obs_uij == 0:  # missing data
            return 0

        o = 1 if obs_uij == 1 else 0
        xuij = self.get_xuij(u_idx=u_idx, i_idx=i_idx, j_idx=j_idx, XI=XI, H=H, T=T)
        l = (o - 1) * xuij - np.log(1 + np.exp(-1 * xuij))
        return l

    def save(self, working_dir: str = None, filename: str = "model-params.npz"):
        """Save model parameters to a file named ``"model-params.npz"`` under
        ``os.path.join(working_dir, self.s3_key_path)``."""
        if not working_dir:
            working_dir = self.working_dir
        working_dir = os.path.join(working_dir, self.s3_key_path)
        os.makedirs(working_dir, exist_ok=True)
        np.savez(
            os.path.join(working_dir, filename),
            XI=self.XI,
            H=self.H,
            T=self.T,
        )
        json.dump(
            self.config,
            open(os.path.join(working_dir, "model_config.json"), "w"),
        )

    def save_checkpoint(self, working_dir: str, checkpoint_id: int = 1):
        """Save model checkpoints to a file under
        ``os.path.join(working_dir, self.s3_key_path)``."""
        filename = f"model-params-{checkpoint_id:05d}.npz"
        self.save(working_dir=working_dir, filename=filename)

    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError

    def load(self, working_dir: str = None, filename: str = None):
        """Load model parameters.

        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model parameters are stored.
            filename (:obj:`str`): Filename containing model parameters. The full path
                of the file will be
                ``os.path.join(working_dir, self.s3_key_path, filename)``.

        """
        if not working_dir:
            working_dir = self.working_dir
        working_dir = os.path.join(working_dir, self.s3_key_path)
        if not filename:
            filename = "model-params.npz"

        file_path = os.path.join(working_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError

        params = np.load(file_path, allow_pickle=True)
        self.XI = params["XI"]
        self.H = params["H"]

    def load_checkpoint(self, working_dir: str, checkpoint_id: int = 1):
        """Load model checkpoints.

        Args:
            working_dir (:obj:`str`): The containing folder of ``self.s3_key_path``
                where model parameters are stored.
            checkpoint_id (:obj:`int`): Model parameters will be loaded from file with
                name ``"model-params-{checkpoint_id:05d}.npz"``.

        """
        filename = "model-params-{checkpoint_id:05d}.npz"
        self.load(working_dir=working_dir, filename=filename)

    def load_best(self, working_dir: str, criterion: str = "ll"):
        filename = "best-based-on-{criterion}.npz"
        self.load(working_dir=working_dir, filename=filename)

    def reset_parameters(self):
        self.initialize_params()

    def get_t_sparsity(self) -> float:
        """Extract the number of elements that are close to ``1`` in item right vectors
        ``self.T`` and return proportion. When ``self.inner_flag`` is ``True``, it is
        exponentiation of ``self.T`` will be used to calculate this number."""
        tol = 1e-1
        T = self.T if not self.inner_flag else np.exp(self.T)
        n_close = np.sum(np.abs(T - 1) < tol)
        return n_close / (self.m * self.k)

    def parameters_for_monitor(self) -> dict:
        """Extract the number of elements that are close to ``1`` in item right vectors
        ``self.T`` and return proportion. When ``self.inner_flag`` is ``True``, it is
        exponentiation of ``self.T`` will be used to calculate this number."""
        t_sparsity = self.get_t_sparsity()
        return {"t_sparisity": t_sparsity}
