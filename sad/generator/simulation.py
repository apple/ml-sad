#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os
import tarfile
from typing import Any, Iterator, Tuple

import numpy as np
from numpy.random import RandomState

from sad.model import SADModel

from .base import GeneratorBase, GeneratorFactory


@GeneratorFactory.register
class SimulationGenerator(GeneratorBase):
    """A concrete generator class that handles simulated data from the generative model
    of ``SAD``. After an instance of this class is created, ``self.add(filepath)`` will
    need to be called to add a local file to this generator. The format of the local
    file is a compressed tarball, containing a ``raw.npz`` file, inside which true model
    parameters ``XI0``, ``T0`` ``H0`` and ``X0`` (derived from the first three matrices)
    are contained. An observation tensor ``Obs0`` is in the raw file as well, containing
    a fully observed personalized pairwise comparision taking values of ``-1`` or ``1``.

    One can set ``self.missing_ratio`` to control the percentage of missing data in the
    observation. Details see below.

    """

    def __init__(self, config: dict, model: SADModel, task: "TrainingTask"):
        super().__init__(config, model, task)

    @property
    def XI0(self) -> np.ndarray:
        """The true user matrix (``k x n``) containing user vectors as columns."""
        return self._XI0

    @XI0.setter
    def XI0(self, XI0: np.ndarray):
        self._XI0 = XI0

    @property
    def H0(self) -> np.ndarray:
        """The true left item matrix (``k x m``) containing item left vectors as columns."""
        return self._H0

    @H0.setter
    def H0(self, H0: np.ndarray):
        self._H0 = H0

    @property
    def T0(self) -> np.ndarray:
        """The true right item matrix (``k x m``) containing item right vectors as columns."""
        return self._T0

    @T0.setter
    def T0(self, T0: np.ndarray):
        self._T0 = T0

    @property
    def X0(self) -> np.ndarray:
        """The three way tensor (``n x m x m``) containing true preference scores."""
        return self._X0

    @X0.setter
    def X0(self, X0: np.ndarray):
        self._X0 = X0

    @property
    def Obs0(self) -> np.ndarray:
        """Three way tensor containing observations. An alias to ``self.tensor``."""
        return self.tensor

    @property
    def missing_ratio(self) -> float:
        """Proportion of missing entries in ``self.Obs0``. Default to ``0`` meaning no
        observation is missing. Will read directly from ``"missing_ratio"`` field in
        ``self.spec``. Missing entries in ``self.Obs0`` will be set to ``0`` when
        ``self.prepare()`` is invoked."""
        return self.spec.get("missing_ratio", 0)

    @property
    def ll0(self) -> float:
        """The log likelihood of non-missing observations under true parameter values.
        Its value will be set after running ``self.prepare()``."""
        return self._ll0

    @ll0.setter
    def ll0(self, ll0: float):
        self._ll0 = ll0

    @property
    def rnd_seed(self) -> int:
        """Random seed. Used for reproducibility purposes. Will read directly from
        ``"rnd_seed"`` field from ``self.spec``."""
        return self.spec.get("rnd_seed", 10203)

    def __iter__(self) -> Iterator[Tuple[int, int, int, int]]:
        return self._gen_producer()

    def _gen_producer(self) -> Iterator[Tuple[int, int, int, int]]:
        mode = self.mode
        if mode == "random":
            return self._gen_producer_random()
        elif mode == "iteration":
            return self._gen_producer_iteration()

    def _gen_producer_random(self) -> Iterator[Tuple[int, int, int, int]]:
        """A protected helper function to produce samples in ``"random"`` mode."""
        model = self.model
        u_batch = self.u_batch
        i_batch = self.i_batch
        for u_idx in np.random.choice(model.n, u_batch, replace=True):
            ii_idxs = np.random.choice(model.m, i_batch, replace=True)
            jj_idxs = np.random.choice(model.m, i_batch, replace=True)
            for i_idx, j_idx in zip(ii_idxs, jj_idxs):
                obs = self.tensor[u_idx, i_idx, j_idx]
                yield (u_idx, i_idx, j_idx, obs)

    def _gen_producer_iteration(self) -> Iterator[Tuple[int, int, int, int]]:
        """A protected helper function to produce samples in ``"iteration"`` mode."""
        model = self.model
        u_shuffled = list(range(model.n))
        np.random.shuffle(u_shuffled)
        i_shuffled = list(range(model.m))
        np.random.shuffle(i_shuffled)
        for uu_idx in range(model.n):
            u_idx = u_shuffled[uu_idx]
            for ii_idx in range(model.m):
                i_idx = i_shuffled[ii_idx]
                for jj_idx in range(ii_idx + 1, model.m):
                    j_idx = i_shuffled[jj_idx]
                    obs = self.tensor[u_idx, i_idx, j_idx]
                    yield (u_idx, i_idx, j_idx, obs)

    def prepare(self):
        """Instance method that will be called to inform a generator that all raw data
        have been added. For this class, the format of raw data is a compressed tarball,
        containing a ``raw.npz`` file. Upon being called, following steps will be
        performed. For this class only one raw data file is allowed to be added to
        the generator.

            1. Unzip raw data tarball. Read true parameter values from ``raw.npz`` file,
               set corresponding attributes of current generator.
            2. Create a ``self.user_idx_to_id`` and ``self.user_id_to_idx`` mapping. The
               same will be created for items.
            3. Randomly set certain proportion of observations to ``0``, suggesting data
               are missing. In the meanwhile, calculate log likelihood of observed
               entries under true parameter values.
            4. Create ``self.user_idx_to_preference``, a mapping between user idx to
               another dictionary, with keys being a tuple of two items
               (in ``item_id``) and values being ``1``. The order of the two items in
               keys indicate their preference.


        """

        assert len(self.input_files) == 1
        input_file = self.input_files[0]
        folder = os.path.dirname(input_file)
        with tarfile.open(input_file) as tf:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, folder)
        input_file = os.path.join(input_file.replace(".tar.gz", ""), "raw.npz")
        data = np.load(input_file)

        XI0, H0, T0, X0, Obs0 = (
            data["XI0"],
            data["H0"],
            data["T0"],
            data["X0"],
            data["Obs0"],
        )
        k, n = XI0.shape
        _, m = H0.shape

        assert n == self.model.n
        assert m == self.model.m

        user_set = list(range(n))
        item_set = list(range(m))

        user_idx_to_id = dict(zip(range(n), user_set))
        item_idx_to_id = dict(zip(range(m), item_set))

        ll0 = 0
        rng = RandomState(self.rnd_seed)
        for u_idx in range(n):
            for i_idx in range(m):
                for j_idx in range(i_idx + 1, m):
                    coin = 0
                    if self.missing_ratio > 0:
                        coin = rng.binomial(1, self.missing_ratio)
                    if coin:
                        Obs0[u_idx, i_idx, j_idx] = 0
                        Obs0[u_idx, j_idx, i_idx] = 0
                    else:
                        o = Obs0[u_idx, i_idx, j_idx]
                        xuij = X0[u_idx, i_idx, j_idx]
                        ll0 += (o - 1) * xuij - np.log(1 + np.exp(-1 * xuij))

        # create mapping between user idx to item preference pairs
        # item preference pair is stored as a map as well, with key being a tuple of a item pair
        user_idx_to_preference = {}
        for u_idx in range(n):
            user_idx_to_preference[u_idx] = {}
            for i_idx in range(m):
                i_id = item_idx_to_id[i_idx]
                for j_idx in range(i_idx + 1, m):
                    j_id = item_idx_to_id[j_idx]
                    obs = Obs0[u_idx, i_idx, j_idx]
                    if obs == 1:
                        user_idx_to_preference[u_idx][(i_id, j_id)] = 1
                    elif obs == -1:
                        user_idx_to_preference[u_idx][(j_id, i_id)] = 1

        self.data = data
        self.user_idx_to_id = user_idx_to_id
        self.item_idx_to_id = item_idx_to_id
        self.user_id_to_idx = dict(zip(user_idx_to_id.values(), user_idx_to_id.keys()))
        self.item_id_to_idx = dict(zip(item_idx_to_id.values(), item_idx_to_id.keys()))
        self.XI0 = XI0
        self.H0 = H0
        self.T0 = T0
        self.X0 = X0
        self.ll0 = ll0
        self.tensor = Obs0
        self.user_idx_to_preference = user_idx_to_preference

    def get_obs_uij(self, u_idx: int, i_idx: int, j_idx: int) -> int:
        """Get the ``(u, i, j)``-th observation from observation tensor ``self.Obs0``.

        Args:
            u_idx (:obj:`int`): The user idx.
            i_idx (:obj:`int`): Index of first item in comparison.
            j_idx (:obj:`int`): Index of second item in comparison.

        Returns:
            :obj:`int`: A value from ``(-1, 1, 0)`` indicating the personalized
            preference of the two items. ``1`` indicates ``i_idx``-th item is preferable
            than ``j_idx``-th; ``-1`` suggests otherwise; ``0`` indicate such information
            is not available.

        """

        return self.tensor[u_idx, i_idx, j_idx]

    def get_trn(self) -> Iterator[Any]:
        pass

    def get_val_or_not(self) -> Iterator[Any]:
        pass
