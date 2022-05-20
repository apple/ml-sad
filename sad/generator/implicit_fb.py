#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy
import json
import os
import tarfile
from typing import Any, Iterator, Tuple

import cornac.data as CData
import numpy as np
import pandas as pd
import surprise
from recommenders.models.ncf.dataset import Dataset as NCFDataset

from sad.model import SADModel

from .base import GeneratorBase, GeneratorFactory

INTERACTION_FILENAME_TRN = "raw.json"
INTERACTION_FILENAME_VAL = "deleted_raw.json"

RATING_FILENAME_TRN = "raw_with_rating.json"
RATING_FILENAME_VAL = "deleted_raw_with_rating.json"


@GeneratorFactory.register
class ImplicitFeedbackGenerator(GeneratorBase):
    """A concrete generator class that handles user-item implicit feedbacks. After an
    instance of this class is created, ``self.add(filepath)`` will need to be called to
    add a local file to this generator. The format of the local file is a compressed
    tarball, containing a ``raw.json`` file, and an optionally ``raw_with_rating.json``
    file.

    The ``raw.json`` file is a dictionary mapping a user (in ``user_id``) to a list of
    items (in ``item_id``) that the user has interacted with.

    The optional ``raw_with_rating.json`` file is a nested dictionary. It is a mapping
    between a user (in ``user_id``) and items that the user has rated. The value of the
    dictionary is another dict with mapping between items (in ``item_id``) and their
    rating scores.


    """

    def __init__(self, config: dict, model: SADModel, task: "TrainingTask"):
        super().__init__(config, model, task)
        self._data_df = None
        self._cornac_dataset = None
        self._msft_ncf_dataset = None
        self._surprise_dataset = None

    @property
    def data_df(self) -> pd.DataFrame:
        """A Pandas Dataframe containing user/item pairs and ratings associated with
        them. For ``ImplicitFeedbackGenerator`` the ratings are set to ``1.0|0.0``. User
        and item IDs are under ``userID`` and ``itemID`` respectively.
        """
        if not self._data_df:
            records = []
            for u_id, u_idx in self.user_id_to_idx.items():
                inter_iidxs, nonint_iidxs = self.uidx_to_iidxs_tuple[u_idx]
                for i_idx in inter_iidxs:
                    i_id = self.item_idx_to_id[i_idx]
                    records.append({"userID": u_id, "itemID": i_id, "rating": 1})
                for i_idx in nonint_iidxs:
                    i_id = self.item_idx_to_id[i_idx]
                    records.append({"userID": u_id, "itemID": i_id, "rating": 0})
            self._data_df = pd.DataFrame(records)

        return self._data_df

    @data_df.setter
    def data_df(self, data_df: pd.DataFrame):
        self._data_df = data_df

    @property
    def cornac_dataset(self) -> CData.Dataset:
        """A Cornac Dataset object containing user/item pairs and ratings associated with
        them. Will be used for fitting models from ``cornac`` package.
        """
        if not self._cornac_dataset:
            data_df = self.data_df
            self._cornac_dataset = CData.Dataset.from_uir(
                data_df.itertuples(index=False)
            )
        return self._cornac_dataset

    @property
    def msft_ncf_dataset(self) -> NCFDataset:
        """A NCF (Neural Collaborative Filtering) Dataset object implemented in
        ``recommenders`` package from MSFT. It contains user/item pairs and ratings
        associated with them. Will be used for fitting a NCF model using
        ``recommenders`` package.
        """
        if not self._msft_ncf_dataset:
            data_df = self.data_df
            self._msft_ncf_dataset = NCFDataset(train=data_df)
        return self._msft_ncf_dataset

    @property
    def surprise_dataset(self) -> surprise.Dataset:
        """A Dataset object implemented in ``surprise`` package. It contains user/item
        pairs and ratings associated with them. Will be used for fitting a SVD model
        using ``surprise`` package.
        """
        if not self._surprise_dataset:
            data_df = self.data_df
            reader = surprise.reader.Reader(rating_scale=(0, 1))
            self._surprise_dataset = surprise.Dataset.load_from_df(
                data_df, reader
            ).build_full_trainset()
        return self._surprise_dataset

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
            inter_item_idxs, nonint_item_idxs = self.uidx_to_iidxs_tuple[u_idx]
            inter_item_idxs = np.random.choice(
                list(inter_item_idxs), i_batch, replace=True
            )
            nonint_item_idxs = np.random.choice(
                list(nonint_item_idxs), i_batch, replace=True
            )
            for i_idx, j_idx in zip(inter_item_idxs, nonint_item_idxs):
                yield (u_idx, i_idx, j_idx, 1)

    def _gen_producer_iteration(self) -> Iterator[Tuple[int, int, int, int]]:
        """A protected helper function to produce samples in ``"iteration"`` mode."""
        model = self.model
        u_shuffled = list(range(model.n))
        np.random.shuffle(u_shuffled)
        for uu_idx in range(model.n):
            u_idx = u_shuffled[uu_idx]
            inter_item_idxs, nonint_item_idxs = self.uidx_to_iidxs_tuple[u_idx]
            inter_item_idxs = list(inter_item_idxs)
            np.random.shuffle(inter_item_idxs)
            for i_idx in inter_item_idxs:
                for j_idx in np.random.choice(
                    list(nonint_item_idxs), size=self.n_negatives
                ):
                    yield (u_idx, i_idx, j_idx, 1)

    def prepare(self):
        """Instance method that will be called to inform a generator instance that all
        raw data have been added. For this class, the format of raw data is a compressed
        tarball, containing a ``raw.json`` file, and optionally, a
        ``raw_with_rating.json`` file, a ``delete_raw.json``, and
        `delete_raw_with_rating.json`. The second two files contain hold-out user-item
        interactions (and their ratings). Upon being called, following steps will be
        performed.

            1. Unzip raw data tarball. Read the ``raw.json`` and
               ``raw_with_rating.json`` file. When multiple such tarballs exist, their
               json files will be merged into one. When hold-out user-item interactions
               exist (``delete_raw.json``, and ``delete_raw_with_rating.json``), those
               interactions will be read too. Interaction data will be read to
               ``self.data_trn``, ``self.data_val`` and ``self.data_all`` fields. Data
               with ratings will be in ``self.ratings_trn``, ``self.ratings_val``, and
               ``self.ratings_all``.
            2. Create a ``self.user_idx_to_id`` and ``self.user_id_to_idx`` mapping. The
               same will be created for items.
            3. Create (optionally) ``self.tensor`` with size ``n x m x m`` containing
               personalized pairwise comparison between items. Its value takes ``-1``,
               ``1`` and ``0``, meaning first item is less preferable, more preferable
               and preference not available respectively. This tensor is only created
               when ``self.tensor_flag`` is set to ``True``. Large values of ``n`` and
               ``m`` may result memory overflow.
            4. Create ``self.uidx_to_iidxs_tuple``, a mapping between user idx to a
               tuple of two sets, with first one being interacted items and second one
               being non-interacted items, in ``item_idx``.
            5. Create ``self.user_idx_to_preference``, a mapping between user idx to
               another dictionary, with keys being a tuple of two items
               (in ``item_id``) and values being ``1``. The order of the two items in
               keys indicate their preference.


        """
        data_trn = {}  # training user-item interaction
        data_val = {}  # validation user-item interaction
        ratings_trn = {}  # training item rating
        ratings_val = {}  # validation item rating

        data_all = {}  # combined
        ratings_all = {}  # combined
        for input_file in self.input_files:
            folder = os.path.dirname(input_file)
            with tarfile.open(input_file) as tf:
                tf.extractall(folder)

            folder = input_file.replace(".tar.gz", "")
            interaction_filename_trn = os.path.join(folder, INTERACTION_FILENAME_TRN)
            data_trn.update(json.load(open(interaction_filename_trn)))
            rating_filename_trn = os.path.join(folder, RATING_FILENAME_TRN)
            ratings_trn.update(json.load(open(rating_filename_trn)))

            interaction_filename_val = os.path.join(folder, INTERACTION_FILENAME_VAL)
            if os.path.exists(interaction_filename_val):
                data_val.update(json.load(open(interaction_filename_val)))
            rating_filename_val = os.path.join(folder, RATING_FILENAME_VAL)
            if os.path.exists(rating_filename_val):
                ratings_val.update(json.load(open(rating_filename_val)))

        data_all = copy.deepcopy(data_trn)
        for u_id, i_list in data_val.items():
            data_all[u_id].extend(i_list)

        ratings_all = copy.deepcopy(ratings_trn)
        for u_id, rating_dict in ratings_val.items():
            ratings_all[u_id].update(rating_dict)

        self.data_trn = data_trn
        self.data_val = data_val
        self.data_all = data_all
        self.ratings_trn = ratings_trn
        self.ratings_val = ratings_val
        self.ratings_all = ratings_all

        user_set = sorted(data_all.keys())
        item_set = {}
        for items in data_all.values():
            for item_id in items:
                item_set[item_id] = 1
        item_set = sorted(item_set.keys())

        assert len(user_set) == self.model.n
        assert len(item_set) == self.model.m

        n = self.model.n
        m = self.model.m

        user_idx_to_id = dict(zip(range(n), user_set))
        item_idx_to_id = dict(zip(range(m), item_set))

        # a dictionary mapping user idx to interacted/noninter item idxs
        # the rest will use data_trn (not data_all) to produce samples
        data = data_trn
        ratings = ratings_trn
        uidx_to_iidxs_tuple = dict()
        if self.tensor_flag:
            tensor = np.zeros((n, m, m))
        else:
            tensor = np.ndarray(0)

        for u_idx in range(n):
            user_id = user_idx_to_id[u_idx]
            inter_items = set(data[user_id])
            inter_flag = np.array(
                [int(item_idx_to_id[i_idx] in inter_items) for i_idx in range(m)]
            )

            if self.tensor_flag:
                tensor[u_idx, :, :] = inter_flag.reshape(m, 1) - inter_flag

            uidx_to_iidxs_tuple[u_idx] = (
                set(np.where(inter_flag == True)[0]),
                set(np.where(inter_flag == False)[0]),
            )

        # create mapping between user idx to item preference pairs
        # item preference pair is stored as a map as well, with key being a tuple of a item pair
        user_idx_to_preference = {}
        if ratings:
            for u_idx in range(n):
                user_idx_to_preference[u_idx] = {}
                rated_items = ratings[user_idx_to_id[u_idx]]
                rated_items = list(zip(rated_items.keys(), rated_items.values()))
                for ii in range(len(rated_items)):
                    ii_id, ii_rating = rated_items[ii]
                    for jj in range(ii + 1, len(rated_items)):
                        jj_id, jj_rating = rated_items[jj]
                        if ii_rating > jj_rating:
                            user_idx_to_preference[u_idx][(ii_id, jj_id)] = 1
                        elif ii_rating < jj_rating:
                            user_idx_to_preference[u_idx][(jj_id, ii_id)] = 1

        self.data = data
        self.user_idx_to_id = user_idx_to_id
        self.item_idx_to_id = item_idx_to_id
        self.user_id_to_idx = dict(zip(user_idx_to_id.values(), user_idx_to_id.keys()))
        self.item_id_to_idx = dict(zip(item_idx_to_id.values(), item_idx_to_id.keys()))
        self.uidx_to_iidxs_tuple = uidx_to_iidxs_tuple
        self.tensor = tensor
        self.user_idx_to_preference = user_idx_to_preference

    def get_obs_uij(self, u_idx: int, i_idx: int, j_idx: int) -> int:
        """Get the ``(u, i, j)``-th observation from personalized three-way tensor
        ``self.tensor``. When ``self.tensor`` is pre-calculated, its value will be
        returned. Otherwise, ``self.uidx_to_iidxs_tuple`` will be used to infer the
        observation at runtime.

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
        if self.tensor_flag:
            return self.tensor[u_idx, i_idx, j_idx]
        else:
            inter_iidxs, nonint_iidxs = self.uidx_to_iidxs_tuple[u_idx]
            if ((i_idx in inter_iidxs) and (j_idx in inter_iidxs)) or (
                (i_idx in nonint_iidxs) and (j_idx in nonint_iidxs)
            ):
                return 0
            elif (i_idx in inter_iidxs) and (j_idx in nonint_iidxs):
                return 1
            elif (i_idx in nonint_iidxs) and (j_idx in inter_iidxs):
                return -1
            else:
                return None

    def get_trn(self) -> Iterator[Any]:
        pass

    def get_val_or_not(self) -> Iterator[Any]:
        pass
