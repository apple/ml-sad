#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Set, Tuple

import numpy as np

from sad.model import ModelBase


class GeneratorBase(ABC):
    """A generator base class that all concrete generator classes should inherit from.
    A generator class should also be an iterable, by implementing ``__iter__`` method.

    The way a generator works is that file(s) containing training/validation
    samples will be first added to the generator by calling ``self.add(file)``. Then
    by calling ``self.prepare()``, the generator is informed that all files have been
    added, and it is the time to get ready to iterate through the files and produce
    samples. At this point, one can use the generator in following manner::

        for features, targets in my_generator:
            # fit my model

    To only iterate through training samples, one can do::

        for features, targets in my_generator.get_trn():
            # fit my model

    Same applies to validation.
    """

    def __init__(self, config: Dict, model: ModelBase, task: "TrainingTask"):
        self.config = config
        self.model = model
        self.task = task
        self.input_files = []
        self.logger = logging.getLogger(f"generator.{self.__class__.__name__}")

    @property
    def config(self) -> Dict:
        """Configuration information that is used to initialize the generator instance."""
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config

    @property
    def spec(self) -> Dict:
        """A reference to ``"spec"`` field in ``self.config``. If such field does not
        exist or its value is ``None``, an empty dictionary will be created."""
        if self.config.get("spec") is None:
            self.config["spec"] = {}
        return self.config["spec"]

    @spec.setter
    def spec(self, spec: Dict):
        self.config["spec"] = spec

    @property
    def model(self) -> ModelBase:
        """A trainable model instance, which will be trained using samples produced by
        current generator instance."""
        return self._model

    @model.setter
    def model(self, model: ModelBase):
        self._model = model

    @property
    def task(self) -> "sad.tasks.training.TrainingTask":
        """An instance of training task associated with the generator. It is the task
        in which current generator is initialized.
        """
        return self._task

    @task.setter
    def task(self, task: "sad.tasks.training.TrainingTask"):
        self._task = task

    @property
    def mode(self):
        """:obj:`str`: The mode of how generator works. Currently supports two
        configurations: ``"random|iteration"``.

            1. ``"random"``: When working in this mode, a number of ``self.u_batch``
               random users will be selected (with replacement) from entire user set
               in an iteration. For items, a number of ``self.i_batch`` positive items
               that each user has interacted with will be randomly (with replacement)
               generated. Same number of negative items that user hasn't interacted
               with will be randomly generated as well, producing triplets of samples
               in the format of (user, item i (interacted), item j (non-interacted)).

            2. ``"iteration"``: When working in this mode, all users will be iterated
               through in a randomized order. Same to items. For each positive user-item
               interaction, a number of ``self.n_negatives`` non-interacted items will
               be randomly selected.

        """
        return self.spec.get("mode", "random")

    @property
    def u_batch(self) -> int:
        """The number of random users that will be chosen when working in ``"random"``
        mode. Read directly from ``"u_batch"`` field in ``self.spec``. When not
        configured, it will be set to 20% users."""
        u_batch = self.spec.get("u_batch")
        if not u_batch:
            n = self.model.n
            u_batch = int(0.2 * n)
        return u_batch

    @property
    def i_batch(self) -> int:
        """The number of random items that will be chosen when working in ``"random"``
        mode. Read directly from ``"i_batch"`` field in ``self.spec``. When not
        configured, it will be set to 20% items."""
        i_batch = self.spec.get("i_batch")
        if not i_batch:
            m = self.model.m
            i_batch = int(0.2 * m)
        return i_batch

    @property
    def n_negatives(self) -> int:
        """The number of negative samples will be drawn for each positive user-item
        interaction. Read directly from ``"n_negatives"`` field in ``self.spec``. Valid
        when the generator is performing in ``"iteration"`` mode. Default to five."""
        return self.spec.get("n_negatives", 5)

    @property
    def batch_size(self) -> int:
        """Batch size when generating samples in minibatch."""
        batch_size = self.spec.get("batch_size", 128)
        return batch_size

    @property
    def user_idx_to_id(self) -> Dict[int, str]:
        """A dictionary with keys being user indices from zero to ``n_users-1``, and
        values being their ids. Will be set after ``self.prepare()`` is called."""
        return self._user_idx_to_id

    @user_idx_to_id.setter
    def user_idx_to_id(self, user_idx_to_id: Dict[int, str]):
        self._user_idx_to_id = user_idx_to_id

    @property
    def user_id_to_idx(self) -> Dict[str, int]:
        """A dictionary with keys being user id and values being the index. It is the
        inverse mapping of ``self.user_idx_to_id``."""
        return self._user_id_to_idx

    @user_id_to_idx.setter
    def user_id_to_idx(self, user_id_to_idx: Dict[str, int]):
        self._user_id_to_idx = user_id_to_idx

    @property
    def item_idx_to_id(self) -> Dict[int, str]:
        """A dictionary with keys being item indices from zero to ``n_items-1``, and
        values being their ids. Will be set after ``self.prepare()`` is called."""
        return self._item_idx_to_id

    @item_idx_to_id.setter
    def item_idx_to_id(self, item_idx_to_id: Dict[int, str]):
        self._item_idx_to_id = item_idx_to_id

    @property
    def item_id_to_idx(self) -> Dict[str, int]:
        """A dictionary with keys being item id and values being the index. It is the
        inverse mapping of ``self.item_idx_to_id``."""
        return self._item_id_to_idx

    @item_id_to_idx.setter
    def item_id_to_idx(self, item_id_to_idx: Dict[str, int]):
        self._item_id_to_idx = item_id_to_idx

    @property
    def uidx_to_iidxs_tuple(self) -> Dict[int, Tuple[Set[int], Set[int]]]:
        """A dictionary mapping from user idx to a tuple in which the first
        element is a set of item idxs the user has interacted with, and the second one
        is a set of non-interacted item idxs. Will be set after ``self.prepare()`` is
        called."""
        return self._uidx_to_iidxs_tuple

    @uidx_to_iidxs_tuple.setter
    def uidx_to_iidxs_tuple(
        self, uidx_to_iidxs_tuple: Dict[int, Tuple[Set[int], Set[int]]]
    ):
        self._uidx_to_iidxs_tuple = uidx_to_iidxs_tuple

    @property
    def data(self) -> Dict[str, List[str]]:
        """A dictionary with keys being user ids and values being a list of item ids
        that user has interacted with. Lists of complete users and items will be inferred
        from it. Will be set after ``self.prepare()`` is called.
        """
        return self._data

    @data.setter
    def data(self, data: dict):
        self._data = data

    @property
    def tensor(self) -> np.ndarray:
        """A three way array with shape of ``n x m x m`` where ``n`` is the number of
        users, and ``m`` is the number of items. A value of ``1`` at location
        ``(u, i, j)`` suggests ``u``-th user prefers ``i``-th item over ``j``-th item.
        ``-1`` suggests the opposite. A value of ``0`` means no information available to
        determine the preference of the two items. Value will be optionally set after
        ``self.prepare()`` is called, depending on the value of ``self.tensor_flag``,
        for the purpose of saving memory.
        """
        return self._tensor

    @tensor.setter
    def tensor(self, tensor: np.ndarray):
        self._tensor = tensor

    @property
    def tensor_flag(self) -> bool:
        """A boolean flag to indicate if three way data tensor ``self.tensor`` will be
        constructed. ``False`` will stop creating the tensor to save memory consumption.
        """
        return self.spec.get("tensor_flag", True)

    @property
    def user_idx_to_preference(self) -> Dict[int, Dict[Tuple[str, str], int]]:
        """A dictionary contains a mapping between user idx and item pairs that the user
        prefer one over the other. The item pairs are stored in a dictionary as well,
        with key being a tuple of two item ids, and value being ``1``."""
        return self._user_idx_to_preference

    @user_idx_to_preference.setter
    def user_idx_to_preference(
        self, user_idx_to_preference: Dict[int, Dict[Tuple[str, str], int]]
    ):
        self._user_idx_to_preference = user_idx_to_preference

    @property
    def input_files(self) -> List[str]:
        """A list of files from where samples will be read."""
        return self._input_files

    @input_files.setter
    def input_files(self, input_files: List[str]):
        self._input_files = input_files

    @property
    def output_dir(self) -> str:
        """Read directly from ``self.task.output_dir``."""
        return self.task.output_dir

    @property
    def input_dir(self) -> str:
        """Read directly from ``self.task.input_dir``."""
        return self.task.input_dir

    @abstractmethod
    def prepare(self):
        """A method to inform generator to setup things in order to be prepared for
        generating samples. Concrete subclasses are responsible to implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trn(self) -> Iterator[Any]:
        """Interface to generator samples for model training.

        Returns:
            :obj:`Iterator[Any]`: An iterable that training samples will be iterated
            through in mini-batches.

        """
        raise NotImplementedError

    @abstractmethod
    def get_val_or_not(self) -> Iterator[Any]:
        """Interface to generator samples for validating model.

        Returns:
            :obj:`Iterator[Any]`: An iterable that validation samples will be iterated
            through in mini-batches.
        """
        raise NotImplementedError

    def add(self, filename: str):
        """A method to add a local file to generator. The local file contains data from
        which mini-batches of training/validation samples will be read.

        Args:
            filename (:obj:`str`): A file path pointing the file.

        """
        if not os.path.exists(filename):
            self.logger.warning(
                f"Unable to add {filename} to generator, file does not exist."
            )
            return
        self.input_files.append(filename)

    def save(self, working_dir: str):
        """Save generator's configuration to a folder.

        Args:
            working_dir (:obj:`str`): A local path where the configuration of the
                generator will be saved.

        """
        if not working_dir:
            working_dir = self.output_dir
        model_s3_key_path = self.model.s3_key_path
        filename = "generator_config.json"
        os.makedirs(os.path.join(working_dir, model_s3_key_path), exist_ok=True)
        with open(os.path.join(working_dir, model_s3_key_path, filename), "w") as fout:
            json.dump(self.config, fout)


class GeneratorFactory:
    """A factory class that is responsible to create generator instances."""

    logger = logging.getLogger("generator.GeneratorFactory")
    """:obj:`logging.Logger`: Class attribute for logging."""

    _registry = dict()
    """:obj:`dict`: Registry dictionary containing a mapping between class name and
    class object."""

    @classmethod
    def register(cls, wrapped_class: GeneratorBase) -> GeneratorBase:
        class_name = wrapped_class.__name__
        if class_name in cls._registry:
            cls.logger.warning(f"Generator {class_name} already registered, Ignoring.")
            return wrapped_class
        cls._registry[class_name] = wrapped_class
        return wrapped_class

    @classmethod
    def produce(
        cls, config: Dict, model: ModelBase, task: "TrainingTask"
    ) -> GeneratorBase:
        generator_name = config.get("name")
        if generator_name not in cls._registry:
            cls.logger.error(f"Unable to produce {generator_name} generator.")
            raise NotImplementedError
        return cls._registry[generator_name](config, model, task)
