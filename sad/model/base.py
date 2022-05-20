#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy
import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from sad.task.dummy import DummyTask
from sad.utils.job import id_generator


class ModelBase(ABC):
    """The abstract model base class. It is the class that all concrete model classes
    will inherit from.

    """

    def __init__(self, config: Dict, task: "TrainingTask" = None):
        self.config = config
        self.task = DummyTask({}) if task is None else task
        self.metrics = dict()

        if not self.s3_key_path:
            now = datetime.datetime.now()
            s3_key_path = os.path.join(
                "model",
                f"{now.year}",
                f"{now.month:02d}",
                f"{now.day:02d}",
                f"model_{id_generator()}",
            )
            self.s3_key_path = s3_key_path

        self.logger = logging.getLogger(f"model.{self.__class__.__name__}")

    @property
    def config(self) -> Dict:
        """Configuration information that is used to initialize the instance."""
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config

    @property
    def spec(self) -> Dict:
        """A reference to ``"spec"`` field in ``self.config``."""
        if self.config.get("spec") is None:
            self.config["spec"] = {}
        return self.config["spec"]

    @spec.setter
    def spec(self, spec: Dict):
        self.config["spec"] = spec

    @property
    def task(self) -> "sad.task.training.TrainingTask":
        """An instance of training task associated with current model. It is the task
        instance in which a model is initialized.
        """
        return self._task

    @task.setter
    def task(self, task: "sad.task.training.TrainingTask"):
        self._task = task

    @property
    def working_dir(self) -> str:
        """Alias to ``self.task.output_dir``."""
        return self.task.output_dir

    @property
    def s3_key_path(self) -> str:
        """A S3 key uniquely assigned to a model instance. Will be setup during model's
        instantiation, and populated to ``self.spec``. It is the S3 key of the model's
        remote store if the model will be pushed to a S3 bucket."""
        return self.spec.get("s3_key_path")

    @s3_key_path.setter
    def s3_key_path(self, s3_key_path: str):
        self.spec["s3_key_path"] = s3_key_path

    @property
    def metrics(self) -> Dict:
        """A dictionary stores metrics of the model. Subject to change during model
        training by callbacks."""
        if "metrics" not in self.spec:
            self.spec["metrics"] = {}
        return self.spec["metrics"]

    @metrics.setter
    def metrics(self, metrics: Dict):
        self.spec["metrics"] = copy.deepcopy(metrics)

    @abstractmethod
    def save_checkpoint(self, working_dir: str, checkpoint_id: int):
        raise NotImplementedError

    @abstractmethod
    def save(self, working_dir: str, filename: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, working_dir: str, filename: str):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, working_dir: str, checkpoint_id: int):
        raise NotImplementedError

    @abstractmethod
    def load_best(self, working_dir: str, criterion: str):
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def parameters_for_monitor(self) -> Dict[str, float]:
        raise NotImplementedError


class ModelFactory:
    """A factory class that is responsible to create model instances."""

    logger = logging.getLogger("model.ModelFactory")
    """:class:`logging.Logger`: Class attribute for logging."""

    _registry = dict()
    """:class:`dict`: Registry dictionary containing a mapping between class name and
    class object.
    """

    @classmethod
    def register(cls, wrapped_class: ModelBase) -> ModelBase:
        """A class decorator responsible to decorate ``sad.model.ModelBase`` classes
        and register them into ``ModelFactory.registry``.
        """
        class_name = wrapped_class.__name__
        if class_name in cls._registry:
            cls.logger.warning(f"Model {class_name} already registered, ignoring.")
            return wrapped_class
        cls._registry[class_name] = wrapped_class
        return wrapped_class

    @classmethod
    def produce(cls, config: Dict, task: "TrainingTask") -> ModelBase:
        """A class method to create instances of ``sad.model.ModelBase``.

        Args:
            config (:obj:`config`): Configuration used to initialize instance object. An
                example is given below::

                    name: SADModel
                    spec:
                      n: 200
                      m: 500
                      k: 100
        """
        model_name = config.get("name")
        if model_name not in cls._registry:
            cls.logger.error(f"Unable to produce {model_name} generator.")
            raise NotImplementedError
        return cls._registry[model_name](config, task)
