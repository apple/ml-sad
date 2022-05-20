#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy
import logging
from abc import abstractmethod
from typing import Dict

from sad.callback import CallerProtocol
from sad.generator import GeneratorBase
from sad.model import ModelBase
from sad.utils.misc import update_dict_recursively

RAND_UPPER = 10000
RAND_LOWER = 100


class TrainerBase(CallerProtocol):
    """The abstract trainer base class. It is the class that all concrete trainer classes
    will inherit from.

    In the meanwhile, this class is complaint of ``sad.callback.CallerProtocol``.
    """

    def __init__(
        self,
        config: Dict,
        model: ModelBase,
        generator: GeneratorBase,
        task: "TrainingTask",
    ):
        """Base __init__ method."""
        self.config = config
        self.model = model
        self.generator = generator
        self.task = task
        self.stop = False
        self.initialize_callback()
        self.logger = logging.getLogger(f"trainer.{self.__class__.__name__}")

    @property
    def config(self) -> Dict:
        """Configuration information that is used to initialize the instance."""
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = copy.deepcopy(config)

    @property
    def spec(self) -> Dict:
        """A reference to ``"spec"`` field in ``self.config``. When no such a field
        available or the value is ``None``, an empty dictionary will be set."""
        if self.config.get("spec") is None:
            self.config["spec"] = {}
        return self.config["spec"]

    @spec.setter
    def spec(self, spec: Dict):
        self.config["spec"] = spec

    @property
    def model(self) -> ModelBase:
        """A reference to a model instance. This model will be trained during training
        loop by current trainer."""
        return self._model

    @model.setter
    def model(self, model: ModelBase):
        self._model = model

    @property
    def generator(self) -> GeneratorBase:
        """A reference to a generator instance, which will be used by current trainer to
        perform a training task."""
        return self._generator

    @generator.setter
    def generator(self, generator: GeneratorBase):
        self._generator = generator

    @property
    def task(self) -> "TrainingTask":
        """A reference to an instance of training task associated with current trainer.
        It is the task instance in which a trainer is initialized.
        """
        return self._task

    @task.setter
    def task(self, task: "TrainingTask"):
        self._task = task

    @property
    def stop(self):
        """:obj:`boolean`: A flag to indicate whether to stop training. Subject to
        changes during training by callbacks."""
        return self._stop

    @stop.setter
    def stop(self, stop: bool):
        self._stop = stop

    @property
    def lr(self):
        """:obj:`float`: Read directly from ``self.spec``. Learning rate. Subject to
        changes during training by callbacks."""
        lr = self.spec.get("lr", 0.1)
        return lr

    @lr.setter
    def lr(self, lr: float):
        self.spec["lr"] = lr

    @property
    def n_iters(self) -> int:
        """The number of iterations that will happen in a trainer. Set to be an alias
        to ``self.n_epochs``."""
        return self.n_epochs

    @property
    def n_epochs(self) -> int:
        """The number of epochs during training, specific to ``TrainerBase``. Will read
        directly from ``"n_epochs"`` field in ``self.spec``."""
        return self.spec.get("n_epochs", 1)

    @property
    def working_dir(self):
        """:obj:`str`: Read directly from ``self.task.output_dir``."""
        return self.task.output_dir

    @abstractmethod
    def train(self):
        """The main training loop. Concrete trainer classes are responsible to provide
        implementations of their training logic."""
        raise NotImplementedError

    @property
    def eval_at_every_step(self):
        """:obj:`int`: Read directly from ``self.spec``. A number to indicate how many
        steps log likelihood will be evaluated. A negative number means do not evaluate
        at step level."""
        return self.spec.get("eval_at_every_step", -1)

    @abstractmethod
    def save(self, working_dir: str):
        """Save an intance of trainer for later usage."""
        raise NotImplementedError

    @abstractmethod
    def load(self, working_dir: str):
        """Load states of an trainer intance; mostly for continue the training loop of
        a saved model."""
        raise NotImplementedError

    def add_final_metrics_to_model_metrics(self, **kwargs):
        """Class specific method to add final metrics to model's metrics attribute. After
        addition, model's metrics will include ``"final"`` field with structure shown
        below::

            metrics = {
                "final": {
                    "ll": float,
                    "t_sparsity": float,
                },
            }

        """
        ll = kwargs.get("ll")
        t_sparsity = kwargs.get("t_sparsity")
        final_metrics = {
            "ll": ll,
            "t_sparsity": t_sparsity,
        }
        metrics = self.model.metrics
        self.model.metrics = update_dict_recursively(metrics, {"final": final_metrics})

    def on_loop_end(self, **kwargs):
        """Will be invoked at the end of training loop. Save trainer instance to
        ``self.working_dir``, and save ``self.model``, ``self.generator`` in the
        meanwhile.

        This method overwrites ``on_loop_end`` in ``sad.callback.CallerProtocol``.
        """
        for callback in self.callbacks:
            callback.on_loop_end(**kwargs)
        self.add_final_metrics_to_model_metrics(**kwargs)
        self.save(self.working_dir)
        self.generator.save(self.working_dir)
        self.model.save(self.working_dir)


class TrainerFactory:
    """A factory class that is responsible to create trainer instances."""

    logger = logging.getLogger("trainer.TrainerFactory")
    """:class:`logging.Logger`: Class attribute for logging."""

    _registry = dict()
    """:class:`dict`: Registry dictionary containing a mapping between class name to 
    class object."""

    @classmethod
    def register(cls, wrapped_class: TrainerBase) -> TrainerBase:
        """A class level decorator responsible to decorate ``sad.trainer.TrainerBase``
        classes and register them into ``TrainerFactory.registry``.
        """
        class_name = wrapped_class.__name__
        if class_name in cls._registry:
            cls.logger.warning(f"Trainer {class_name} already registered, Ignoring.")
            return wrapped_class
        cls._registry[class_name] = wrapped_class
        return wrapped_class

    @classmethod
    def produce(
        cls,
        config: Dict,
        model: ModelBase,
        generator: GeneratorBase,
        task: "TrainingTask",
    ) -> TrainerBase:
        """A class level method to initialize instances of ``sad.trainer.TrainerBase``
        classes.

        Args:
            config (:obj:`config`): Configuration used to initialize instance object. An
                example is given below::

                    name: "SGDTrainer"
                    spec:
                        w_l1: 0.01
                        w_l2: 0.01:
                        ...

            model (:obj:`sad.model.ModelBase`): An instance of model, a trainable that a
                trainer will train.
            generator (:obj:`sad.generator.GeneratorBase`): An instance of generator,
                from which training and validation data are generated.
            task (:obj:`sad.tasks.training.TrainingTask`): An instance of training
                task, from which a trainer is created.
        """
        trainer_name = config.get("name")
        if trainer_name not in cls._registry:
            cls.logger.error(f"Unable to produce {trainer_name} trainer.")
            raise NotImplementedError
        return cls._registry[trainer_name](config, model, generator, task)
