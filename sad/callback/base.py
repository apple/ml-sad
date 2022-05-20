#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import logging
from abc import ABC, abstractmethod
from typing import Dict

from .caller import CallerProtocol


class CallbackBase(ABC):
    """A callback base class that every concrete callback subclass will inherit from.

    Instance of this class will be managed by a caller instance that is compliant with
    ``CallerProtocol``. Currently instances of ``sad.trainer.TrainerBase`` classes could
    be such callers. Callback instances will be created during caller's initialization.
    Configurations for this callback is provided under
    ``caller:spec:callbacks:``. An example is shown below::

        trainer:
          name: SGDTrainer
          spec:
            n_iters: 20
            w_l1: 0.1
            w_l2: 0.0
            u_idxs: [0, 1, 2, 3]
            callbacks:
            - name: "CheckpointingCallback"
              spec:
                start: 10
                every: 1

    """

    def __init__(self, config: Dict, caller: CallerProtocol):
        self.config = config
        self.caller = caller
        self.logger = logging.getLogger(f"callback.{self.__class__.__name__}")

    @property
    def config(self) -> Dict:
        """Configuration dictionary that is used to initialize the instance."""
        return self._config

    @config.setter
    def config(self, config: dict):
        self._config = config

    @property
    def spec(self) -> Dict:
        """A reference to ``"spec"`` field in ``self.config``. When no such field exists
        or the value is ``None``, an empty dictionary will be set."""
        if self.config.get("spec") is None:
            self.config["spec"] = {}
        return self.config.get("spec")

    @spec.setter
    def spec(self, spec: Dict):
        self.config["spec"] = spec

    @property
    def caller(self) -> CallerProtocol:
        """Reference to an instance of a caller class that is compliant with
        ``CallerProtocol``. Could be an instance of ``sad.trainer.TrainerBase``."""
        return self._caller

    @caller.setter
    def caller(self, caller: CallerProtocol):
        self._caller = caller
        self._caller.register_callback(self)

    @abstractmethod
    def on_loop_begin(self, **kwargs):
        """Will be called from caller when main loop begins. The main loop could be
        training loop in ``sad.trainer.TrainerBase``."""
        raise NotImplementedError

    @abstractmethod
    def on_loop_end(self, **kwargs):
        """Will be called from caller when main loop ends."""
        raise NotImplementedError

    @abstractmethod
    def on_iter_begin(self, iter_idx: int, **kwargs):
        """Will be called from caller when an iteration begins. An iteration could be
        an epoch during training loop.

        Args:
            iter_idx (:obj:`int`): The index of iteration, 0-based.

        """
        raise NotImplementedError

    @abstractmethod
    def on_iter_end(self, iter_idx: int, **kwargs):
        """Will be called from caller when an iteration ends.

        Args:
            iter_idx (:obj:`int`): The index of iteration. 0-based.

        """
        raise NotImplementedError

    @abstractmethod
    def on_step_begin(self, iter_idx: int, step_idx: int, **kwargs):
        """Will be called from caller when a step begins. A step could be one gradient
        updates from a minibatch during training.

        Args:
            iter_idx (:obj:`int`): The index of iteration. 0-based.
            step_idx (:obj:`int`): The index of step. 0-based.

        """
        raise NotImplementedError

    @abstractmethod
    def on_step_end(self, iter_idx: int, step_idx: int, **kwargs):
        """Will be called from caller when a step finishes.

        Args:
            iter_idx (:obj:`int`): The index of iteration. 0-based.
            step_idx (:obj:`int`): The index of step. 0-based.

        """
        raise NotImplementedError


class CallbackFactory:
    """A factory class that is responsible to create callback instances."""

    logger = logging.getLogger("callback.CallbackFactory")
    """:class:`logging.Logger`: Class attribute for logging."""

    _registry = dict()
    """:class:`dict`: Registry dictionary containing a mapping between class name and
    class object."""

    @classmethod
    def register(cls, wrapped_class: CallbackBase) -> CallbackBase:
        """A class level decorator responsible to decorate ``sad.callback.CallbackBase``
        classes and register them into ``CallbackFactory._registry``.
        """
        class_name = wrapped_class.__name__
        if class_name in cls._registry:
            cls.logger.warning(f"Callback {class_name} already registered, Ignoring.")
            return wrapped_class
        cls._registry[class_name] = wrapped_class
        return wrapped_class

    @classmethod
    def produce(cls, config: Dict, caller: CallerProtocol) -> CallbackBase:
        """A class method to initialize instances of ``sad.callback.CallbackBase``.

        Args:
            config (:obj:`config`): Configuration used to initialize an instance object.
                An example is given below::

                    name: "EarlyStoppingCallback"
                    spec:
                        allow_incomplete_epoch: False

            caller (:obj:`sad.callback.CallerProtocol`): An instance of a class that
                is compliant with ``CallerProtocol``. Currently
                ``sad.trainer.TrainerBase`` is of this class type. A callback
                instance will be created with its caller. During caller's loop, callback
                methods will be invoked.
        """
        callback_name = config.get("name")
        if callback_name not in cls._registry:
            cls.logger.error(f"Unable to produce {callback_name} callback.")
            raise NotImplementedError
        return cls._registry[callback_name](config, caller)
